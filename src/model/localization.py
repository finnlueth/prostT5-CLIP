import random

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    T5Config,
    T5EncoderModel,
    Trainer,
    modeling_outputs,
    utils,
    modeling_utils,
)
from transformers.models.clip import CLIPModel
from transformers.models.t5 import T5EncoderModel
from transformers.models.t5.modeling_t5 import (
    T5EncoderModel,
    T5ClassificationHead,
    T5Config,
)


class LightAttentionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.eval_sample_size = kwargs.pop("eval_sample_size", 32)
        super().__init__(*args, **kwargs)

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Samples the evaluation dataset and returns a subset of size self.eval_sample_size.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        # Use random subset of eval dataset
        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        ).select(random.sample(range(len(self.eval_dataset)), self.eval_sample_size))
        data_collator = self.data_collator

        if utils.is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)


def trim_attention_mask(attention_mask, trim_beginning=0, trim_end=0):
    """
    Finds indices of first n and last m 1s in attention mask and sets them to 0.
    Vectorized implementation.
    Args:
        attention_mask: tensor of shape (batch_size, seq_length)
        trim_beginning: number of 1s to trim from beginning
        trim_end: number of 1s to trim from end
    Returns:
        Modified attention mask with first n and last m 1s set to 0
    """
    if trim_beginning == 0 and trim_end == 0:
        return attention_mask

    attention_mask = attention_mask.clone()

    cumsum_forward = torch.cumsum(attention_mask, dim=1)

    cumsum_backward = torch.cumsum(attention_mask.flip(dims=[1]), dim=1).flip(dims=[1])

    if trim_beginning > 0:
        beginning_mask = cumsum_forward > trim_beginning
        attention_mask = attention_mask * beginning_mask

    if trim_end > 0:
        end_mask = cumsum_backward > trim_end
        attention_mask = attention_mask * end_mask

    return attention_mask


# From Hannes
class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)
        self.linear = nn.Sequential(nn.Linear(2 * embeddings_dim, 32), nn.Dropout(dropout), nn.ReLU(), nn.BatchNorm1d(32))
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class LightAttentionPLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.plm = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=config.plm,
            device_map=config.device,
            torch_dtype="auto",
        )

        self.decoder = LightAttention(
            output_dim=config.light_attention["output_dim"],
            dropout=config.light_attention["dropout"],
            kernel_size=config.light_attention["kernel_size"],
        )
        self.decoder.to(config.device)

        for name, init_func in modeling_utils.TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)
        self.post_init()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        x = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x = x.last_hidden_state[:, :1, :]
        x = x.transpose(1, 2)

        attention_mask = trim_attention_mask(attention_mask, trim_beginning=0, trim_end=1)
        x = self.decoder(x=x, mask=attention_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(x.view(-1, x.size(-1)), labels.view(-1))

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss if labels is not None else None,
            logits=x,
            hidden_states=None,
            attentions=None,
        )

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
            f"|| trainable%: {100 * trainable_params / all_param:.2f}%"
        )


class LightAttentionCLIP(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.decoder = LightAttention(
            output_dim=config.light_attention["output_dim"],
            dropout=config.light_attention["dropout"],
            kernel_size=config.light_attention["kernel_size"],
        )
        self.decoder.to(config.device)

        self.plm = CLIPModel.from_pretrained(
            pretrained_model_name_or_path=config.plm,
            device_map=config.device,
            torch_dtype="auto",
        )

        self.post_init()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        x = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, :1, :]
        x = x.transpose(1, 2)

        attention_mask = trim_attention_mask(attention_mask, trim_beginning=0, trim_end=1)
        x = self.light_attention(x=x, mask=attention_mask)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(x.view(-1, x.size(-1)), labels.view(-1))

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=x,
            hidden_states=None,
            attentions=None,
        )

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
            f"|| trainable%: {100 * trainable_params / all_param:.2f}%"
        )


# class LinearPLM(PreTrainedModel):
#     def __init__(
#         self,
#         config: T5Config,
#         custom_num_labels,
#         custom_dropout_rate,
#     ):
#         super().__init__(config)

#         self.plm = T5EncoderModel.from_pretrained(
#             pretrained_model_name_or_path=config.plm,
#             device_map=config.device,
#             torch_dtype="auto",
#         )
#         self.decoder = T5ClassificationHead(config)

#         self.post_init()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         x = self.encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         attention_mask = trim_attention_mask(attention_mask, trim_beginning=0, trim_end=1)
#         x = self.decoder(x)

#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + encoder_outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#         return modeling_outputs.SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#         )

#     def print_trainable_parameters(self):
#         """
#         Prints the number of trainable parameters in the model.
#         """
#         trainable_params = 0
#         all_param = 0
#         for _, param in self.named_parameters():
#             all_param += param.numel()
#             if param.requires_grad:
#                 trainable_params += param.numel()
#         print(
#             f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
#             f"|| trainable%: {100 * trainable_params / all_param:.2f}%"
#         )


# class LinearCLIP(PreTrainedModel):
#     pass
