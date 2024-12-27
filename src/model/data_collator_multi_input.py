from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


@dataclass
class DataCollatorForProtT5CLIP:
    tokenizer_plm: PreTrainedTokenizerBase
    tokenizer_llm: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        print("------------------------------- data collator -------------------------------")
        features_plm = {'input_ids': [], 'attention_mask': []}
        features_llm = {'input_ids': [], 'attention_mask': []}
        
        for feature in features:
            # print(feature)
            features_plm['input_ids'].append(feature['input_ids_sequence'])
            features_plm['attention_mask'].append(feature['attention_mask_sequence'])
            features_llm['input_ids'].append(feature['input_ids_text'])
            features_llm['attention_mask'].append(feature['attention_mask_text'])

        print([len(x) for x in features_plm['input_ids']])
        print([len(x) for x in features_llm['input_ids']])

        batch_plm = pad_without_fast_tokenizer_warning(
            self.tokenizer_plm,
            encoded_inputs=features_plm,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch_llm = pad_without_fast_tokenizer_warning(
            self.tokenizer_llm,
            encoded_inputs=features_llm,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        print(batch_plm['input_ids'].shape)
        print(batch_llm['input_ids'].shape)

        return {
            "input_ids_sequence": batch_plm["input_ids"],
            "input_ids_text": batch_llm["input_ids"],
            "attention_mask_sequence": batch_plm["attention_mask"],
            "attention_mask_text": batch_llm["attention_mask"],
        }
