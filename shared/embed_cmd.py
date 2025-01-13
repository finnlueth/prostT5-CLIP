import gc
import os
import re
from pathlib import Path

import h5py
import pandas as pd
import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    T5Tokenizer,
)

from model.configuration_protein_clip import ProtT5CLIPConfig
from model.modeling_protein_clip import ProtT5CLIP


def process_sequence(sequence):
    """Process a protein sequence for tokenization"""
    return " ".join(list(sequence))


def read_input_file(input_file, mode, split=None):
    """Read input file based on mode and split"""
    if mode == "protein":
        if os.path.isdir(input_file):
            dataset = load_from_disk(input_file)
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in the dataset.")
            sequences = dataset[split]["sequence"]
            idents = dataset[split]["identifier"]
            return idents, sequences
        else:
            raise ValueError(
                "Input for mode 'protein' must be a HuggingFace dataset directory."
            )
    elif mode == "go_terms":
        data = pd.read_csv(input_file, sep="\t")
        identifiers = data["term"].tolist()
        texts = data["sentence"].tolist()
        return identifiers, texts
    else:
        raise ValueError("Unsupported mode. Choose either 'protein' or 'go_terms'.")


'''
def simple_vectorize(sequences=None, texts=None):
    """Dummy vectorization using hash-based encoding"""
    vectors = []
    max_length = 100  # Fixed size for vectors
    if sequences:
        for seq in sequences:
            vector = np.array([hash(char) % 1000 for char in seq])
            vector = np.pad(vector, (0, max(0, max_length - len(vector))), mode='constant')[:max_length]
            vectors.append(vector)
    elif texts:
        for text in texts:
            vector = np.array([hash(word) % 1000 for word in text.split()])
            vector = np.pad(vector, (0, max(0, max_length - len(vector))), mode='constant')[:max_length]
            vectors.append(vector)
    return np.array(vectors)
    '''


def get_embeddings(
    model,
    tokenizer_plm,
    tokenizer_llm,
    sequences=None,
    texts=None,
    device="cuda",
    mean_pooling=False,
):
    """Extract embeddings for a sequence and/or text"""

    with torch.no_grad():
        protein_embedding = None
        text_embedding = None

        if sequences is not None:
            processed_seq = [
                " ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences
            ]
            inputs = tokenizer_plm(
                processed_seq,
                return_tensors="pt",
                padding=True,
            ).to(device)

            outputs = model(
                input_ids_sequence=inputs["input_ids"],
                attention_mask_sequence=inputs["attention_mask"],
            )
            protein_embedding = outputs["proj_protein_embeds"]
            if mean_pooling:
                protein_embedding = torch.mean(protein_embedding, dim=1)

        if texts is not None:
            inputs = tokenizer_llm(
                texts,
                return_tensors="pt",
                padding=True,
            ).to(device)

            outputs = model(
                input_ids_text=inputs["input_ids"],
                attention_mask_text=inputs["attention_mask"],
            )
            text_embedding = outputs["proj_text_embeds"]
            if mean_pooling:
                text_embedding = torch.mean(text_embedding, dim=1)

    return protein_embedding, text_embedding


def save_embeddings(output_file, identifiers, embeddings):
    """Save embeddings to an HDF5 file"""
    with h5py.File(output_file, "w") as f:
        for ident, embedding in zip(identifiers, embeddings):
            f.create_dataset(ident, data=embedding.cpu().numpy())


def main():
    """
    parser = argparse.ArgumentParser(description="Generate embeddings for proteins or text sentences.")
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="Input file: a HuggingFace dataset for 'protein' or a CSV file for 'go_terms'.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["protein", "go_terms"],
        help="Mode: 'protein' for protein sequences or 'go_terms' for GO terms.",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="train",
        help="Dataset split to use (e.g., 'train' or 'test'). Applicable only for 'protein' mode.",
    )
    parser.add_argument("-o", "--output_file", required=True, help="Output file for the embeddings (HDF5 format).")

    args = parser.parse_args()
    """

    model_path = Path("tmp/protT5-CLIP-2025-01-07-12-49-43-ddp")

    with open(Path("configs/model.yaml"), "r") as f:
        train_config = yaml.safe_load(f)

    plm_name = train_config["model"]["protein_encoder_name"]
    llm_name = train_config["model"]["text_encoder_name"]

    plm_config = AutoConfig.from_pretrained(plm_name)
    llm_config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)

    model_config = ProtT5CLIPConfig(
        name_or_path_plm=plm_name,
        name_or_path_llm=llm_name,
        plm_config=plm_config,
        llm_config=llm_config,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
        projection_dim=train_config["model"]["text_projection_dim"],
        logit_scale_init_value=train_config["model"]["logit_scale_init_value"],
        device="cpu",
    )

    model = ProtT5CLIP(model_config)
    model.load_adapter(model_path)
    model.model_plm = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    model.to("cuda")
    model.eval()

    tokenizer_plm = T5Tokenizer.from_pretrained(model_config.name_or_path_plm)
    tokenizer_llm = AutoTokenizer.from_pretrained(model_config.name_or_path_llm)

    go_terms, to_embed = read_input_file(Path("output/go_sentences.tsv"), "go_terms")

    embeddings, identifiers = [], []

    for identifier, text in tqdm(
        zip(go_terms, to_embed),
        total=len(go_terms),
        desc="Extracting outputs of GO terms",
    ):
        text += "."
        try:
            _, text_emb = get_embeddings(
                model,
                tokenizer_plm,
                tokenizer_llm,
                sequences=None,
                texts=text,
                device="cuda",
                mean_pooling=True,
            )

            torch.cuda.empty_cache()

            embeddings.append(text_emb)
            identifiers.append(identifier)

        except torch.cuda.OutOfMemoryError:
            print(f"Skipped text due to CUDA Out of Memory: {text}")
            torch.cuda.empty_cache()

    save_embeddings(Path("tmp/output_embeddings.h5"), identifiers, embeddings)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
