import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from pyfaidx import Fasta
from tqdm import tqdm
from typing import Tuple, Union
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer

from ..utils.config import get_params


def setup_model(checkpoint, device: torch.device = "mps") -> Tuple:

    if "esm" in checkpoint:
        mod_type = "esm"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = EsmModel.from_pretrained(checkpoint)
        model = model.to(device)
    elif "ankh" in checkpoint:
        mod_type = "ankh"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = T5EncoderModel.from_pretrained(checkpoint)
        model = model.to(device)
    else:
        mod_type = "pt"
        tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)
        model = T5EncoderModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        model = model.to(device)
        model = model.half()

    return model, tokenizer, mod_type


def process_fasta(fasta_path: Path, max_len=1022) -> Path:
    """Remove sequences longer than max_len and save their identifiers to a file."""
    filtered_fasta_path = fasta_path.with_suffix(".filtered.fasta")
    long_sequences_path = fasta_path.with_suffix(".long_sequences.txt")

    with filtered_fasta_path.open("w") as filtered_fasta, long_sequences_path.open(
        "w"
    ) as long_sequences:
        for header, seq in Fasta(str(fasta_path)).items():
            if len(seq) > max_len:
                long_sequences.write(f"{header.split()[0]}\n")
            else:
                filtered_fasta.write(f">{header}\n{seq}\n")

    return filtered_fasta_path


def seq_preprocess(df, model_type="esm") -> Union[pd.DataFrame, None]:
    df["sequence"] = df["sequence"].str.replace("[UZO]", "X", regex=True)

    if model_type in "esm":
        return df
    elif model_type == "ankh":
        return df
    elif model_type == "pt":
        df["sequence"] = df.apply(lambda row: " ".join(row["sequence"]), axis=1)
        return df
    else:
        return None


def read_fasta(file_path: Path) -> Tuple[list, list]:
    headers = []
    sequences = []
    fasta = Fasta(str(file_path))
    for seq in fasta:
        headers.append(seq.name)
        sequences.append(str(seq))
    return headers, sequences


def create_embedding(
    checkpoint,
    df,
    emb_type="per_prot",
    max_seq_len=1024,
    device: torch.device = "mps",
    outputs: Path = Path("protein_embeddings.h5"),
):
    model, tokenizer, mod_type = setup_model(checkpoint, device=device)
    model.eval()
    df = seq_preprocess(df, mod_type)

    def compute_embedding(sequence, emb_type):
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding=True,
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.cpu().numpy()
        if emb_type == "per_res":
            # remove special tokens
            if mod_type in ["pt", "ankh"]:
                outputs = outputs[:-1, :]
            elif mod_type == "esm":
                outputs = np.squeeze(outputs, axis=0)[:-1, :]
            return outputs
        elif emb_type == "per_prot":
            return outputs.mean(axis=1).flatten()
        else:
            raise ValueError("Input valid embedding type")

    with h5py.File(outputs, "a") as hdf:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sequence = row["sequence"]
            header = row["header"]

            # Check if the embedding already exists
            if header in hdf:
                continue

            embedding = compute_embedding(sequence, emb_type)
            hdf.create_dataset(name=header, data=embedding)

    print(f"Embeddings saved to {outputs}")

    # clean up gpu
    del model
    del tokenizer
    del df
    torch.cuda.empty_cache()


def main():
    params = get_params()
    embed_params = params["embed"]

    for split in ["train_set", "test_set"]:
        inputs = Path(embed_params[split]["sequences"]).resolve()

        if embed_params.get("max_seq_len"):
            filtered_path = process_fasta(inputs, embed_params["max_seq_len"])
            headers, sequences = read_fasta(filtered_path)
        else:
            headers, sequences = read_fasta(inputs)

        df = pd.DataFrame({"header": headers, "sequence": sequences})

        # Generate embeddings
        create_embedding(
            checkpoint=embed_params["model"],
            df=df,
            emb_type=embed_params["emb_type"],
            device=torch.device(embed_params["device"]),
            outputs=embed_params[split]["outputs"],
        )


if __name__ == "__main__":
    main()
