import logging
from pathlib import Path
from typing import Tuple

import h5py
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_from_disk
from src.utils.config import get_params


def setup_model(checkpoint="microsoft/Phi-3.5-mini-instruct", device: torch.device = "cuda") -> Tuple:
    torch.backends.cuda.enable_flash_sdp(False)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # for flash attention set attn_implementation="flash_attention_2
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def create_embedding(
    df: pd.DataFrame,
    model,
    tokenizer,
    output: Path,
    sentence_level: str = "sentence_embeddings",
    max_length: int = 512,
    device: torch.device = "cuda",
):
    model, tokenizer = setup_model(checkpoint=model, device=device)

    def compute_embedding(text: str, level: str, max_length: int = 128):
        torch.cuda.empty_cache()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        if level == "token_embeddings":
            return hidden_states.squeeze(0).cpu().float().numpy()
        elif level == "sentence_embeddings":
            return hidden_states.mean(dim=1).squeeze(0).cpu().float().numpy()
        else:
            raise ValueError("Sentence level must be either 'token_embeddings' or 'sentence_embeddings'")

    with h5py.File(output, "a") as hdf:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding texts"):
            """
            go = row["term"]

            if go in hdf:
                tqdm.write(f"GO term {go} already in text embedding, skipping")
                continue
                
            """

            protein = row["proteins"]

            if protein in hdf:
                tqdm.write(f"Protein {protein} already has text embedding, skipping")
                continue

            text = f"{row["sentences"]}"
            embedding = compute_embedding(text, sentence_level, max_length)
            hdf.create_dataset(name=protein, data=embedding)

    logging.info(f"Embeddings saved to {output}")


def main():
    params = get_params("embed_text")

    # ds = pd.read_csv(Path(params["sentence"]), sep="\t")

    ds = load_from_disk(Path(params["source"]))

    create_embedding(
        df=ds["train"].to_pandas(),
        model=params["model"],
        tokenizer=params["tokenizer"],
        output=Path(params["output"]),
        sentence_level=params["sentence_level"],
        max_length=params["max_len"],
        device=torch.device(params["device"]),
    )


if __name__ == "__main__":
    main()
