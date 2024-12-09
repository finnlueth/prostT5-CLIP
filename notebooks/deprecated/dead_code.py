# print("\nModel device:", next(model.parameters()).device)
# print("\nModel PLM device:", next(model.model_plm.parameters()).device)
# print("\nModel LLM device:", next(model.model_llm.parameters()).device)

# print("\nProtein Model (T5) Parameter dtypes:")
# for name, param in model.model_plm.named_parameters():
#     print(f"{name}: {param.dtype}")

# print("\nText Model (Phi) Parameter dtypes:")
# for name, param in model.model_llm.named_parameters():
#     print(f"{name}: {param.dtype}")

# print("\nProjection Layer Parameter dtypes:")
# for name, param in model.protein_projection.named_parameters():
#     print(f"protein_projection.{name}: {param.dtype}")
# for name, param in model.text_projection.named_parameters():
#     print(f"text_projection.{name}: {param.dtype}")

# print(f"\nLogit Scale dtype: {model.logit_scale.dtype}")

















# dataset = [
#     {
#         "uid": "A001",
#         "sequence": "MLEVPVWIPILAFAVGLGLGLLIPHLQKPFQRFPHLQKPFQRF",
#         "text": "This protein is involved in membrane transport.",
#     },
#     {
#         "uid": "A002",
#         "sequence": "MSLEQKKGADIISKILQIQNSIGKTTSPSTLKTTSPSTLKT",
#         "text": "This enzyme catalyzes the hydrolysis of ATP.",
#     },
#     {
#         "uid": "A003",
#         "sequence": "MKMKQQGLVADLLPNIRVMKTFGHFVFNYYNDN",
#         "text": "This transcription factor regulates gene expression.",
#     },
# ] * 1000

# dataset = Dataset.from_list(dataset)
# dataset = dataset.add_column("sequence_original", dataset["sequence"])
# dataset = dataset.map(lambda x: {"sequence": " ".join(list(re.sub(r"[UZOB]", "X", x["sequence"])))})

# tknz_plm = tokenizer_plm(text=dataset["sequence"], padding=False, truncation=False)
# tknz_llm = tokenizer_llm(text=dataset["text"], padding=False, truncation=False)

# dataset = dataset.add_column(
#     "input_ids", [{"sequence": seq, "text": txt} for seq, txt in zip(tknz_plm["input_ids"], tknz_llm["input_ids"])]
# )
# dataset = dataset.add_column(
#     "attention_mask", [{"sequence": seq, "text": txt} for seq, txt in zip(tknz_plm["attention_mask"], tknz_llm["attention_mask"])]
# )

# dataset = dataset.remove_columns(["uid", "sequence", "text", "sequence_original"])
# dataset = DatasetDict({"train": dataset, "test": dataset})

# print(dataset)
# print(dataset["train"][0])









# dataset[split] = dataset[split].add_column(
#     "input_ids", [{"sequence": seq, "text": txt} for seq, txt in zip(tknz_plm["input_ids"], tknz_llm["input_ids"])]
# )
# dataset[split] = dataset[split].add_column(
#     "attention_mask", [{"sequence": seq, "text": txt} for seq, txt in zip(tknz_plm["attention_mask"], tknz_llm["attention_mask"])]
# )