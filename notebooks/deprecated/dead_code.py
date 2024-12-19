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





 # features_plm = {
        #     'input_ids': [feature['input_ids_sequence'] for feature in features],
        #     'attention_mask': [feature['attention_mask_sequence'] for feature in features]
        # }
        # features_llm = {
        #     'input_ids': [feature['input_ids_text'] for feature in features],
        #     'attention_mask': [feature['attention_mask_text'] for feature in features]
        # }
        
        
        
        
        
# overwrite = False
# processed_dataset_path = "../tmp/data/processed_train_val_GO"
# # processed_dataset_path = "tmp/data/processed_train_val_GO_FULL"

# if not overwrite and os.path.exists(processed_dataset_path):
#     print("Loading processed dataset from disk...")
#     dataset = load_from_disk(processed_dataset_path)
# else:
#     print("Processing dataset...")
#     dataset = load_from_disk("../tmp/data/train_val_GO")
#     dataset = DatasetDict({
#         'train': dataset['train'].select(range(2000)),
#         'test': dataset['test'].select(range(600))
#     })

#     for split in dataset:
#         dataset[split] = dataset[split].filter(lambda x: len(x["sequence"]) < 256)

#         dataset[split] = dataset[split].map(lambda x: {"sequence": " ".join(list(re.sub(r"[UZOB]", "X", x["sequence"])))})
#         dataset[split] = dataset[split].remove_columns(["identifier", "term", "aspect", "GO Name", "species", "__index_level_0__"])

#         tknz_plm = tokenizer_plm(text=dataset[split]["sequence"], padding=False, truncation=False)
#         tknz_llm = tokenizer_llm(text=dataset[split]["GO Sentence"], padding=False, truncation=False)


#         dataset[split] = dataset[split].add_column("input_ids_sequence", tknz_plm["input_ids"])
#         dataset[split] = dataset[split].add_column("attention_mask_sequence", tknz_plm["attention_mask"])
#         dataset[split] = dataset[split].add_column("input_ids_text", tknz_llm["input_ids"])
#         dataset[split] = dataset[split].add_column("attention_mask_text", tknz_llm["attention_mask"])

#     dataset = dataset.remove_columns(["sequence", "GO Sentence"])

#     print("Saving processed dataset to disk...")
#     dataset.save_to_disk(processed_dataset_path)