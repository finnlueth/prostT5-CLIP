import pandas as pd
from datasets import load_from_disk
from transformers import TrainingArguments
from subset_trainer import SubsetTrainer
import torch
from torch import nn

# Load real dataset
dataset_path = "tmp/data/processed_train_val_GO"
dataset = load_from_disk(dataset_path)

# Use a subset of 1000 rows from the train split
train_dataset_full = dataset["train"]
train_dataset = train_dataset_full.select(range(3000)) 

# Define a minimal PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):
        return {"loss": torch.tensor(0.0)}  # Return a dummy loss for testing

# Training arguments
training_args = TrainingArguments(
    output_dir="./test_output",
    per_device_train_batch_size=10,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=1,
)

# Instantiate the model
model = SimpleModel()

# Define a dummy data collator
def data_collator(features):
    return features

# Instantiate SubsetTrainer
trainer = SubsetTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # Not testing evaluation here
    tokenizer=None,
    data_collator=data_collator,
)

# DataFrame to collect results
output_data = []

# Simulate training to test the dataloader
for epoch in range(int(training_args.num_train_epochs)):
    dataloader = trainer.get_train_dataloader()
    for batch in dataloader:
        for example in batch:
            output_data.append({
                "epoch": epoch,
                "identifier": example["identifier"],
                "GO term": example["term"]
            })
        break  # Only collect one batch per epoch for brevity

# Save to pandas DataFrame
df = pd.DataFrame(output_data)

# Write to CSV file
output_file = "epoch_output.csv"
df.to_csv(output_file, index=False)
print(f"Output saved to {output_file}")
