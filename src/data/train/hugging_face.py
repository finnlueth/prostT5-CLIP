import h5py

new_fasta_file_path = 'C:/Users/ameli/OneDrive/Dokumente/Amelie/Master/cath-dataset-nonredundant-S20-v4_3_0.fa'

new_sequences = []
with open(new_fasta_file_path, "r") as file:
    current_id = None
    current_seq = []
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if current_id is not None:
                new_sequences.append({"id": current_id, "sequence": "".join(current_seq)})
            current_id = line[1:]
            current_seq = []
        else:
            current_seq.append(line)

    if current_id is not None:
        new_sequences.append({"id": current_id, "sequence": "".join(current_seq)})

new_h5_file_path = "C:/Users/ameli/OneDrive/Dokumente/Amelie/Master/cath-dataset-nonredundant-S20-v4_3_0.h5"

with h5py.File(new_h5_file_path, "w") as h5_file:
    dataset_group = h5_file.create_group("dataset")
    ids = [entry["id"] for entry in new_sequences]
    sequences_data = [entry["sequence"] for entry in new_sequences]

    dataset_group.create_dataset("ids", data=[s.encode("utf-8") for s in ids])
    dataset_group.create_dataset("sequences", data=[s.encode("utf-8") for s in sequences_data])

print(new_h5_file_path)

with h5py.File(new_h5_file_path, "r") as h5_file:
    keys = list(h5_file.keys())
    ids = [id.decode("utf-8") for id in h5_file["dataset"]["ids"][:]]
    sequences = [seq.decode("utf-8") for seq in h5_file["dataset"]["sequences"][:]]

sample_data = {
    "ids": ids[:5],
    "sequences": sequences[:5]
}
print(sample_data)
