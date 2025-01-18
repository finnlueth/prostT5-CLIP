import h5py
import weaviate
from weaviate import AuthClientPassword

# File path to the uploaded HDF5 file
file_path = "C:/Users/ameli/OneDrive/Dokumente/cath_embeddings_v2.h5"

# Initialize Weaviate client
client = weaviate.Client(
    url="https://izyu3iperbgsrwbao7c88q.c0.europe-west3.gcp.weaviate.cloud",
    auth_client_secret=AuthClientPassword(
        username="amelie.hilbig@t-online.de",
        password="HobbitEinrad812!"
    ),
    startup_period=15  # Increased timeout period
)

# Define schema
schema = {
    "classes": [
        {
            "class": "ProteinEmbedding",
            "description": "A class to store protein identifiers, embeddings, sequences, and descriptions",
            "properties": [
                {
                    "name": "key",
                    "dataType": ["text"],
                    "description": "Unique identifier for the protein",
                },
                {
                    "name": "sequence",
                    "dataType": ["text"],
                    "description": "Protein sequence",
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Description of the protein",
                }
            ],
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": "cosine"
            }
        }
    ]
}

# Ensure schema exists
existing_classes = [cls['class'] for cls in client.schema.get()['classes']]
if "ProteinEmbedding" not in existing_classes:
    client.schema.create(schema)
    print("Schema created successfully.")
else:
    print("Schema 'ProteinEmbedding' already exists. Skipping creation.")

def process_dataset(key, dataset, client):
    """Process an HDF5 dataset and upload it to Weaviate."""
    if dataset.shape == (1, 1024) and dataset.dtype == "float32":
        embedding = dataset[:].flatten().tolist()  # Convert to list for Weaviate

        # Example placeholders for sequence and description
        sequence = f"Example sequence for {key}"
        description = f"Example description for {key}"

        data_object = {
            "key": key,
            "sequence": sequence,
            "description": description
        }

        client.data_object.create(
            data_object,
            class_name="ProteinEmbedding",
            vector=embedding
        )
        print(f"Uploaded dataset '{key}'.")
    else:
        print(f"Skipping dataset '{key}' due to unexpected shape or type: {dataset.shape}, {dataset.dtype}.")

def process_group(group, client):
    """Recursively process HDF5 groups and upload datasets to Weaviate."""
    for key in group.keys():
        item = group[key]

        if isinstance(item, h5py.Dataset):
            process_dataset(key, item, client)
        elif isinstance(item, h5py.Group):
            print(f"Exploring group '{key}'...")
            process_group(item, client)

# Open and process the HDF5 file
with h5py.File(file_path, "r") as h5file:
    for top_key in h5file.keys():
        item = h5file[top_key]
        if isinstance(item, h5py.Group):
            print(f"Processing top-level group '{top_key}'...")
            process_group(item, client)
        elif isinstance(item, h5py.Dataset):
            process_dataset(top_key, item, client)
        else:
            print(f"Skipping top-level key '{top_key}' as it is neither a group nor a dataset.")

print("Data upload completed!")


