import json
import weaviate

# Initialize client
client = weaviate.Client(
    url="https://izyu3iperbgsrwbao7c88q.c0.europe-west3.gcp.weaviate.cloud",
    auth_client_secret=weaviate.AuthClientPassword(
        username="amelie.hilbig@t-online.de",
        password="HobbitEinrad812!"
    )
)

# Query all data
query = """
{
  Get {
    ProteinEmbedding(limit: 10000) {
      key
      sequence
      description
      _additional {
        vector
      }
    }
  }
}
"""
result = client.query.raw(query)

# Save the data for offline use
with open("C:/Users/ameli/OneDrive/Dokumente/protein_data.json", "w") as file:
    json.dump(result, file, indent=4)

print("Data exported to protein_data.json")

