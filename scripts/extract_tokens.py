import csv

# Define the input and output file paths
input_file = "go-basic.obo"
output_file = "go_terms.csv"

# Initialize variables
terms = []
current_term = {}

# Read and parse the OBO file
with open(input_file, "r") as file:
    for line in file:
        line = line.strip()
        
        if line == "[Term]":
            # Save the current term if it's valid
            if current_term and not current_term.get("name", "").startswith("obsolete"):
                terms.append(current_term)
            # Start a new term
            current_term = {}
        elif line.startswith("id: "):
            current_term["id"] = line.split("id: ")[1]
        elif line.startswith("name: "):
            current_term["name"] = line.split("name: ")[1]
        elif line.startswith("namespace: "):
            current_term["namespace"] = line.split("namespace: ")[1].replace("_", " ")
        elif line == "":
            continue

# Save the last term if it's valid
if current_term and not current_term.get("name", "").startswith("obsolete"):
    terms.append(current_term)

# Write the terms to a CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["GO:term", "name", "namespace", "description"])
    
    for term in terms:
        go_term = term["id"]
        name = term["name"]
        namespace = term["namespace"]
        description = f"The {namespace} is {name.replace('_', ' ')}."
        writer.writerow([go_term, name, namespace, description])

print(f"CSV file '{output_file}' has been created successfully!")
