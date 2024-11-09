import re
import pandas as pd

train_terms_path = 'C:/Users/ameli/OneDrive/Dokumente/train_terms.tsv'
go_obo_path = 'C:/Users/ameli/OneDrive/Dokumente/go-basic2.obo'
output_tsv_path = 'C:/Users/ameli/OneDrive/Dokumente/train_terms_extended_new.tsv'

train_terms_df = pd.read_csv(train_terms_path, sep='\t')

def parse_go_obo(file_path):
    with open(file_path, 'r') as file:
        go_terms = {}
        term_data = {}
        inside_term = False

        for line in file:
            line = line.strip()
            if line == "[Term]":
                inside_term = True
                if term_data and 'id' in term_data:
                    go_terms[term_data['id']] = term_data
                    term_data = {}
            elif inside_term and line == "":
                inside_term = False
            elif inside_term:
                if line.startswith("id:"):
                    term_data["id"] = line.split("id: ")[1]
                elif line.startswith("name:"):
                    term_data["name"] = line.split("name: ")[1]
                elif line.startswith("namespace:"):
                    term_data["namespace"] = line.split("namespace: ")[1]
                elif line.startswith("def:"):
                    match = re.match(r'def: "(.*)" \[.*\]', line)
                    if match:
                        term_data["definition"] = match.group(1)

        if term_data and 'id' in term_data:
            go_terms[term_data['id']] = term_data

    return go_terms

go_terms_data = parse_go_obo(go_obo_path)

terms_in_train = set(train_terms_df['term'])

filtered_go_terms_data = {go_id: data for go_id, data in go_terms_data.items() if go_id in terms_in_train}

train_terms_df['GO Name'] = train_terms_df['term'].map(lambda x: filtered_go_terms_data.get(x, {}).get('name', 'N/A'))
train_terms_df['GO Sentence'] = train_terms_df['term'].map(
    lambda x: f"The {filtered_go_terms_data.get(x, {}).get('namespace', 'unknown')} is {filtered_go_terms_data.get(x, {}).get('name', 'N/A')}."
)

train_terms_df = train_terms_df.replace('_', ' ', regex=True)
train_terms_df.to_csv(output_tsv_path, sep='\t', index=False)

print({output_tsv_path})
