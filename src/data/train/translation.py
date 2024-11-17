import re
import csv
import pandas as pd
from collections import defaultdict

train_terms_path = 'C:/Users/ameli/OneDrive/Dokumente/train_terms.tsv'
go_obo_path = 'C:/Users/ameli/OneDrive/Dokumente/go-basic2.obo'
output_tsv_path = 'C:/Users/ameli/OneDrive/Dokumente/train_terms_extended_new1.tsv'
def parse_go_obo(file_path):
    go_terms = {}
    with open(file_path, 'r') as file:
        term_data = {}
        inside_term = False

        for line in file:
            line = line.strip()
            if line == "[Term]":
                if term_data.get("id"):
                    go_terms[term_data["id"]] = term_data
                term_data = {}
                inside_term = True
            elif inside_term and line == "":
                inside_term = False
            elif inside_term:
                key, *value = line.split(": ", 1)
                if key in {"id", "name", "namespace"}:
                    term_data[key] = value[0]
                elif key == "def":
                    match = re.match(r'"(.*)" \[.*\]', value[0])
                    if match:
                        term_data["definition"] = match.group(1)

        if term_data.get("id"):
            go_terms[term_data["id"]] = term_data
    return go_terms

go_terms_data = parse_go_obo(go_obo_path)

train_terms_df = pd.read_csv(train_terms_path, sep='\t')
terms_in_train = set(train_terms_df['term'])
filtered_go_terms_data = {go_id: data for go_id, data in go_terms_data.items() if go_id in terms_in_train}

train_terms_df['GO Name'] = train_terms_df['term'].map(lambda x: filtered_go_terms_data.get(x, {}).get('name', 'N/A'))
train_terms_df['GO Sentence'] = train_terms_df['term'].map(
    lambda x: f"The {filtered_go_terms_data.get(x, {}).get('namespace', 'unknown')} is {filtered_go_terms_data.get(x, {}).get('name', 'N/A')}."
)

train_terms_df.drop_duplicates(subset='term', inplace=True)
train_terms_df = train_terms_df.replace('_', ' ', regex=True)
train_terms_df.to_csv(output_tsv_path, sep='\t', index=False)
print(f"Extended TSV file saved to: {output_tsv_path}")
def parse_obo_by_aspect(file_path):
    aspects = {"BPO": set(), "CCO": set(), "MFO": set()}
    with open(file_path, 'r') as file:
        current_id, current_namespace = None, None
        for line in file:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split(": ")[1]
            elif line.startswith("namespace:"):
                ns_map = {
                    "biological_process": "BPO",
                    "cellular_component": "CCO",
                    "molecular_function": "MFO",
                }
                current_namespace = ns_map.get(line.split(": ")[1])
            elif line == "" and current_id and current_namespace:
                aspects[current_namespace].add(current_id)
                current_id, current_namespace = None, None
    return aspects

def parse_tsv_by_aspect(file_path):
    aspects = {"BPO": set(), "CCO": set(), "MFO": set()}
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            _, go_term, aspect = line.strip().split('\t')[:3]
            aspects[aspect].add(go_term)
    return aspects

obo_aspects = parse_obo_by_aspect(go_obo_path)
tsv_aspects = parse_tsv_by_aspect(output_tsv_path)
results = {}
for aspect, obo_terms in obo_aspects.items():
    tsv_terms = tsv_aspects.get(aspect, set())
    missing_terms = obo_terms - tsv_terms
    coverage = (len(tsv_terms) / len(obo_terms)) * 100 if obo_terms else 0
    results[aspect] = {
        "total_in_obo": len(obo_terms),
        "total_in_tsv": len(tsv_terms),
        "missing": len(missing_terms),
        "coverage_percentage": coverage,
        "missing_terms": list(missing_terms),
    }

output_path = 'C:/Users/ameli/OneDrive/Dokumente/results.txt'
with open(output_path, 'w', encoding='utf-8') as output_file:
    for aspect, data in results.items():
        print(f"Aspect: {aspect}")
        print(f"  Total in OBO: {data['total_in_obo']}")
        print(f"  Total in TSV: {data['total_in_tsv']}")
        print(f"  Coverage: {data['coverage_percentage']:.2f}%")
        print(f"  Missing terms: {data['missing']}")
        print(f"  Example missing terms: {data['missing_terms'][:10]}")
        print()
        output_file.write(f"Aspect: {aspect}\n")
        output_file.write(f"Missing terms ({len(data['missing_terms'])}):\n")
        output_file.writelines(term + '\n' for term in data['missing_terms'])
        output_file.write('\n')

print(f"Results saved to: {output_path}")


