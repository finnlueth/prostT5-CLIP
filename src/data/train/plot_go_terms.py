import matplotlib.pyplot as plt
import pandas as pd
import re

train_terms_path = 'C:/Users/ameli/OneDrive/Dokumente/train_terms.tsv'
go_obo_path = 'C:/Users/ameli/OneDrive/Dokumente/go-basic2.obo'
results_path = 'C:/Users/ameli/OneDrive/Dokumente/results1.txt'

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
train_terms_df = pd.read_csv(train_terms_path, sep='\t')
tsv_aspects = parse_tsv_by_aspect(train_terms_path)

# Calculate results
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

aspects = list(results.keys())
missing_counts = [results[a]["missing"] for a in aspects]
coverage_values = [results[a]["coverage_percentage"] for a in aspects]

fig, axs = plt.subplots(1, 3, figsize=(16, 8))

colors = ['blue', 'red', 'green'][:len(aspects)]
axs[0].bar(aspects, missing_counts, color=colors)
axs[0].set_title('Missing GO-Terms by Aspect')
axs[0].set_xlabel('Aspect')
axs[0].set_ylabel('Number of Missing GO-Terms')

go_terms_per_aspect = [len(obo_aspects[a]) for a in aspects]
axs[1].bar(aspects, go_terms_per_aspect, color=colors)
axs[1].set_title('Total GO-Terms by Aspect')
axs[1].set_xlabel('Aspect')
axs[1].set_ylabel('Number of GO-Terms')


axs[2].bar(aspects, coverage_values, color=colors)
axs[2].set_title('Coverage by Aspect')
axs[2].set_xlabel('Aspect')
axs[2].set_ylabel('Coverage Percentage')

plt.tight_layout()
plt.show()
