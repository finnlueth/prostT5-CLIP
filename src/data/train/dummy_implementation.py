import csv
import math
from collections import defaultdict

def load_go_terms(file_path):
    go_map = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            go_map[row['text'].lower()] = row['uid']
    return go_map

def sentences_to_go_terms(sentences, go_map):
    go_terms = []
    for sentence in sentences:
        sentence = sentence.lower()
        if sentence in go_map:
            go_terms.append(go_map[sentence])
    return go_terms

def add_ic(go_terms):
    return {term: 1.0 for term in go_terms}

def calculate_metrics(predicted_terms, true_terms, ic_values):
    tp = set(predicted_terms) & set(true_terms)
    fp = set(predicted_terms) - set(true_terms)
    fn = set(true_terms) - set(predicted_terms)

    weighted_tp = sum(ic_values[term] for term in tp)
    weighted_fp = sum(ic_values[term] for term in fp)
    weighted_fn = sum(ic_values[term] for term in fn)

    precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def cafa_evaluation(input_sentences, true_go_terms, go_terms_file):
    go_map = load_go_terms(go_terms_file)
    predicted_go_terms = sentences_to_go_terms(input_sentences, go_map)
    ic_values = add_ic(set(predicted_go_terms + true_go_terms))

    precision, recall, f1 = calculate_metrics(predicted_go_terms, true_go_terms, ic_values)

    print(f"Predicted GO terms: {predicted_go_terms}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


input_sentences = [
    "The biological process is cell morphogenesis.",
    "The molecular function is RNA binding.",
    "The cellular component is mitochondrion."
]

true_go_terms = ["GO:0000902", "GO:0003723", "GO:0005739"]

cafa_evaluation(input_sentences, true_go_terms, 'C:/Users/ameli/OneDrive/Dokumente/go_terms.csv')
