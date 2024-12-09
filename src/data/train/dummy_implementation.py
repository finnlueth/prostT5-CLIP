import pandas as pd
import numpy as np
def cafa_f1_per_protein(predictions, true_labels, sentence_to_go_mapping, ic_values):

    metrics_per_protein = {}

    for protein_id, predicted_sentences in predictions.items():

        predicted_go_terms = [
            sentence_to_go_mapping.get(sentence.strip()) for sentence in predicted_sentences
        ]
        predicted_go_terms = [term for term in predicted_go_terms if term is not None]


        true_go_terms = true_labels.get(protein_id, [])

        tp = set(predicted_go_terms) & set(true_go_terms)
        fp = set(predicted_go_terms) - set(true_go_terms)
        fn = set(true_go_terms) - set(predicted_go_terms)

        weighted_tp = sum(ic_values.get(term, 0) for term in tp)
        weighted_fp = sum(ic_values.get(term, 0) for term in fp)
        weighted_fn = sum(ic_values.get(term, 0) for term in fn)

        precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0
        recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_per_protein[protein_id] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    return metrics_per_protein

def prepare_5_protein_data():

    predictions = {
        "Protein1": ["The biological process is metabolic process."],
        "Protein2": ["The biological process is RNA processing."],
        "Protein3": ["The biological process is cell division."],
        "Protein4": ["The biological process is signal transduction."],
        "Protein5": ["The biological process is protein folding."]
    }

    true_labels = {
        "Protein1": ["GO:0008152"],
        "Protein2": ["GO:0006396"],
        "Protein3": ["GO:0051301"],
        "Protein4": ["GO:0007165"],
        "Protein5": ["GO:0006457"]
    }

    sentence_to_go_mapping = {
        "The biological process is metabolic process.": "GO:0008152",
        "The biological process is RNA processing.": "GO:0006396",
        "The biological process is cell division.": "GO:0051301",
        "The biological process is signal transduction.": "GO:0007165",
        "The biological process is protein folding.": "GO:0006457"
    }

    ic_values = {
        "GO:0008152": 12.3,
        "GO:0006396": 10.5,
        "GO:0051301": 11.2,
        "GO:0007165": 9.8,
        "GO:0006457": 13.1
    }

    return predictions, true_labels, sentence_to_go_mapping, ic_values

if __name__ == "__main__":

    predictions, true_labels, sentence_to_go_mapping, ic_values = prepare_5_protein_data()
    metrics = cafa_f1_per_protein(predictions, true_labels, sentence_to_go_mapping, ic_values)

    for protein_id, values in metrics.items():
        print(f"{protein_id}: Precision = {values['Precision']:.4f}, Recall = {values['Recall']:.4f}, F1 = {values['F1']:.4f}")
