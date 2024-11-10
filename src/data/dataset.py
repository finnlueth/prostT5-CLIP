import pandas as pd
import networkx as nx
from goatools import obo_parser
from Bio import SeqIO
from collections import defaultdict


class ProteinData:
    def __init__(self):
        self.sequences = {}  # protein_id -> sequence
        self.go_terms = defaultdict(set)  # protein_id -> set of GO terms
        self.go_graph = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.headers[idx], self.sequences[idx]

    def load_sequences(self, fasta_file):
        """Load protein sequences from FASTA"""
        for record in SeqIO.parse(fasta_file, "fasta"):
            self.sequences[record.id] = str(record.seq)

    def load_go_terms(self, go_annotation_file):
        """Load GO term annotations"""
        df = pd.read_csv(go_annotation_file, sep="\t")
        for _, row in df.iterrows():
            self.go_terms[row["protein_id"]].add(row["go_term"])

    def load_go_hierarchy(self, go_obo_file):
        """Load GO term hierarchy"""
        self.go_graph = obo_parser.GODag(go_obo_file)

    def propagate_go_terms(self):
        """Propagate GO terms up the hierarchy"""
        for protein_id in self.go_terms:
            propagated_terms = set()
            for go_term in self.go_terms[protein_id]:
                if go_term in self.go_graph:
                    propagated_terms.update(self.go_graph[go_term].get_all_parents())
            self.go_terms[protein_id].update(propagated_terms)

    def create_dataset(self, output_file):
        """Create final dataset with sequences and GO terms"""
        data = []
        for protein_id in self.sequences:
            if protein_id in self.go_terms:
                data.append(
                    {
                        "protein_id": protein_id,
                        "sequence": self.sequences[protein_id],
                        "go_terms": "|".join(self.go_terms[protein_id]),
                    }
                )

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)


# Usage example
def process_cafa_with_go():
    processor = ProteinData()

    # Load data
    processor.load_sequences("cafa_sequences.fasta")
    processor.load_go_terms("cafa_go_annotations.tsv")
    processor.load_go_hierarchy("go.obo")

    # Process GO terms
    processor.propagate_go_terms()

    # Create final dataset
    processor.create_dataset("processed_cafa_with_go.csv")
