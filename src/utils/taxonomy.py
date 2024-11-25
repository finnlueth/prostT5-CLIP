from Bio import Entrez
from pathlib import Path
import pickle
from typing import Union, List


class TaxonomyMapper:
    def __init__(
        self,
        email: str = "xihenghe@tum.de",
        cache_file: Path = Path("tmp/taxonomy_cache.pkl"),
    ):
        Entrez.email = email
        self.cache_file = cache_file
        self._load_cache()

    def _load_cache(self):
        """Load or create taxonomy cache"""
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                self.tax_cache = pickle.load(f)
        else:
            self.tax_cache = {}

    def _save_cache(self):
        """Save taxonomy cache"""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.tax_cache, f)

    def map_taxonomy_to_species(self, tax_id: Union[str, List[str]]) -> dict:
        """Map NCBI taxonomy IDs to species names"""
        if isinstance(tax_id, str):
            tax_id = [tax_id]

        to_fetch = [tid for tid in tax_id if tid not in self.tax_cache]

        if to_fetch:
            try:
                handle = Entrez.efetch(db="taxonomy", id=to_fetch, retmode="xml")
                records = Entrez.read(handle)
                handle.close()

                for tid, record in zip(to_fetch, records):
                    self.tax_cache[tid] = {
                        "scientific_name": record["ScientificName"],
                        "lineage": record.get("LineageEx", []),
                        "rank": record.get("Rank", ""),
                    }
                self._save_cache()

            except Exception as e:
                print(f"Error fetching taxonomy: {e}")
                return {}

        return {tid: self.tax_cache[tid]["scientific_name"] for tid in tax_id}

    def get_broad_taxonomy(self, tax_id: str) -> str:
        """Get broad taxonomic category for a taxonomy ID"""
        if tax_id not in self.tax_cache:
            self.map_taxonomy_to_species(tax_id)

        if tax_id not in self.tax_cache:
            return "Unknown"

        lineage = self.tax_cache[tax_id].get("lineage", [])
        ranks = {entry.get("Rank"): entry.get("ScientificName") for entry in lineage}

        if "superkingdom" in ranks:
            if ranks["superkingdom"] == "Bacteria":
                return "Bacteria"
            elif ranks["superkingdom"] == "Eukaryota":
                if "kingdom" in ranks:
                    if ranks["kingdom"] == "Metazoa":
                        return "Animals"
                    elif ranks["kingdom"] == "Fungi":
                        return "Fungi"
                    elif ranks["kingdom"] == "Viridiplantae":
                        return "Plants"
                return "Other Eukaryotes"
            elif ranks["superkingdom"] == "Archaea":
                return "Archaea"
            elif ranks["superkingdom"] == "Viruses":
                return "Viruses"
        return "Unknown"
