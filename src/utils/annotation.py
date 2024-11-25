import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List

import requests
from joblib import Parallel, delayed
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

RELATIONSHIPS = {
    "part_of": "part of",
    "regulates": "regulates",
    "negatively_regulates": "negatively regulates",
    "positively_regulates": "positively regulates",
}


class AnnotationParser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _parse(self):
        raise NotImplementedError

    @abstractmethod
    def get_annotations(self, **kwargs) -> List[str]:
        raise NotImplementedError


class GOParser(AnnotationParser):
    def __init__(self, go_file: Path):
        """
        Initialize GO annotation Parser.

        Args:
            go_file: Path to the go-basic.obo file
        """
        self.go_file = go_file
        self.go_hierarchy, self.go_names, self.go_sentences = self._parse()

    def _process_hierarchy_chunk(self, chunks: list) -> tuple:
        """Process a chunk of complete GO terms."""
        hierarchy = defaultdict(set)
        names, sentences = {}, {}
        relation_to_exclude = set()

        for chunk in chunks:
            lines = chunk.strip().split("\n")
            current_id = None
            current_name = None
            is_obsolete = False

            for line in lines:
                line = line.strip()
                if line.startswith("id: "):
                    current_id = line[4:]
                elif line.startswith("name: "):
                    is_obsolete = "obsolete" in line
                    if not is_obsolete:
                        current_name = line[6:]
                        names[current_id] = current_name
                elif not is_obsolete and line.startswith("namespace: "):
                    namespace = line[11:]
                    if namespace == "external":
                        names.pop(current_id)
                        break
                    sentences[current_id] = {
                        "namespace": namespace,
                        "sentence": f"The {namespace} is {current_name}",
                    }
                elif not is_obsolete and line.startswith("is_a: "):
                    parent = line.split()[1]
                    hierarchy[parent].add(current_id)
                elif not is_obsolete and line.startswith("relationship: "):
                    relation_to_exclude.add(line.split()[2])

        return hierarchy, names, sentences, relation_to_exclude

    def _parse(self) -> tuple:
        """Parse the GO term hierarchy from the go-basic.obo."""
        with open(self.go_file, "r") as f:
            content = f.read()

        terms = content.split("[Term]")
        terms = [t for t in terms if t.strip()]

        chunk_size = len(terms) // (8 * 2)
        chunks = [terms[i : i + chunk_size] for i in range(0, len(terms), chunk_size)]

        logging.info(f"Parsing {len(terms)} GO terms")
        results = Parallel(n_jobs=-1)(delayed(self._process_hierarchy_chunk)(chunk) for chunk in chunks)

        hierarchy = defaultdict(set)
        names = {}
        sentences = {}
        excludes = set()

        for r in results:
            for parent, children in r[0].items():
                hierarchy[parent].update(children)
            names.update(r[1])
            sentences.update(r[2])
            excludes.update(r[3])

        for term in excludes:
            names.pop(term)
            if term in hierarchy:
                hierarchy.pop(term)

        return hierarchy, names, sentences

    def _calculate_depths(self) -> dict:
        """Calculate the depth of each GO term in the hierarchy."""
        depths = {}
        visited = set()

        def get_depth(term):
            if term in visited:
                return depths[term]

            visited.add(term)
            children = self.go_hierarchy.get(term, set())
            if not children:
                depths[term] = 0
                return 0

            max_child_depth = max(get_depth(child) for child in children)
            depths[term] = max_child_depth + 1
            return depths[term]

        for term in self.go_hierarchy:
            if term not in visited:
                get_depth(term)

        return depths

    def _reduce_redundancy(self, terms: set) -> set:
        """
        Remove redundant GO terms by keeping only the most specific (deepest) terms.

        Args:
            terms: Set of GO terms
        Returns:
            set: Set of non redundant Go terms
        """
        non_redundants = set()

        for term in terms:
            if term in self.go_names:
                children = self.go_hierarchy.get(term, set())
                if not children.intersection(terms):
                    non_redundants.add(term)

        return non_redundants

    def get_annotations(self, terms: List) -> List[str]:
        """Get GO term annotations from a series of redundant GO terms for proteins.
        Args:
            terms: A List of redundant GO terms lists for proteins
        Returns:
            List: List of non-redundant GO terms (str that joined by ',') for proteins
        """
        return [",".join(sorted(self._reduce_redundancy(set(t)))) for t in tqdm(terms, desc="Processing GO terms")]

    def get_term_names(self, term: str) -> str:
        """Get GO term name from GO ID"""
        return self.go_names[term]


class UniprotParser(AnnotationParser):
    def __init__(self, cache_file: str = "tmp/uniprot_annotation.json"):
        super().__init__()
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(5, backoff_factor=0.2)))

    def _load_cache(self) -> dict:
        """Load GO terms cache from JSON file"""
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save GO terms cache to JSON file"""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def _parse(self, prot_id: str) -> List[str]:
        """Parse GO annotations from Uniprot"""
        try:
            response = self.session.get(
                f"https://rest.uniprot.org/uniprotkb/search?query=accession:{prot_id}&fields=go_id&format=tsv", timeout=10
            )
            response.raise_for_status()

            go_terms = response.text.split("\n")[1].split("; ")

            self.cache[prot_id] = go_terms
            self._save_cache()
            return go_terms

        except requests.exceptions.RequestException as e:
            logging.warning(f"Unable to fetch annotations for {prot_id}: {str(e)}")
            return ["FETCH_ERROR"]

    def get_annotations(self, proteins: List[str]) -> List[str]:
        """Get GO annotations for list of proteins"""
        results = []

        for prot in tqdm(proteins, desc="Processing Uniprot annotations"):
            if prot in self.cache:
                results.append(",".join(sorted(self.cache[prot])))
            else:
                results.append(",".join(sorted(self._parse(prot))))

        return results
