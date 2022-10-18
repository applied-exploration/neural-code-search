import pandas as pd
import torch
from typing import Tuple, List
from cosine_similarity import CosineSimilaritySearch
from code_embeddings import CodeEmbeddings
from constants import Const, DFCols
import os
from utils import pretty_print_results


def ce_init():
    ce = CodeEmbeddings()

    data_fn = f"{Const.root_data_processed}/data_embedded.parquet"
    if not os.path.exists(data_fn):
        raise FileNotFoundError(f"{data_fn} not found")
    df = pd.read_parquet(data_fn)

    print("Calculating TF-IDF...")

    ce.tfidf_init(df[DFCols.processed_feature.value])

    return ce, df


class NeuralSearch:
    def __init__(self) -> None:
        self.embbeding, library = ce_init()
        self.cos_search = CosineSimilaritySearch(library=library, k=3)

    def preprocess(self, search_corpus: pd.DataFrame) -> None:
        pass

    def _embbed_query(self, query: str) -> torch.Tensor:
        tokens_str = self.embbeding.extract_tokens_str(query)
        e = self.embbeding.get_doc_embedding(tokens_str)

        embbedded_query = torch.from_numpy(e).float()
        return embbedded_query

    def predict(self, query: str) -> Tuple[List[int], pd.DataFrame]:
        embbeded_query = self._embbed_query(query)
        k_best_indicies = self.cos_search.get_similarity(embbeded_query)
        original_snippets = self.cos_search.snippet_lookup(k_best_indicies)

        pretty_print_results(query, original_snippets)

        return k_best_indicies, original_snippets
