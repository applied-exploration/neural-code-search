import os
import time
from typing import Tuple, List

import pandas as pd
import torch

import utils
from code_embeddings import CodeEmbeddings
from constants import Const, DFCols
from cosine_similarity import CosineSimilaritySearch
from utils import pretty_print_results

logger = utils.get_logger()


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
        self.embedding, library = ce_init()
        self.cos_search = CosineSimilaritySearch(library=library, k=3)

    def preprocess(self, search_corpus: pd.DataFrame) -> None:
        pass

    def _embed_query(self, query: str) -> torch.Tensor:
        e = self.embedding.get_doc_embedding(self.embedding.generate_tokens(query))

        embedded_query = torch.from_numpy(e).float()
        return embedded_query

    def predict(self, query: str) -> Tuple[List[int], pd.DataFrame]:
        start = time.time()
        embedded_query = self._embed_query(query)
        k_best_indices = self.cos_search.get_similarity(embedded_query)
        original_snippets = self.cos_search.snippet_lookup(k_best_indices)

        pretty_print_results(query, original_snippets)
        end = time.time()
        logger.debug(f"Prediction took f{end - start} seconds")
        return k_best_indices, original_snippets
