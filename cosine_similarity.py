import pandas as pd
from constants import Const, DFCols
from torch.nn import CosineSimilarity
import torch
from typing import Tuple
import numpy as np


def load_library(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    return df


class CosineSimilarity_Search:
    def __init__(self, library: pd.DataFrame, k: int) -> None:
        self.library = library
        self.vectors = torch.stack(
            [
                torch.from_numpy(vector).float()
                for vector in self.library[DFCols.embedded_feature.value].to_list()
                if vector is not None
            ],
            dim=1,
        )
        self.cos = CosineSimilarity(dim=1)
        self.k = k

    def get_similarity(
        self, embedded_texts: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        embedded_texts_torched = torch.tensor(embedded_texts)
        output = self.cos(embedded_texts_torched.unsqueeze(0), self.vectors.T)

        k_best_similarity_value, k_best_indices = torch.topk(output, dim=0, k=self.k)

        return output, k_best_indices
