import pandas as pd
from typing import List
from constants import Const, DFCols
import os
from utils import get_logger
from code_embeddings import CodeEmbeddings

logger = get_logger()


def shorten_dataset(
    data_name: str = "full_data", output_name: str = "full_data_small", size: int = 10
) -> None:
    df = pd.read_parquet(f"{Const.root_data_original}/{data_name}.parquet")
    df = df[: min(size, len(df))]
    df.to_parquet(f"{Const.root_data_original}/{output_name}_{str(size)}.parquet")


def safe_save(df: pd.DataFrame, path: str, name: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(f"{path}/{name}.csv")
    df.to_parquet(f"{path}/{name}.parquet")


# shorten_dataset()


def run(path: str = Const.root_data_processed, name: str = "data_embedded") -> None:
    df = pd.read_parquet(f"{Const.root_data_original}/full_data_small_10.parquet")

    ce = CodeEmbeddings()

    df[DFCols.processed_feature.value] = ce.extract_code(
        df[DFCols.unprocessed_feature.value]
    )
    ce.init_tfidf(df[DFCols.processed_feature.value])
    tfidf = ce.get_tfidf("def self __init__ hello self")
    print(f"tfidf: {tfidf}")

    df[DFCols.embedded_feature.value] = de.create_embeddings(
        df[DFCols.processed_feature.value]
    )
    # safe_save(df, path, name)


if __name__ == "__main__":
    run()
