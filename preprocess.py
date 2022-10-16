import pandas as pd
from typing import List
from constants import Const, DFCols
import os
import sys
from utils import get_logger
from code_embeddings import CodeEmbeddings

logger = get_logger()


def shorten_dataset(
    data_name: str = "full_data", output_name: str = "full_data_small", size: int = 10
) -> None:
    df = pd.read_parquet(f"{Const.root_data_original}/{data_name}.parquet")
    df = df[: min(size, len(df))]
    df.to_parquet(f"{Const.root_data_original}/{output_name}_{str(size)}.parquet")


def safe_save(df: pd.DataFrame, path: str, name: str, format: str = "csv") -> None:
    if not os.path.exists(path):
        os.makedirs(path)

    if format == "csv":
        df.to_csv(f"{path}/{name}.csv")
    elif format == "parquet":
        df.to_parquet(f"{path}/{name}.parquet")
    else:
        raise ValueError(f"Unrecognized format: {format}")


# shorten_dataset()


def run(
    path: str = Const.root_data_processed,
    name: str = "data_embedded",
    pandas_sample_n=None,
) -> None:

    ce = CodeEmbeddings()
    codeextract_fn = f"{Const.root_data_processed}/data_codeextracted.parquet"
    if False and os.path.exists(codeextract_fn):
        logger.info("Loading cached data (code extract)...")
        df = pd.read_parquet(codeextract_fn)
    else:
        logger.info("Loading raw dataframe...")
        df = pd.read_parquet(f"{Const.root_data_original}/full_data_small_500.parquet")
        df = df[df[DFCols.unprocessed_feature.value].str.contains("<pre>")]

        if pandas_sample_n is not None:
            df = df[df[DFCols.unprocessed_feature.value].str.contains("pandas")]
            df = df.head(pandas_sample_n)
            print(f"Generating {len(df)} pandas only items.")

        logger.info("Extracting code...")

        df = df[df[DFCols.unprocessed_feature.value].str.contains("<pre>")]

        df[DFCols.processed_feature.value] = ce.extract_code(
            df[DFCols.unprocessed_feature.value]
        )

        safe_save(df, Const.root_data_processed, "data_codeextracted", format="parquet")

    df = df[~df[DFCols.processed_feature.value].isnull()]

    if pandas_sample_n is not None:
        df = df[df[DFCols.unprocessed_feature.value].str.contains("pandas")]
        df = df.head(pandas_sample_n)
        print(f"Working with {len(df)} pandas only items.")

    logger.info("Calculating TF-IDF...")

    ce.tfidf_init(df[DFCols.processed_feature.value])

    logger.info("Generating Doc embeddings")

    df[DFCols.embedded_feature.value] = ce.create_doc_embeddings(
        df[DFCols.processed_feature.value]
    )

    logger.info(f"Saving final dataset to {path}/{name}.parquet.")
    safe_save(df, path, name, format="parquet")


if __name__ == "__main__":
    run(pandas_sample_n=500)
