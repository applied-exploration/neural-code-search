import pandas as pd
from bs4 import BeautifulSoup
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


def get_code_block(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    all_code = soup.findAll("pre")
    return "\n".join([s.string for s in all_code])


def extract_code(df: pd.DataFrame) -> pd.DataFrame:
    # example = '<p>Given a module <code>foo</code> with method <code>bar</code>:</p><pre><code>import foobar = getattr(foo, "bar")result = bar()</code></pre><p><a href="https://docs.python.org/library/functions.html#getattr" rel="noreferrer"><code>getattr</code></a> can similarly be used on class instance bound methods, module-level methods, class methods... the list goes on.</p>'

    df[DFCols.processed_feature.value] = df[DFCols.unprocessed_feature.value].apply(
        lambda x: get_code_block(x)
    )

    return df


def create_embeddings(df: pd.DataFrame, cd: CodeEmbeddings) -> pd.DataFrame:
    df[DFCols.embedded_feature.value] = df[DFCols.processed_feature.value].apply(
        lambda x: cd.get_embeddings(x)
    )

    return df


def safe_save(df: pd.DataFrame, path: str, name: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(f"{path}/{name}.csv")
    df.to_parquet(f"{path}/{name}.parquet")


# shorten_dataset()


def run(path: str = Const.root_data_processed, name: str = "data_embedded") -> None:
    df = pd.read_parquet(f"{Const.root_data_original}/full_data_small_10.parquet").head(
        n=3
    )

    ce = CodeEmbeddings()

    df = extract_code(df)
    df = create_embeddings(df, ce)

    safe_save(df, path, name)


if __name__ == "__main__":
    run()
