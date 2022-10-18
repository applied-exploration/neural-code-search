from ipaddress import v6_int_to_packed
import os.path
import re
from typing import List, Tuple, Dict, Optional

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import sklearn
import transformers
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoModel, AutoTokenizer

from constants import Const, DFCols
import utils

tqdm.pandas()


class CodeEmbeddings:
    def __init__(self):
        self.embeddings = {}
        self.logger = utils.get_logger()

        self.fitted_tfidf_vec: TfidfVectorizer
        self.pipeline: transformers.Pipeline

        if not os.path.exists(Const.embeddings_model_path):
            p = pipeline(
                "feature-extraction",
                model="microsoft/codebert-base",
                tokenizer="microsoft/codebert-base",
            )
            p.save_pretrained(Const.embeddings_model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                Const.embeddings_model_path, return_tensors="pt"
            )
            model = AutoModel.from_pretrained(Const.embeddings_model_path)

            p = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

        self.pipeline = p

    def get_word_embeddings(
        self, code_str: str
    ) -> Optional[List[Tuple[str, List[float]]]]:
        if code_str is None or len(code_str.strip()) == 0:
            return None
        try:
            features = self.pipeline(code_str)[0]
        except Exception as e:
            raise Exception(len(code_str))
        text_arr = code_str.split(" ")
        feat_wo_sos_eos = list(features)[1:-1]
        return list(zip(text_arr, feat_wo_sos_eos))

    def _normalize_1d(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def get_doc_embedding(self, code_str: str) -> Optional[np.ndarray]:
        # implementing the formula from https://ai.facebook.com/blog/neural-code-search-ml-based-code-search-using-natural-language-queries/
        tfidf = None
        if code_str is None or len(code_str.strip()) == 0:
            return None
        we = self.get_word_embeddings(code_str)
        tfidf = self.get_tfidf(code_str)
        e_sum = np.zeros(len(we[0][1]))
        for w, e in we:
            v = self._normalize_1d(e) * tfidf[w]
            e_sum += v
        return self._normalize_1d(e_sum)

    def create_doc_embeddings(self, code_series: pd.Series) -> pd.Series:
        return code_series.progress_apply(lambda x: self.get_doc_embedding(x))

    def get_code_block(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        all_code = soup.findAll("pre", text=True, recursive=True)
        if len(all_code) == 0:
            return ""
        return "\n".join([s.string for s in all_code])

    def extract_tokens_str(self, code_str: str) -> str:
        # TODO: Figure out what's a good for limiting length and token size.
        # Based on the codebert paper token size must be limited to 512, but it still breaks with some longer text,
        # hence limiting code length to 1000

        if code_str is None or len(code_str.strip()) == 0:
            return ""
        code_str = code_str[:1000]
        words = re.findall(r"[a-zA-Z_]+[a-zA-Z0-9_]*", code_str)
        words = words[:512]  # Codebert model can't work with a longer input than 512
        return " ".join(words)

    def extract_code(self, code_series: pd.Series) -> pd.Series:
        # example = '<p>Given a module <code>foo</code> with method <code>bar</code>:</p><pre><code>import foobar = getattr(foo, "bar")result = bar()</code></pre><p><a href="https://docs.python.org/library/functions.html#getattr" rel="noreferrer"><code>getattr</code></a> can similarly be used on class instance bound methods, module-level methods, class methods... the list goes on.</p>'

        return code_series.progress_apply(
            lambda x: self.extract_tokens_str(self.get_code_block(x))
        )

        return df

    def tfidf_init(self, code_series: pd.Series) -> None:
        vectorizer = TfidfVectorizer()
        self.fitted_tfidf_vec = vectorizer.fit(code_series)
        self.tfidf_features = self.fitted_tfidf_vec.get_feature_names_out()

    def get_tfidf(self, text: str) -> Dict[str, float]:
        docterm_matrix = self.fitted_tfidf_vec.transform([text])
        scores = {word: 0 for word in text.split(" ")}
        rows, cols = docterm_matrix.nonzero()
        for row, col in zip(rows, cols):
            scores[self.tfidf_features[col]] = docterm_matrix[row, col]
        return scores


if __name__ == "__main__":
    df = pd.read_parquet(f"{Const.root_data_original}/full_data_small_10.parquet")
