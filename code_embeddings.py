import os.path
import pickle
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import transformers
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoModel,
    AutoTokenizer,
    RobertaTokenizer,
    RobertaModel,
)

import utils
from constants import Const

tqdm.pandas()

logger = utils.get_logger()


class CodeEmbeddings:
    def __init__(self):
        self.embeddings = {}
        self.logger = utils.get_logger()

        self.tfidf_features: Dict[str, float]
        self.tfidf_features: Optional[np.ndarray[str]] = None
        self.fitted_tfidf: Optional[TfidfVectorizer] = None

        self.tokenizer: Optional[RobertaTokenizer] = None
        self.model: Optional[RobertaModel] = None

        self.pipeline: Optional[transformers.Pipeline] = None

        if not os.path.exists(Const.embeddings_model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", truncate=True, max_length=3
            )
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.pipeline = pipeline(
                "feature-extraction", model=self.model, tokenizer=self.tokenizer
            )
            self.pipeline.save_pretrained(Const.embeddings_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                Const.embeddings_model_path, return_tensors="pt"
            )
            self.model = AutoModel.from_pretrained(Const.embeddings_model_path)

            self.pipeline = pipeline(
                "feature-extraction", model=self.model, tokenizer=self.tokenizer
            )

    def get_word_embeddings(
        self, code_str: str
    ) -> Optional[List[Tuple[str, List[float]]]]:
        if code_str is None or len(code_str.strip()) == 0:
            return None

        features = self.pipeline(code_str)[0]

        text_arr = code_str.split(" ")  # TODO use tokenizer
        feat_wo_sos_eos = list(features)[1:-1]
        return list(zip(text_arr, feat_wo_sos_eos))

    @staticmethod
    def _normalize_1d(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def get_doc_embedding(self, code_str: str) -> Optional[np.ndarray]:
        # implementing the formula from https://ai.facebook.com/blog/neural-code-search-ml-based-code-search-using-natural-language-queries/
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

    @staticmethod
    def get_code_block(text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        all_code = soup.findAll("pre", text=True)
        if len(all_code) == 0:
            return ""
        return "\n".join([s.string for s in all_code])

    @staticmethod
    def extract_tokens_str(code_str: str) -> str:
        # TODO: Figure out what's a good for limiting length and token size.
        # Based on the codebert paper token size must be limited to 512, but it still breaks with some longer text,
        # hence limiting code length to 510

        if code_str is None or len(code_str.strip()) == 0:
            return ""
        code_str = code_str[:510]
        words = re.findall(r"[a-zA-Z0-9]+", code_str)
        words = words[:512]  # Codebert model can't work with a longer input than 512
        return " ".join(words)

    def extract_code(self, code_series: pd.Series) -> pd.Series:
        # example = '<p>Given a module <code>foo</code> with method <code>bar</code>:</p><pre><code>import foobar = getattr(foo, "bar")result = bar()</code></pre><p><a href="https://docs.python.org/library/functions.html#getattr" rel="noreferrer"><code>getattr</code></a> can similarly be used on class instance bound methods, module-level methods, class methods... the list goes on.</p>'

        return code_series.progress_apply(
            lambda x: self.extract_tokens_str(self.get_code_block(x))
        )

    def tfidf_init(self, code_series: pd.Series) -> None:
        tfidf_cache_fn = f"{Const.root_data_processed}/tfidf.pkl"
        if os.path.exists(tfidf_cache_fn):
            logger.debug(f"Loading TF/IDF cache from {tfidf_cache_fn}")
            tfidf_cache_payload = pickle.load(open(tfidf_cache_fn, "rb"))
            self.fitted_tfidf = tfidf_cache_payload["fitted_tfidf"]
            self.tfidf_features = tfidf_cache_payload["features"]
        else:
            logger.debug(
                f"Calculating TF/IDF from scratch and saving it to f{tfidf_cache_fn}"
            )
            vectorizer = TfidfVectorizer()
            self.fitted_tfidf = vectorizer.fit(code_series)
            self.tfidf_features = self.fitted_tfidf.get_feature_names_out()
            tfidf_cache_payload = {
                "fitted_tfidf": self.fitted_tfidf,
                "features": self.tfidf_features,
            }
            pickle.dump(tfidf_cache_payload, open(tfidf_cache_fn, "wb"))

    def get_tfidf(self, text: str) -> Dict[str, float]:
        docterm_matrix = self.fitted_tfidf.transform([text])
        scores = {word: 0 for word in text.split(" ")}
        rows, cols = docterm_matrix.nonzero()
        for row, col in zip(rows, cols):
            scores[self.tfidf_features[col]] = docterm_matrix[row, col]
        return scores


if __name__ == "__main__":
    df = pd.read_parquet(f"{Const.root_data_original}/full_data_small_10.parquet")
