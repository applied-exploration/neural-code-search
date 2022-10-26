import os.path
import pickle
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
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
from utils import identity

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
        self, token_list: List[str]
    ) -> Optional[List[Tuple[str, List[float]]]]:
        tokens_ids = self.generate_token_ids(token_list)
        context_embeddings = self.model(torch.tensor(tokens_ids)[None, :])[0][0]
        return list(zip(token_list, context_embeddings))

    @staticmethod
    def _normalize_1d(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def get_doc_embedding(self, token_list: List[str]) -> Optional[torch.Tensor]:
        # implementing the formula from https://ai.facebook.com/blog/neural-code-search-ml-based-code-search-using-natural-language-queries/
        we = self.get_word_embeddings(token_list)
        tfidf = self.get_tfidf(token_list)
        e_sum = np.zeros(len(we[0][1]))
        for w, e in we:
            v = torch.nn.functional.normalize(e, dim=0) * tfidf[w]
            e_sum += v.detach().numpy()
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

    def generate_token_ids(self, token_list: List[str]) -> List[int]:
        tokens_ids = self.tokenizer.convert_tokens_to_ids(token_list)
        return tokens_ids

    def generate_tokens(self, code_str, max_length=512):
        code_tokens = self.tokenizer.tokenize(
            code_str, truncation=True, max_length=max_length - 2
        )  # deduct 2 as we are adding the sep_tokens
        tokens = [self.tokenizer.sep_token] + code_tokens + [self.tokenizer.sep_token]
        return tokens

    def extract_code(self, code_series: pd.Series) -> pd.Series:
        # example = '<p>Given a module <code>foo</code> with method <code>bar</code>:</p><pre><code>import foobar = getattr(foo, "bar")result = bar()</code></pre><p><a href="https://docs.python.org/library/functions.html#getattr" rel="noreferrer"><code>getattr</code></a> can similarly be used on class instance bound methods, module-level methods, class methods... the list goes on.</p>'

        return code_series.progress_apply(
            lambda x: self.generate_tokens(self.get_code_block(x))
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

            vectorizer = TfidfVectorizer(
                tokenizer=identity,
                preprocessor=identity,
                token_pattern=None,
            )
            self.fitted_tfidf = vectorizer.fit(code_series)
            self.tfidf_features = self.fitted_tfidf.get_feature_names_out()
            tfidf_cache_payload = {
                "fitted_tfidf": self.fitted_tfidf,
                "features": self.tfidf_features,
            }
            pickle.dump(tfidf_cache_payload, open(tfidf_cache_fn, "wb"))

    def get_tfidf(self, token_list: List[str]) -> Dict[str, float]:
        docterm_matrix = self.fitted_tfidf.transform([token_list])
        scores = {word: 0 for word in token_list}
        rows, cols = docterm_matrix.nonzero()
        for row, col in zip(rows, cols):
            scores[self.tfidf_features[col]] = docterm_matrix[row, col]
        return scores


if __name__ == "__main__":
    df = pd.read_parquet(f"{Const.root_data_original}/full_data_small_10.parquet")
