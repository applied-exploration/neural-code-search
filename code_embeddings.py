import os.path
import re
from typing import List

from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoModel, AutoTokenizer

from constants import Const, DFCols
import utils


class CodeEmbeddings:
    def __init__(self):
        self.embeddings = {}
        self.logger = utils.get_logger()
        self.fitted_tfidf_vec = None  # type: TfidfVectorizer

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

    def get_embeddings(self, text: str):
        features = self.pipeline(text)[0]
        first_vectors = [v[0] for v in features]
        text_arr = text.split(" ")
        self.logger.debug(
            f"Embedded text has {len(text_arr)} words but {len(features)} embeddings have been created.\ntext: {text}\nfirst vectors for all embeddings:\n{first_vectors}"
        )
        return features

    def create_embeddings(self, text_series: pd.Series) -> pd.Series:
        set_series.apply(lambda x: self.get_embeddings(x))

        return df

    def get_code_block(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        all_code = soup.findAll("pre")
        return "\n".join([s.string for s in all_code])

    def extract_tokens_str(self, text: str) -> str:
        words = re.findall(r"[a-zA-Z_]+[a-zA-Z0-9_]*", text)
        return " ".join(words)

    def extract_code(self, text_Series: pd.Series) -> pd.Series:
        # example = '<p>Given a module <code>foo</code> with method <code>bar</code>:</p><pre><code>import foobar = getattr(foo, "bar")result = bar()</code></pre><p><a href="https://docs.python.org/library/functions.html#getattr" rel="noreferrer"><code>getattr</code></a> can similarly be used on class instance bound methods, module-level methods, class methods... the list goes on.</p>'

        return code_series.apply(
            lambda x: self.extract_tokens_str(self.get_code_block(x))
        )

        return df

    def init_tfidf(self, code_series: pd.Series) -> None:
        vectorizer = TfidfVectorizer()
        self.fitted_tfidf_vec = vectorizer.fit(code_series)
        self.tfidf_features = self.fitted_tfidf_vec.get_feature_names_out()

    def get_tfidf(self, text: str):
        docterm_matrix = self.fitted_tfidf_vec.transform([text])
        scores = {word: 0 for word in text.split(" ")}
        rows, cols = docterm_matrix.nonzero()
        for row, col in zip(rows, cols):
            scores[self.tfidf_features[col]] = docterm_matrix[row, col]
        print(scores)
        return scores


if __name__ == "__main__":
    ce = CodeEmbeddings()

    print("Strange thing #1: you always get n+2 embeddings for an n word text:")
    e = ce.get_embeddings("def")
    e = ce.get_embeddings("def hello")
    e = ce.get_embeddings("def hello print")

    print(
        "Strange thing #2: we have different embeddings for the same world both in the same text and accross texts:"
    )
    e = ce.get_embeddings("def")
    e = ce.get_embeddings("def def def")
