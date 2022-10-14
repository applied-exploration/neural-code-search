import os.path

from constants import Const
from transformers import pipeline, AutoModel, AutoTokenizer


class CodeEmbeddings:
    def __init__(self):
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
        features = self.pipeline(text)
        return features


if __name__ == "__main__":
    ce = CodeEmbeddings()
    e = ce.get_embeddings("hello world")
    print(e)
