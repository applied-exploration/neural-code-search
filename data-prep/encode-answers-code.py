#%%
import torch
from torch.nn import CosineSimilarity
from transformers import pipeline


def embeddings(text):
    feature_extraction = pipeline(
        "feature-extraction",
        model="microsoft/codebert-base",
        tokenizer="microsoft/codebert-base",
    )
    features = feature_extraction(text)
    return features


s1 = embeddings("def greet(user):\n print(f'hello {user}!')")
s2 = embeddings("def greet(person):\n print(f'hello {person}!')")
s3 = embeddings("x")


cos = CosineSimilarity()
print(cos(s1, s3))
