#%%

from transformers import pipeline


def get_embeddings(text):
    feature_extraction = pipeline(
        "feature-extraction",
        model="microsoft/codebert-base",
        tokenizer="microsoft/codebert-base",
    )
    features = feature_extraction(text)
    return features


if __name__ == "__main__":
    s1 = get_embeddings("def greet(user):\n print(f'hello {user}!')")
