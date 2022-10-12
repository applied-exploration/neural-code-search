#%%
import torch
from torch.nn import CosineSimilarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

# %%

# %%


def embeddings(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    with torch.no_grad():
        output = model(input_ids, decoder_input_ids=input_ids)
        return output[0][0]


# %%
s1 = embeddings("def greet(user): print(f'hello {user}!')")
s2 = embeddings("def greet(person): print(f'hello {person}!')")
s3 = embeddings("x")

cos = CosineSimilarity()
print(cos(s1, s3))
