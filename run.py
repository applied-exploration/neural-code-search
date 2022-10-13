import torch
from torch.nn import CosineSimilarity


cos = CosineSimilarity()


print(cos(s1, s3))
