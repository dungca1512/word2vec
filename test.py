import io
import math
import numpy as np

def load_vectors(fname):
    vocab = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split(' ')
            data[-1] = data[-1].strip('\n')
            vocab.append({"token": data[0],
                          "vector": data[1:]})
    return vocab

def cosine_distance(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine_similarity = dot_product / (norm_A * norm_B)
    return 1 - cosine_similarity

def get_vector(token, data):
    for item in data:
        if item["token"] == token:
            return np.array(item["vector"])

data = load_vectors("./glove.6B.50d.txt")

token1 = input("Token 1: ")
token2 = input("Token 2: ")

vector1 = get_vector(token1, data)
vector2 = get_vector(token2, data)

print(vector1.shape)
print(vector2.shape)

# distance = cosine_distance(vector1, vector2)
# print(f"Khoảng cách Cosine giữa {token1} và {token2}: {distance}")