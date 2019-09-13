"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = [
    'City',
    'Highway',
    'Industry',
    'Predator',
    'Animal',
    'Glasses',
    'Window',
    'Wheel',
]
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = [
    'Jaguar', 'Car', 'Jaguar car', 'Jaguar panthera',
]
query_embeddings = embedder.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 20
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print(f"\nTop {closest_n} most similar sentences in corpus:")

    for i, (idx, distance) in enumerate(results[0:closest_n]):
        print(f'\t{i}\t', corpus[idx].strip(), "(Score: %.4f)" % (1-distance))

