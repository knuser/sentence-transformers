"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial
import time

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = [
    # glasses - wikipedia
    'Glasses, also known as eyeglasses or spectacles, are devices consisting of glass or hard plastic lenses mounted '
    'in a frame that holds them in front of a person\'s eyes, typically using a bridge over the nose and arms which '
    'rest over the ears.',
    'Glasses are typically used for vision correction, such as with reading glasses and glasses used for '
    'nearsightedness.',
    'Safety glasses provide eye protection against flying debris for construction workers or lab technicians.',
    'These glasses may have protection for the sides of the eyes as well as in the lenses.',
    'Some types of safety glasses are used to protect against visible and near-visible light or radiation.',
    # glasses - wikidata
    # 'Accessories that improve vision.',
    # 'Glasses, subclass of corrective lens.',
    # 'Glasses, subclass of costume accessory.',
    # 'Glasses, subclass of costume accessory.',

    # city - wikipedia
    'A city is a large human settlement.',
    'Cities generally have extensive systems for housing, transportation, sanitation, utilities, land use, and '
    'communication.',
    'Cities density facilitates interaction between people, government organisations and businesses, sometimes '
    'benefiting different parties in the process.',
    # city - wikidata
    # 'Large and permanent human settlement.',
    # 'City, subclass of community.',
    # 'City, subclass of city/town',
    # 'City, subclass of urban area',

    # jaguar cars - wikipedia
    'Jaguar Cars was the company that was responsible for the production of Jaguar cars until its operations were '
    'fully merged with those of Land Rover to form Jaguar Land Rover on 1 January 2013.',
    'Jaguar\'s business was founded as the Swallow Sidecar Company in 1922, originally making motorcycle sidecars '
    'before developing bodies for passenger cars.',
    # jaguar cars - wikidata
    # 'Former British car company.',
    # 'Jaguar, instance of automobile manufacturer.',
    # 'Jaguar, industry automotive industry.',

    # jaguar (animal) - wikipedia
    'The jaguar\'s present range extends from Southwestern United States and Mexico in North America, across much of '
    'Central America, and south to Paraguay and northern Argentina in South America.',
    'Overall, the jaguar is the largest native cat species of the New World and the third largest in the world.',
    'The adult jaguar is an apex predator, meaning it exists at the top of its food chain and is not preyed on in the '
    'wild.',
    # jaguar (animal) - wikidata
    # 'Species of big cat native to the Americas.',
    # 'Jaguar, subclass of mammal.',
]

print('c', len(corpus))
new_corupus = corpus * (10_000 // len(corpus))
print('new_c', len(new_corupus))

start = time.time()
corpus_embeddings = embedder.encode(new_corupus, show_progress_bar=True)
print(f"took {time.time() - start} sec")

# Query sentences:
queries = [
    'Jaguar is the luxury vehicle brand of Jaguar Land Rover, a British multinational car manufacturer with its '
    'headquarters in Whitley, Coventry, England.',
    'The jaguar is a large felid species and the only extant member of the genus Panthera native to '
    'the Americas.',
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

