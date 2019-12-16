"""
This examples clusters different sentences that come from the same wikipedia article.

It uses the 'wikipedia-sections' model, a model that was trained to differentiate if two sentences from the
same article come from the same section or from different sections in that article.
"""
import json

from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm


def blank_relation_example(d, tokenizer):
    try:
        head, tail = d['head'].split(' | ')
    except ValueError:
        print(d['head'])
        raise

    content = d['text']
    # head_pos = content.index(head)
    # tail_pos = content.index(tail)

    content = content.replace(head, ' '.join([BLANK_HEAD] * len(tokenizer.tokenize(head))))
    content = content.replace(tail, ' '.join([BLANK_TAIL] * len(tokenizer.tokenize(head))))

    return content, d['tail']


source_file = Path.home() / "ml/questions_gen/data/fewrel_combined_train.json"
assert source_file.is_file()


# embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# extending tokens hack
oryg_tokenizer_len = len(embedder._modules['0'].tokenizer)
BLANK_HEAD = '[BLANK_HEAD]'
BLANK_TAIL = '[BLANK_TAIL]'
e_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]', BLANK_HEAD, BLANK_TAIL]
embedder._modules['0'].tokenizer.add_tokens(e_tokens)
assert oryg_tokenizer_len + len(e_tokens) == len(embedder._modules['0'].tokenizer)
embedder._modules['0'].bert.resize_token_embeddings(len(embedder._modules['0'].tokenizer))


corpus = [
    # ("Bushnell is located at 40°33′6″N 90°30′29″W (40.551667, -90.507921).", "Geography"),
]

with open(source_file, "r") as f:
    data = json.load(f)

for d in tqdm(data, desc=f"Building sentences"):
    corpus.append(blank_relation_example(d, embedder._modules['0'].tokenizer))

sentences = [row[0] for row in corpus]

num_clusters = len(set([row[1] for row in corpus]))
print(f"Num of clusters: {num_clusters}")

corpus_embeddings = embedder.encode(sentences, show_progress_bar=True)
