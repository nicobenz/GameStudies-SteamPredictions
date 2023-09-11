import json
from tqdm import tqdm

with open("data/finished_corpus/corpora/corpus-1-AdventureStrategySimulationRPGPuzzle_cleaned.json", "r") as file_in:
    corpus = json.loads(file_in.read())

additional_stopwords = [
    "nt", "dlc", "dlcs", "rts",
    "got", "simulator", "strategy", "better",
    "buy", "thing", "things", "strategys"
]

for key, value in corpus.items():
    for i in tqdm(range(len(value))):
        cleaned = [item for item in value[i] if item not in additional_stopwords]
        corpus[key][i] = cleaned

with open("data/finished_corpus/corpora/corpus-1-AdventureStrategySimulationRPGPuzzle_cleaned.json", "w") as file_out:
    json.dump(corpus, file_out)
