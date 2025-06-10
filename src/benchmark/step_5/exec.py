import argparse
import json

from tqdm import tqdm

from src.models.gteqwen import GteQwenEmbedding
from src.models.mxbai import MxbaiEmbedding
from src.models.qwen_embedding import Qwen3Embedding600m, Qwen3Embedding4b, Qwen3Embedding8b
from src.index.cosim_index import FaissIndex

DICT_MODEL = {
    "gte_qwen": GteQwenEmbedding,
    "mxbai": MxbaiEmbedding,
    "qwen_embedding_600m": Qwen3Embedding600m,
    "qwen_embedding_4b": Qwen3Embedding4b,
    "qwen_embedding_8b": Qwen3Embedding8b,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database-path', default='export/output/step_3/result.json', help='Database path')
    parser.add_argument('--benchmark-path', default='export/benchmark/step_4/result_outside_file.json', help='Benchmark path')
    parser.add_argument('--model-name', default='mxbai', help="Embedding model's name")
    parser.add_argument('--device', default='cpu', help="Device for embedding model")
    args = parser.parse_args()

    with open(args.database_path, 'r') as file:
        database = json.load(file)
    
    with open(args.benchmark_path, 'r') as file:
        benchmark = json.load(file)

    model = DICT_MODEL[args.model_name](device=args.device)
    index = FaissIndex(model, database)
    to_do = []

    count = 0
    cumulative_rank = 0
    for entry in tqdm(benchmark):
        query = entry['query']
        constant_fqn = entry['query_constant']['fqn']

        score = index.query(query, top_k=10)
        for rank, (_, fqn) in enumerate(score):
            if fqn == constant_fqn:
                count += 1
                cumulative_rank += rank
    
    print(count / len(benchmark)*100)
    print(cumulative_rank/count)
        


    