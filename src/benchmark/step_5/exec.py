import argparse
import json
import os
from copy import deepcopy

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
    parser.add_argument('--export-result',  default='export/benchmark/step_5')
    parser.add_argument('--model-name', default='mxbai', help="Embedding model's name")
    parser.add_argument('--device', default='cpu', help="Device for embedding model")
    parser.add_argument('--batch-size', default=1, help="Batch size used to pre compute embedding")
    parser.add_argument('--top-k', default=10, help="Top-k parameter use for retrieval", type=int)
    args = parser.parse_args()

    with open(args.database_path, 'r') as file:
        database = json.load(file)
    
    with open(args.benchmark_path, 'r') as file:
        benchmark = json.load(file)

    benchmark_name = args.benchmark_path.split('/')[-1]
    model = DICT_MODEL[args.model_name](device=args.device)
    index = FaissIndex(model, database)
    to_do = []

    count = 0
    cumulative_rank = 0
    result = {'success':[], 'failure': []}
    result_full = {'success':[], 'failure': []}
    for entry in tqdm(benchmark):
        query = entry['query']
        constant_fqn = entry['query_constant']['fqn']
        parent = entry['query_constant']['parent']
        relative_name = entry['query_constant']['relative_name']

        constant = database[parent][relative_name]
        
        score = index.query(query, top_k=args.top_k)
        found = False
        new_entry = {
            "rank": -1,
            "query": query,
            "fullname": constant['fullname'],
            "docstring": constant['docstring']
        }
        new_entry_full = deepcopy(new_entry)
        new_entry['full_rank'] = [(float(rank), fqn) for rank, (_, fqn) in enumerate(score)]
        for rank, (_, fqn) in enumerate(score):
            if fqn == constant_fqn:
                count += 1
                cumulative_rank += rank
                new_entry['rank'] = rank
                
                result['success'].append(new_entry)
                result_full['success'].append(new_entry_full)
                found = True
                break
        
        if not found:
            result['failure'].append(new_entry)
            result_full['failure'].append(new_entry_full)
    print(count / len(benchmark)*100)
    print(cumulative_rank/count)
    
    os.makedirs(args.export_result, exist_ok=True)
    export_path = os.path.join(args.export_result, f'{model.name()}_top_{args.top_k}_{benchmark_name}')
    export_debug_path = os.path.join(args.export_result, f'debug_{model.name()}_top_{args.top_k}_{benchmark_name}')
    with open(export_path, 'w') as file:
        json.dump(result, file, indent=4)
    with open(export_debug_path, 'w') as file:
        json.dump(result_full, file, indent=4)

    