import argparse
import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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

    for parent in index.content:
        matrix = []
        for element_name in index.content[parent]:
            element = index.content[parent][element_name]
            matrix.append(element['embedding'])
        
        mat = torch.stack(matrix)                   # shape (n, d)
        mat = F.normalize(mat, p=2, dim=1) 
        cosim_matrix = mat @ mat.T                        # shape (n, n)
        
        # 2) convert to NumPy
        cosim_np = cosim_matrix.detach().cpu().numpy()

        # 3) get labels for axes
        labels = list(index.content[parent].keys())

        # 4) plot heatmap
        plt.figure(figsize=(8, 8))
        im = plt.imshow(cosim_np, aspect='auto')          # default colormap
        plt.colorbar(im, fraction=0.046, pad=0.04)      # add a color scale

        # 5) annotate ticks (rotate x-labels if long)
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        plt.yticks(range(len(labels)), labels, fontsize=8)

        plt.title(f"Embedding Similarity Heatmap for '{parent}'")
        plt.tight_layout()
        plt.show()

