import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))  # Create a single figure outside the loop

# Process each file in the directory
for filename in os.listdir('export/benchmark/step_5/'):
    if not filename.endswith('.json'):  # Skip non-JSON files
        continue
    
    with open(f'export/benchmark/step_5/{filename}') as f:
        data = json.load(f)
    
    # Extract model name from filename
    model_name = filename.split('_top_')[0]
    ranks = [item['rank'] for item in data['success']]
    
    if not ranks:  # Skip empty data sets
        continue
    
    # Calculate cumulative distribution
    sorted_ranks = np.sort(ranks)
    cum_percent = np.arange(1, len(ranks) + 1) / len(ranks) * 100
    
    # Plot each model's data with label
    plt.step(sorted_ranks, cum_percent, where='post', label=model_name)

# Configure plot settings
plt.xlabel('Rank (k)', fontsize=12)
plt.ylabel('Percent <= k', fontsize=12)
plt.title('Cumulative Rank Distribution by Model', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.xlim(0, 100)
plt.savefig('cumulative_ranks_comparison.png', dpi=300)  # Save before showing
plt.show()