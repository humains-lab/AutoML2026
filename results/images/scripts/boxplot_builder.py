import json
import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def extract_component_name(component_str):
    return component_str.split('(')[0]

rows = []

for file in glob.glob("results/processed_results/output/*_complete_stats.json"):
    dataset_name = os.path.basename(file).replace("_complete_stats.json", "")
    
    with open(file) as f:
        data = json.load(f)
    
    for execution_id, entries in data.items():
        for entry in entries:
            comp_name = extract_component_name(entry["component"])
            
            values_with = pd.to_numeric(entry.get("values_with", []), errors='coerce')
            values_without = pd.to_numeric(entry.get("values_without", []), errors='coerce')
            
            values_with = [v for v in values_with if not np.isnan(v)]
            values_without = [v for v in values_without if not np.isnan(v)]
            
            if len(values_without) == 0:
                continue
            
            for v in values_with:
                rows.append({
                    "dataset": dataset_name,
                    "component": comp_name,
                    "value": v,
                    "mode": "with"
                })
            
            for v in values_without:
                rows.append({
                    "dataset": dataset_name,
                    "component": comp_name,
                    "value": v,
                    "mode": "without"
                })

df = pd.DataFrame(rows)

for dataset in df["dataset"].unique():
    subset = df[df["dataset"] == dataset]
    
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(
        data=subset,
        x="component",
        y="value",
        hue="mode",
        dodge=True,
        linewidth=1,
        fliersize=3,
        gap=0.15,
        width=0.5,
    )

    plt.legend(title='')
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.xlabel("Component")
    plt.ylabel("Performance")

    Path("results/images/output").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"results/images/output/{dataset}_boxplots.pdf", dpi=300)