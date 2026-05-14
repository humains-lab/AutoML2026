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
            values_without = entry.get("values_without", [])
            if not values_without:
                continue

            comp_name = extract_component_name(entry["component"])
            parent_value = entry.get("parent_value", None)

            if parent_value is None or parent_value == 0:
                continue
            
            values = entry.get("values_with", [])
            
            values = pd.to_numeric(values, errors='coerce')
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) == 0:
                continue
            
            mean_value = np.mean(values)

            improvement_pct = ((mean_value - parent_value) / parent_value) * 100
            improvement_pct = 100 - abs(improvement_pct)
            
            rows.append({
                "dataset": dataset_name,
                "component": comp_name,
                "value": improvement_pct
            })

df = pd.DataFrame(rows)
df_grouped = df.groupby(["dataset", "component"], as_index=False)["value"].mean()

heatmap_df = df_grouped.pivot(index="dataset", columns="component", values="value")

component_order = heatmap_df.mean(axis=0).sort_values(ascending=False).index
heatmap_df = heatmap_df[component_order]

plt.figure(figsize=(14, 9))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",      
    center=80,            
    linewidths=0.5,
    linecolor="gray",
)

plt.xlabel("Component")
plt.ylabel("Dataset")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

Path("heatresults/images/output").mkdir(parents=True, exist_ok=True)
plt.savefig("results/images/output/heatmap.pdf", dpi=300)

plt.show()