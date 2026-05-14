import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

root_folder = "results/raw_results"
output_folder = "results/processed_results/output"

os.makedirs(output_folder, exist_ok=True)

for dataset in os.listdir(root_folder):
    dataset_path = os.path.join(root_folder, dataset)
    results_json = {}
    for seed in os.listdir(dataset_path):
        seed_path = os.path.join(dataset_path, seed)
        file_path = os.path.join(seed_path, "registry_data.json")
        if not os.path.isfile(file_path):
            continue\

        with open(file_path, "r") as f:
            records = json.load(f)

        groups = defaultdict(list)
        for item in records:
            for parent_id in item.get("parents_id", []):
                groups[parent_id].append(item)

        results = []
        results_json_seed = []
        for parent_id, children in groups.items():
            if len(children) < 2:
                continue

            parent = next((obj for obj in records if obj["id"] == parent_id), None)
            parent_pipeline = parent["pipeline"].split(";")

            parent_value = None
            if parent is not None and "fitness" in parent and len(parent["fitness"]) > 0:
                parent_value = parent["fitness"][0]

            all_families = set(f for child in children for f in child["families"])
            for fam in all_families:
                with_fam = []
                without_fam = []

                for child in children:
                    val = child["fitness"][0]

                    if fam in child["families"]:
                        with_fam.append(val)
                    else:
                        without_fam.append(val)

                def stats(arr):
                    if len(arr) == 0:
                        return (None, None, None)

                    arr = np.array(arr, dtype=float)

                    if np.all(np.isnan(arr)):
                        return (None, None, None)

                    return (
                        float(np.nanmean(arr)),
                        float(np.nanmedian(arr)),
                        float(np.nanstd(arr))
                    )

                mean_w, med_w, std_w = stats(with_fam)
                mean_wo, med_wo, std_wo = stats(without_fam)

                results.append({
                    "dataset": dataset,
                    "parent_id": parent_id,
                    "parent_value":parent_value,
                    "family": fam,
                    "component": parent_pipeline[int(fam)],

                    "with_mean": mean_w,
                    "with_median": med_w,
                    "with_std": std_w,

                    "without_mean": mean_wo,
                    "without_median": med_wo,
                    "without_std": std_wo,

                    "count_with": len(with_fam),
                    "count_without": len(without_fam)
                })

                results_json_seed.append({
                    "dataset": dataset,
                    "parent_id": parent_id,
                    "parent_value":parent_value,
                    "family": fam,
                    "component": parent_pipeline[int(fam)],

                    "with_mean": mean_w,
                    "with_median": med_w,
                    "with_std": std_w,

                    "without_mean": mean_wo,
                    "without_median": med_wo,
                    "without_std": std_wo,

                    "count_with": len(with_fam),
                    "count_without": len(without_fam),

                    "values_with": with_fam,
                    "values_without": without_fam,
                })
            
            results_json[seed] = results_json_seed

        df = pd.DataFrame(results)
        if not df.empty and {'parent_id', 'component'}.issubset(df.columns):
            df = df.sort_values(by=['parent_id', 'family'], ascending=[True, True])
        df.to_csv(os.path.join(output_folder, f"{dataset}_{seed}_stats.csv"), index=False)

    with open(os.path.join(output_folder, f"{dataset}_complete_stats.json"), "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4)