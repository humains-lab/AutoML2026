import os
import subprocess
import sys

DATASETS_DIR = "data"

dataset_dirs = [
        os.path.join(DATASETS_DIR, d)
        for d in os.listdir(DATASETS_DIR)
        if os.path.isdir(os.path.join(DATASETS_DIR, d))
    ]

seeds = 10
for dataset_path in dataset_dirs:
    dataset_name = os.path.basename(dataset_path)
    for seed in range(seeds):
        output_file = f"results/logs/{dataset_name}/output_{seed}.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as out_file:
            subprocess.run(
                [sys.executable, "code/main.py", dataset_path, str(seed)],
                stdout=out_file,
                stderr=subprocess.STDOUT,
                check=False
            )