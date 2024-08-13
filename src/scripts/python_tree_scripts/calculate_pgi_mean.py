# dir_path_str should be the folder with pgi topk results
# pgi mean will be saved in the same folder

import numpy as np
import os
from pathlib import Path
from typing import Optional


def get_pgi_name(file_path: Path) -> Optional[str]:
    file_path_str = str(file_path)
    substr_start = file_path_str.rfind("/")
    substr_end = file_path_str.find("_topk_")
    if substr_end < 0:
        return None
    return file_path_str[substr_start + 1 : substr_end]


def load_topk_results(dir_path: Path) -> dict:
    pgi_names = {}
    for file_path in dir_path.iterdir():
        name = get_pgi_name(file_path)
        if not name:
            continue
        if name not in pgi_names:
            pgi_names[name] = 0
        pgi_names[name] += 1
    data = {}
    for name in pgi_names:
        data[name] = []
        for k in range(1, pgi_names[name] + 1):
            file_path = dir_path / f"{name}_topk_{k}.npy"
            data[name].append(np.load(file_path))
    return data


def calc_mean(dir_path: Path, data: dict):
    for name in data:
        big_array = np.array(data[name])
        avg = np.mean(big_array, axis=0)
        np.save(dir_path / f"{name}_mean.npy", avg)


if __name__ == "__main__":
    dir_paths = ["results/wine_model_single/pgi", 
                 "results/wine_model/pgi",
                 "results/housing_model_single/pgi",
                 "results/housing_model/pgi",
                 "results/telemetry_model_single/pgi",
                 "results/telemetry_model/pgi",

                 ]

    for d in dir_paths:
        d = Path(d)
        for pgi_dir in [x[0] for x in os.walk(d)]:
            dir_path = Path(pgi_dir)
            data = load_topk_results(dir_path)
            calc_mean(dir_path, data)
