import os
from typing import List


def read_sites_subset(sites_subset_path: str) -> List[int]:
    lines = open(sites_subset_path).read().strip().split("\n")
    try:
        num_sites, s = lines[0].split(" ")
        if s != "sites":
            raise Exception
        num_sites = int(num_sites)
    except Exception:
        raise Exception(
            f"Sites subset file: {sites_subset_path} should start with line "
            f"'[num_sites] sites', but started with: {lines[0]} instead."
        )
    try:
        if num_sites == 0:
            res = []
        else:
            res = list(map(int, lines[1].split(" ")))
    except Exception:
        raise Exception(
            f"Could nor read sites subset in file: {sites_subset_path}. Lines: "
            f"{lines}"
        )
    if len(res) != num_sites:
        raise Exception(
            f"Sites subset file: {sites_subset_path} was supposed to have "
            f"{num_sites} sites, but it has {len(res)}"
        )
    return res


def write_sites_subset(sites_subset: List[int], sites_subset_path: str) -> None:
    sites_subset_dir = os.path.dirname(sites_subset_path)
    if not os.path.exists(sites_subset_dir):
        os.makedirs(sites_subset_dir)
    res = f"{len(sites_subset)} sites\n" + " ".join(
        list(map(str, sites_subset))
    )
    with open(sites_subset_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()
