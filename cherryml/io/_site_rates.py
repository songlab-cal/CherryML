import os
from typing import List


def read_site_rates(site_rates_path: str) -> List[float]:
    lines = open(site_rates_path).read().strip().split("\n")
    try:
        num_sites, s = lines[0].split(" ")
        if s != "sites":
            raise Exception
        num_sites = int(num_sites)
    except Exception:
        raise Exception(
            f"Site rates file: {site_rates_path} should start with line "
            f"'[num_sites] sites', but started with: {lines[0]} instead."
        )
    try:
        res = list(map(float, lines[1].split(" ")))
    except Exception:
        raise Exception(f"Could nor read site rates in file: {site_rates_path}")
    if len(res) != num_sites:
        raise Exception(
            f"Site rates file: {site_rates_path} was supposed to have "
            f"{num_sites} sites, but it has {len(res)}"
        )
    return res


def write_site_rates(site_rates: List[float], site_rates_path: str) -> None:
    site_rates_dir = os.path.dirname(site_rates_path)
    if not os.path.exists(site_rates_dir):
        os.makedirs(site_rates_dir)
    res = f"{len(site_rates)} sites\n" + " ".join(list(map(str, site_rates)))
    with open(site_rates_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()
