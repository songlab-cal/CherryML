import os
from typing import List, Optional, Tuple


def write_log_likelihood(
    log_likelihood: Tuple[float, Optional[List[float]]],
    log_likelihood_path: str,
) -> None:
    log_likelihood_dir = os.path.dirname(log_likelihood_path)
    if not os.path.exists(log_likelihood_dir):
        os.makedirs(log_likelihood_dir)
    ll, lls = log_likelihood
    res = ""
    res += f"{ll}\n"
    if lls is not None:
        res += f"{len(lls)} sites\n"
        res += " ".join(list(map(str, lls)))
    open(log_likelihood_path, "w").write(res)


def read_log_likelihood(
    log_likelihood_path: str,
) -> Tuple[float, Optional[List[float]]]:
    lines = open(log_likelihood_path, "r").read().strip().split("\n")
    ll = float(lines[0])
    if len(lines) == 1:
        return ll, None
    try:
        num_sites, s = lines[1].split(" ")
        if s != "sites":
            raise Exception
        num_sites = float(num_sites)
    except Exception:
        raise Exception(
            f"Log likelihood file at:{log_likelihood_path} "
            f"should have second line '[num_sites] sites', "
            f"but had second line: {lines[1]} instead."
        )
    lls = list(map(float, lines[2].split(" ")))
    if len(lls) != num_sites:
        raise Exception(
            f"Log likelihood file at:{log_likelihood_path} "
            f"should have {num_sites} values in line 3,"
            f"but had {len(lls)} values instead."
        )
    return ll, lls
