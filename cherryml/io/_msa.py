import os
from typing import Dict


def read_msa(
    msa_path: str,
) -> Dict[str, str]:
    msa = {}
    lines = open(msa_path, "r").read().strip().split("\n")
    if len(lines) % 2 != 0:
        raise Exception(
            f"The MSA at {msa_path} should have an even number of lines"
        )
    if len(lines) == 0:
        raise Exception(f"The MSA at {msa_path} is empty")
    msa_size = len(lines) // 2
    for i in range(msa_size):
        if not lines[2 * i].startswith(">"):
            raise Exception(
                f"MSA at {msa_path}: at line {2 * i} expected '>[seq_name]' but"
                f" found {lines[2 * i]}"
            )
        seq_name = lines[2 * i][1:]
        seq = lines[2 * i + 1]
        msa[seq_name] = seq
    if len(set([len(seq) for seq in msa.values()])) != 1:
        raise Exception(
            f"MSA at {msa_path}: All sequences should have the same length."
        )
    return msa


def write_msa(msa: Dict[str, str], msa_path: str) -> None:
    msa_dir = os.path.dirname(msa_path)
    if not os.path.exists(msa_dir):
        os.makedirs(msa_dir)
    res = ""
    for seq_name in sorted(list(msa.keys())):
        res += f">{seq_name}\n"
        res += f"{msa[seq_name]}\n"
    with open(msa_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()


def get_msa_num_sites(msa_path: str) -> int:
    """
    Quick way to get the number of sites in an MSA.
    """
    with open(msa_path, "r") as msa_file:
        for (i, line) in enumerate(msa_file):
            if i == 1:
                return len(line.strip())
