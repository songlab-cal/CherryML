import os
from typing import Dict


def get_msa_num_sites(
    msa_path: str,
) -> int:
    """
    Get the number of sites in an MSA (efficiently).
    """
    for i, line in enumerate(open(msa_path, "r")):
        if i == 1:
            return len(line.strip())
    raise Exception("We shouldn't be here!")


def get_msa_num_residues(
    msa_path: str,
    exclude_gaps: bool,
) -> int:
    """
    Get the number of residues in an MSA.

    Assumes that gaps are represented with '.', '-', or '_'.
    """
    msa = read_msa(msa_path)
    num_sequences = len(msa)
    num_sites = len(list(msa.values())[0])
    if not exclude_gaps:
        return num_sequences * num_sites
    else:
        res = sum(
            [
                len(seq) - seq.count(".") - seq.count("-") - seq.count("_")
                for seq in list(msa.values())
            ]
        )
        return res


def get_msa_num_sequences(
    msa_path: str,
) -> int:
    """
    Get the number of sequences in an MSA.
    """
    msa = read_msa(msa_path)
    return len(msa)


def read_msa(
    msa_path: str,
) -> Dict[str, str]:
    msa = {}
    with open(msa_path, "r") as msa_file:
        lines = msa_file.read().strip().split("\n")
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
    return msa


def write_msa(msa: Dict[str, str], msa_path: str) -> None:
    msa_dir = os.path.dirname(msa_path)
    if not os.path.exists(msa_dir):
        os.makedirs(msa_dir)
    res_list = []
    for seq_name in sorted(list(msa.keys())):
        res_list.append(f">{seq_name}\n")
        res_list.append(f"{msa[seq_name]}\n")
    res = "".join(res_list)
    with open(msa_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()
