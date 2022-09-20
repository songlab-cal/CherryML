import os

import numpy as np


def read_contact_map(
    contact_map_path: str,
) -> np.array:
    lines = open(contact_map_path).read().strip().split("\n")
    try:
        num_sites, s = lines[0].split(" ")
        if s != "sites":
            raise Exception
        num_sites = int(num_sites)
    except Exception:
        raise Exception(
            f"Contact map file should start with line '[num_sites] sites', "
            f"but started with: {lines[0]} instead."
        )
    if len(lines) != num_sites + 1:
        raise Exception(
            f"Contact Map at: {contact_map_path} should have {num_sites} rows, "
            f"but has {len(lines) - 1}"
        )
    res = np.zeros(shape=(num_sites, num_sites), dtype=int)
    for i in range(num_sites):
        res[i, :] = np.array(list(map(int, [ch for ch in lines[i + 1]])))
    return res


def write_contact_map(
    contact_map: np.array,
    contact_map_path: str,
) -> None:
    contact_map_dir = os.path.dirname(contact_map_path)
    if not os.path.exists(contact_map_dir):
        os.makedirs(contact_map_dir)
    with open(contact_map_path, "w") as contact_map_file:
        contact_map_file.write(f"{contact_map.shape[0]} sites\n")
        np.savetxt(contact_map_file, contact_map, delimiter="", fmt="%i")
        contact_map_file.flush()
