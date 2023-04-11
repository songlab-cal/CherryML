from typing import Optional

import numpy as np


def create_synthetic_contact_map(
    num_sites: int, num_sites_in_contact: int, random_seed: Optional[int]
) -> np.array:
    if num_sites_in_contact % 2 != 0:
        raise Exception(
            "num_sites_in_contact should be even, but provided: "
            f"{num_sites_in_contact}"
        )
    num_contacting_pairs = num_sites_in_contact // 2
    contact_map = np.zeros(shape=(num_sites, num_sites), dtype=int)
    if random_seed:
        np.random.seed(random_seed)
    sites_in_contact = np.random.choice(
        range(num_sites), num_sites_in_contact, replace=False
    )
    contacting_pairs = [
        (sites_in_contact[2 * i], sites_in_contact[2 * i + 1])
        for i in range(num_contacting_pairs)
    ]
    for i, j in contacting_pairs:
        contact_map[i, j] = contact_map[j, i] = 1
    for i in range(num_sites):
        contact_map[i, i] = 1
    return contact_map
