import os
from typing import List
from cherryml.io import read_pickle, write_pickle


def read_transitions_log_likelihood_per_site(
    transitions_log_likelihood_per_site_path: str,
) -> List[List[float]]:
    transitions_log_likelihood_per_site = read_pickle(transitions_log_likelihood_per_site_path)
    if len(transitions_log_likelihood_per_site) == 0:
        raise Exception(
            f"The transitions log likelihood file at "
            f"{transitions_log_likelihood_per_site_path} is empty"
        )
    return transitions_log_likelihood_per_site

def write_transitions_log_likelihood_per_site(
    transitions_log_likelihood_per_site: List[List[float]],
    transitions_log_likelihood_per_site_path: str,
) -> None:
    """
    Serialize a List[List[float]] containing log likelihoods per site 
    corresponding to (x, y, t) triples.
    """
    transitions_log_likelihood_dir = os.path.dirname(
        transitions_log_likelihood_per_site_path
    )
    if not os.path.exists(transitions_log_likelihood_dir):
        os.makedirs(transitions_log_likelihood_dir)
    write_pickle(transitions_log_likelihood_per_site, transitions_log_likelihood_per_site_path)