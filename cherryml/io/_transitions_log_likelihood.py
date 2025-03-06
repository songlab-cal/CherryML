import os
from typing import List

TransitionsLogLikelihoodType = List[float]


def read_transitions_log_likelihood(
    transitions_log_likelihood_path: str,
) -> List[float]:
    transitions_log_likelihood = []
    lines = (
        open(transitions_log_likelihood_path, "r").read().strip().split("\n")
    )
    if len(lines) == 0:
        raise Exception(
            f"The transitions log likelihood file at "
            f"{transitions_log_likelihood_path} is empty"
        )
    for i, line in enumerate(lines):
        if i == 0:
            tokens = line.split(" ")
            if len(tokens) != 2:
                raise ValueError(
                    "Transitions log likelihoodfile at "
                    f"'{transitions_log_likelihood_path}' should start "
                    f"with '[NUM_TRANSITIONS] transitions'."
                )
            if tokens[1] != "transitions":
                raise ValueError(
                    "Transitions log likelihood file at "
                    f"'{transitions_log_likelihood_path}' should start "
                    f"with '[NUM_TRANSITIONS] transitions'."
                )
            if len(lines) - 1 != int(tokens[0]):
                raise ValueError(
                    f"Expected {int(tokens[0])} transitions at "
                    f"'{transitions_log_likelihood_path}', but found only "
                    f"{len(lines) - 1}."
                )
        else:
            ll = float(line)
            transitions_log_likelihood.append(ll)
    return transitions_log_likelihood


def write_transitions_log_likelihood(
    transitions_log_likelihood: List[float],
    transitions_log_likelihood_path: str,
) -> None:
    transitions_log_likelihood_dir = os.path.dirname(
        transitions_log_likelihood_path
    )
    if not os.path.exists(transitions_log_likelihood_dir):
        os.makedirs(transitions_log_likelihood_dir)
    res = (
        f"{len(transitions_log_likelihood)} transitions\n"
        + "\n".join([str(ll) for ll in transitions_log_likelihood])
        + "\n"
    )
    with open(transitions_log_likelihood_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()
