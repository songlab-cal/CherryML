import os
from typing import List, Tuple

TransitionsType = List[Tuple[str, str, float]]


def read_transitions(
    transitions_path: str,
) -> TransitionsType:
    transitions = []
    lines = open(transitions_path, "r").read().strip().split("\n")
    if len(lines) == 0:
        raise Exception(f"The transitions file at {transitions_path} is empty")
    for i, line in enumerate(lines):
        if i == 0:
            tokens = line.split(" ")
            if len(tokens) != 2:
                raise ValueError(
                    f"Transitions file at '{transitions_path}' should start "
                    f"with '[NUM_TRANSITIONS] transitions'."
                )
            if tokens[1] != "transitions":
                raise ValueError(
                    f"Transitions file at '{transitions_path}' should start "
                    f"with '[NUM_TRANSITIONS] transitions'."
                )
            if len(lines) - 1 != int(tokens[0]):
                raise ValueError(
                    f"Expected {int(tokens[0])} transitions at "
                    f"'{transitions_path}', but found only {len(lines) - 1}."
                )
        else:
            x, y, t_str = line.split(" ")
            t = float(t_str)
            transitions.append((x, y, t))
    return transitions


def write_transitions(
    transitions: TransitionsType, transitions_path: str
) -> None:
    transitions_dir = os.path.dirname(transitions_path)
    if not os.path.exists(transitions_dir):
        os.makedirs(transitions_dir)
    res = (
        f"{len(transitions)} transitions\n"
        + "\n".join([f"{x} {y} {t}" for (x, y, t) in transitions])
        + "\n"
    )
    with open(transitions_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()
