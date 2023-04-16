import pickle
from typing import Any


def read_pickle(
    pickle_path: str,
):
    with open(pickle_path, "rb") as intput_file:
        return pickle.load(intput_file)


def write_pickle(
    obj: Any,
    output_path: str,
):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
