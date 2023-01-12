import os


def write_float(
    val: float,
    float_path: str,
) -> None:
    float_dir = os.path.dirname(float_path)
    if not os.path.exists(float_dir):
        os.makedirs(float_dir)
    open(float_path, "w").write(str(val))


def read_float(
    float_path: str,
) -> float:
    lines = open(float_path, "r").read().strip().split("\n")
    ll = float(lines[0])
    return ll
