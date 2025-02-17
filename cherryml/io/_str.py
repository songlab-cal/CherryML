def read_str(s_path: str):
    with open(s_path, "r") as input_file:
        return input_file.read()


def write_str(s: str, s_path: str):
    with open(s_path, "w") as output_file:
        output_file.write(s)
