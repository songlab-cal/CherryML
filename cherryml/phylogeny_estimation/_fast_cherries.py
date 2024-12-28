import os

def _make_fast_cherries_and_return_bin_path(remake=False) -> str:
    """
    makes the binary and returns it's path
    """
    dir_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "FastCherries"
    )
    bin_path = os.path.join(dir_path, "build/mcle_cherries")
    if remake or not os.path.exists(bin_path):
        with pushd(dir_path):
            os.system("make clean && make main")
            if not os.path.exists(bin_path):
                raise Exception("Was not able to build fast cherries")
    return bin_path

