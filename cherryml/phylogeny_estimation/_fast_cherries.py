import os

def _make_fast_cherries_and_return_bin_path(remake=False) -> str:
    """
    makes the binary and returns it's path
    """
    dir_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mcle_cherries"
    )
    bin_path = os.path.join(dir_path, "build/mcle_cherries")
    if remake or not os.path.exists(bin_path):
        with pushd(os.path.join(dir_path,"blossom5-v2.05.src")):
            os.system("make clean && make all")

        with pushd(dir_path):
            os.system("make clean && make main")
            if not os.path.exists(bin_path):
                raise Exception("Was not able to build mcle_cherries")
    return bin_path

