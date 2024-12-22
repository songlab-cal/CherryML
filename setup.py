from setuptools import setup, find_packages
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


def get_version_from_init(init_path='cherryml/__init__.py'):
    with open(init_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return eval(line.split('=')[-1])


version = get_version_from_init()


extensions = [
    Extension(
        "cherryml._siterm.fast_site_rates",
        ["cherryml/_siterm/fast_site_rates.pyx"],
        include_dirs=[np.get_include()],
        language="c++",  # Tell the compiler to use C++
        extra_compile_args=["-std=c++11", "-O3"],  # Optional: Use C++11 standard
    )
]


setup(
    name='cherryml',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'biotite',
        'black',
        'ete3',
        'flake8',
        'isort',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'parameterized',
        'pytest',
        'scipy',
        'seaborn',
        'threadpoolctl',
        'torch',
        'tqdm',
        'wget',
    ],
    ext_modules=cythonize(extensions),
)
