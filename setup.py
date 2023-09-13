from setuptools import setup, find_packages


def get_version_from_init(init_path='cherryml/__init__.py'):
    with open(init_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return eval(line.split('=')[-1])


version = get_version_from_init()

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
)
