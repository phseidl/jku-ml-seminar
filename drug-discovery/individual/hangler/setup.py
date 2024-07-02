from setuptools import setup, find_packages

__version__ = "0.1.0"

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "torch>=1.0",
    "torchdata",
    "pytorch-ignite",
    "seaborn",
    "altair",
    "rdkit-pypi",
    "pandas>1.0",
    "jupyter",
    "matplotlib",
    "numpy",
    "scipy",
    "scikit-learn",
    "boto3",
    "botocore",
    "tqdm",
    "fcd",
    "due @ git+https://github.com/y0ast/DUE.git",
]

setup(
    authors="Stefan Hangler",
    author_email="stefan.hangler@outlook.com",
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: German and English",
        "Programming Language :: Python :: 3.8",
    ],
    description="COATI Model Evaluation with Guacamol Benchmark and Linear Probing",
    install_requires=requirements,
    packages=find_packages(),
    long_description=readme,
    include_package_data=True,
    keywords="coati evaluation",
    name="coati evaluation",
    version=__version__,
    zip_safe=False,
)
