from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        "transformers==4.15.0",
        "scipy",
        "torch",
        "gensim==4.1.2",
        "icu==0.0.1",
        "scikit_learn==0.24.2",
    ],
)
