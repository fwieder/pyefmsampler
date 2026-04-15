import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="pyefmsampler",
    version="0.1.0",
    author="Frederik Wieder",
    description="pyEFMsampler",
    url="https://github.com/fwieder/pyefmsampler",
    
    package_dir={"": "src"},                 # 🔥 THIS LINE IS THE FIX
    packages=find_packages(where="src"),     # already correct
    
    long_description=read("README.md"),      # ⚠️ fix capitalization
    
    install_requires=[
        "numpy",
        "efmtool",
        "scipy",
        "cobra",
        "tqdm",
        "umap-learn",
        "matplotlib"
    ],
)