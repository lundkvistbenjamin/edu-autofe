from setuptools import setup, find_packages

setup(
    name="eduautofe",
    version="0.1.0",
    description="Educational Automated Feature Engineering",
    author="Benjamin Lundkvist",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.7",
)
