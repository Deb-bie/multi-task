"""
Minimal setup.py so the package can be installed with `pip install -e .`
inside the Kubernetes training container or local environment.

Usage:
    pip install -e .                    # editable install (development)
    pip install -e ".[dev]"             # + pytest for running tests
"""

from setuptools import find_packages, setup

setup(
    name="synthrad2025-multitask",
    version="0.1.0",
    description="Multi-task CycleGAN for joint MRI↔CT synthesis and organ segmentation (SynthRAD2025)",
    author="Deborah Asamoah",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "torchmetrics>=1.0.0",
        "SimpleITK==2.3.1",
        "numpy==1.24.3",
        "scipy>=1.10.0",
        "scikit-image>=0.21.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "opencv-python-headless>=4.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
