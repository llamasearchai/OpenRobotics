from setuptools import setup, find_packages

setup(
    name="openrobotics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "langchain>=0.3.0",
        "pyserial>=3.5",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "mlx": [
            "mlx>=0.0.1",
        ],
    },
    author="LlamaSearch AI",
    author_email="info@llamasearch.ai",
    description="A modular and extensible robotics framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearchai/OpenRobotics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
