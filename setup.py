from pathlib import Path

from setuptools import setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="summarize-lib",
    version="2.0.0",
    description="Hierarchical document summarization for SymbolicAI",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    package_dir={"summarize_lib": "src"},
    packages=["summarize_lib"],
    include_package_data=True,
    install_requires=[
        "symbolicai>=1.10.0",
        "pydantic",
        "nest_asyncio",
        "tenacity",
        "loguru",
        "numpy",
        "tiktoken",
        "tokenizers",
    ],
)
