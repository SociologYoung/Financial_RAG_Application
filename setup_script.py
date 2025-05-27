from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sec-finance-rag-assistant",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A RAG application for analyzing SEC 10-K filings with AI-powered Q&A",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sec-finance-rag-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="sec, finance, rag, ai, llm, financial-analysis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/sec-finance-rag-assistant/issues",
        "Source": "https://github.com/yourusername/sec-finance-rag-assistant",
        "Documentation": "https://github.com/yourusername/sec-finance-rag-assistant#readme",
    },
)