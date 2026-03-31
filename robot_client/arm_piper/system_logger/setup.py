"""Setup script for system_logger package."""

from setuptools import setup, find_packages

setup(
    name="system_logger",
    version="0.1.0",
    description="Unified state recording and rewind orchestration for Tidybot",
    author="Tidybot Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
