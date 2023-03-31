#!/usr/bin/env python

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.readlines()

setup(
    name="survey-parser-plugins",
    version="2.1.0",
    description="ALeRCE Survey Parser Plugins",
    author="ALeRCE Team",
    author_email="contact@alerce.online",
    packages=find_packages(),
    install_requires=required_packages,
    extras_require={
        "dev": ["pytest", "coverage", "black", "python-snappy", "fastavro>=0.22.0,<1.5.0"]
    },
    build_requires=required_packages,
    project_urls={
        "Github": "https://github.com/alercebroker/survey-parser-plugins",
        "Documentation": "",
    },
)
