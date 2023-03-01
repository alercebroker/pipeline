#!/usr/bin/env python

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.readlines()

setup(
    name="survey-parser-plugins",
    version="1.0.1",
    description="ALeRCE Survey Parser Plugins",
    author="ALeRCE Team",
    author_email="contact@alerce.online",
    packages=find_packages(),
    install_requires=required_packages,
    build_requires=required_packages,
    project_urls={
        "Github": "https://github.com/alercebroker/survey-parser-plugins",
        "Documentation": "",
    },
)
