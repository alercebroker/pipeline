#!/usr/bin/env python

from setuptools import setup, find_namespace_packages

with open("requirements.txt") as f:
    required_packages = f.readlines()

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="apf_base",
    version="2.4.0",
    description="ALeRCE Alert Processing Framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ALeRCE Team",
    author_email="contact@alerce.online",
    packages=find_namespace_packages(include=["apf.*"]),
    scripts=["scripts/apf"],
    package_data={"": ["*.txt", "Dockerfile"]},
    include_package_data=True,
    install_requires=required_packages,
    build_requires=required_packages,
    project_urls={
        "Github": "https://github.com/alercebroker/APF",
        "Documentation": "https://apf.readthedocs.io/en/latest/index.html",
    },
)
