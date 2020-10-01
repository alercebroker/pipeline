#!/usr/bin/env python

from setuptools import setup, find_namespace_packages

with open("requirements.txt") as f:
    required_packages = f.readlines()

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="db-plugins",
    version="1.0.0",
    description="ALeRCE Database Plugins.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="ALeRCE Team",
    author_email="contact@alerce.online",
    packages=find_namespace_packages(include=["db_plugins.*"]),
    scripts=["scripts/dbp"],
    install_requires=required_packages,
    build_requires=required_packages,
    project_urls={
        "Github": "https://github.com/alercebroker/db-plugins",
        "Documentation": "",
    },
)
