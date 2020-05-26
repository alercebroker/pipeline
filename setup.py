#!/usr/bin/env python

from setuptools import setup, find_namespace_packages
with open("requirements.txt") as f:
    required_packages = f.readlines()

setup(
   name='db-plugins',
   version="0.0.0",
   description='ALeRCE Alert Processing Framework.',
   author='ALeRCE Team',
   author_email='contact@alerce.online',
   packages=find_namespace_packages(include=['plugins.*']),
   scripts=['scripts/dbp'],
   install_requires=required_packages,
   build_requires=required_packages
)

