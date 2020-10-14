#!/usr/bin/env python

from setuptools import setup, find_namespace_packages
with open("requirements.txt") as f:
    required_packages = f.readlines()

setup(
   name='apf_base',
   version="1.0.1",
   description='ALeRCE Alert Processing Framework.',
   author='ALeRCE Team',
   author_email='contact@alerce.online',
   packages=find_namespace_packages(include=['apf.*']),
   scripts=['scripts/apf'],
   install_requires=required_packages,
   build_requires=required_packages
)
