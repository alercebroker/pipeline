#!/usr/bin/env python

from setuptools import setup
with open("requirements.txt") as f:
    required_packages = f.readlines()

setup(
   name='apf',
   version="0.0.0",
   description='ALeRCE Alert Processing Framework.',
   author='ALeRCE Team',
   author_email='contact@alerce.online',
   packages=['apf'],
   scripts=['scripts/apf'],
   install_requires=required_packages,
   build_requires=required_packages
)
