#!/usr/bin/env python
from setuptools import setup

import os

main_package = "alerce_classifiers"
sub_packages = [
    sp[0] for sp in os.walk(main_package) if not sp[0].endswith("__pycache__")
]

extras_require = {
    sp.split("/")[-1]: [
        x for x in open(os.path.join(sp, "requirements.txt")).read().split("\n")
    ]
    for sp in sub_packages
    if os.path.exists(os.path.join(sp, "requirements.txt"))
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

packages = [sp.replace("/", ".") for sp in sub_packages]

with open("requirements.txt") as f:
    install_requires = f.readlines()

setup(
    name="alerce_classifiers",
    version="0.0.9",
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
)
