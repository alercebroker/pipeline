#!/usr/bin/env python
from setuptools import setup

import os

main_package = "alerce_classifiers"
sub_packages = [
    sp
    for sp in os.listdir(main_package)
    if os.path.isdir(os.path.join(main_package, sp))
]

extras_require = {
    sp: open(os.path.join(main_package, sp, "requirements.txt")).read().split()
    for sp in sub_packages
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

packages = [main_package] + [f"{main_package}.{sp}" for sp in sub_packages]

setup(
    name="alerce_classifiers",
    version="0.0.1",
    packages=packages,
    extras_require=extras_require,
    include_package_data=True,
)
