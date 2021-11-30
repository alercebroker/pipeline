from setuptools import setup, find_packages

setup(
    name="alert_archive",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Click",
    ],
    entry_points={
        "console_scripts": [
            "alert_archive = cli.alert_archive:cli",
        ],
    },
)
