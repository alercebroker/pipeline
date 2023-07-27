# How to generate documentation

Requirements:

- Use python 3.9
- Upgrade pip with `pip install --upgrade pip`

``` sh
pip install sphinx
pip install -r libs/docs/requirements.txt
cd libs/docs
make html
```

# How to develop documentation

Requirements:

- Use python 3.9
- Upgrade pip with `pip install --upgrade pip`

``` sh
pip install sphinx-autobuild
pip install -r libs/docs/requirements.txt
cd libs
sphinx-autobuild docs docs/_build/html
```

This will leave a server running on localhost:8000 with the built documentation and listening to changes on any file.
