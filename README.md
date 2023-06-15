# ALeRCE Classifiers 

A mono-repo that contains all implementations of classifier models.

# Steps to implement my classifier
1. Create a new package in the folder `alerce_classifiers`.
2. Create `__init__.py`, `requirements.txt` and `model.py`:
- In `requiremenst.txt` put all python packages required for run your model.
- In `model.py` create a class that implements all methods of a classifier from [alerce_base_model](https://github.com/alercebroker/alerce_base_model). 
- In `__init__.py` make sures that export your model.
3. Create tests cases for your model.
4. Push your changes to a new branch with the format `model/<your_model_name>`.

# Advices for your implementation
The repository has code with good practices (sort imports, codestyle, etc). We recommend you use `pre-commit` tool: 
1. Install `pre-commit` with
```bash
pip install pre-commit
```
2. Install the git hook scripts. Run pre-commit install to set up the git hook scripts.
```bash
pre-commit install
```
3. When you do a commit, the `pre-commit` warning you if some standard is not satisfied.

# How to work setup.py?
The `setup.py` script allows to pack all the mini-packages on `alerce_classifiers` folder in spited packages. So you can install specific mini-packages in your project:

```commandline
pip install https://${GH_TOKEN}@github.com/alercebroker/alerce_classifiers.git@0.0.6#egg=alerce_classifiers[balto]
```

Or in your local environment:
```commandline
pip install .[balto]
```
So you will install the specific requirements for this model (and not install all the classifiers).

# What are the utils?

In the `utils` package exists mappers that allow transform stream data to readable data for classifiers.
For example the `ELAsTiCCMapper` transform light curves and features of the stream to data readable for ELAsTiCC models.

# How to use my classifier?

Once installed the model, you can use the model just with:

```python
from alerce_classifiers.balto import BaltoClassifier

light_curves_dataframe = ...
model = BaltoClassifier("MODEL_PATH", "HEADER_QUANTILES_PATH")
predictions = model.predict_proba(light_curves_dataframe)
```


