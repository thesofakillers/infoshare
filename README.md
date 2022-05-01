# Studying information-sharing in BERT

## Requirements and Setup

Details such as python and package versions can be found in the generated
[pyproject.toml](pyproject.toml) and [poetry.lock](poetry.lock) files.

We recommend using an environment manager such as
[conda](https://docs.conda.io/en/latest/). After setting up your environment
with the correct python version, please proceed with the installation of the
required packages

For [poetry](https://python-poetry.org/) users, getting setup is as easy as
running

```terminal
poetry install
```

We also provide a [requirements.txt](requirements.txt) file for
[pip](https://pypi.org/project/pip/) users who do not wish to use poetry. In
this case, simply run

```terminal
pip install -r requirements.txt
```

This `requirements.txt` file is generated by running the following

```terminal
poetry export --without-hashes -o requirements.txt
```
