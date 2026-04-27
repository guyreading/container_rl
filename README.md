# Container RL Env

![PyPI version](https://img.shields.io/pypi/v/container-rl.svg)

An RL environment to simulate the board game Container and train agents in

* GitHub: https://github.com/guyreading/container-rl/
* PyPI package: https://pypi.org/project/container-rl/
* Created by: **[Guy Reading](https://guyreading.github.io/)** | GitHub https://github.com/guyreading | PyPI https://pypi.org/user/guyreading/
* Free software: MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://guyreading.github.io/container_rl/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/container-rl.git
cd container-rl

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `container_rl`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

Container RL Env was created in 2026 by Guy Reading.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
