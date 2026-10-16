# Coding Standards

## Table of Contents

<!-- vim-markdown-toc GFM -->

* [Structure of the repository](#structure-of-the-repository)
* [Installation and dependencies](#installation-and-dependencies)
* [Coding styles](#coding-styles)
  * [Headnote](#headnote)
  * [Linters and formatters](#linters-and-formatters)
  * [Code coverage](#code-coverage)
* [Testing](#testing)
* [Pull requests](#pull-requests)
* [Maintainers](#maintainers)

<!-- vim-markdown-toc -->

## Structure of the repository

```bash
aicon
├── aicon_base          # the framework
├── aicon_ros           # ROS 2 middleware (ament package)
├── aicon_real_robots   # reusable components for the Panda, the RBO hand and the cameras
├── docs
├── examples
│   ├── …
│   ├── drawer_sim      # examples that only run in simulation end in _sim
│   └── object_pushing_real
├── README.md
└── tests
```

- For each individual AICON example, please document as detailed as you can your thought process such that others
  could understand and extend it.
- Examples that only run in simulation get the suffix `_sim`. Each example has a `run_experiment.py` that is run with
  `python -m examples.<name>.run_experiment` from the repository root and imports the rest of the example relatively.
- If your example requires extra dependencies, add them as an optional-dependency group (an "extra") in
  `pyproject.toml`, not to the core dependencies.

## Installation and dependencies

- `pyproject.toml` configures the `aicon` package and its extras; uv and pixi both read it.
  - uv and pixi install the `aicon` package as an editable install. In other words, changes in the code have
    immediate effect.
  - If you make changes to the `pyproject.toml`, update the lock files with `uv lock` and `pixi lock`.
- See the [Installation section of the README](../README.md#installation) for the routes with and without ROS 2.
  For development, add the `dev` extra (linters, formatters and coverage), e.g. `uv sync --extra dev --extra test`.

## Coding styles

### Headnote

- Modules start with a docstring: a title line, optionally a description, and the author(s) and license, followed by
  the module metadata:

  ```python
  """
  Estimators for the Drawer Tutorial

  Authors: Adrian Pfisterer, Patrick Lowin, Vito Mengers
  License: MIT
  """

  __author__ = "Adrian Pfisterer, Patrick Lowin, Vito Mengers"
  __copyright__ = "Copyright 2025 RBO TU Berlin"
  __license__ = "MIT"
  __version__ = "1.0.0"
  ```

- Test files start with a one-line docstring saying what they test.

### Linters and formatters

- Lines are at most 120 characters long.
- Linters:
  - [`flake8`](https://flake8.pycqa.org/en/latest/) configured with [`.flake8`](../.flake8)
  - [`pylint`](https://github.com/pylint-dev/pylint) configured with [`.pylintrc`](../.pylintrc)
- Formatter: [`ruff format`](https://docs.astral.sh/ruff/formatter/), configured in [`pyproject.toml`](../pyproject.toml).
  It only changes the layout of the code, and the CI checks that every file is formatted:
  `ruff format --check $(git ls-files '*.py')`

There are different ways to install/use the linters and formatters:

- Install as extensions in your favorite IDE (eg: VSCode).
- Install as dependencies in Python virtual environment and run in CLI--the recommended way. These packages are included in the `dev` extra.

  ```bash
  pylint $(git ls-files '*.py')
  pylint **/*.py
  ```

- With the `dev` extra installed, `scripts/refactor.sh` formats all files

  ```bash
  ./scripts/refactor.sh
  ```

- Exceptions the CI accepts for now:
  - `flake8` ignores E203 (whitespace before `:` in slices, as the formatter writes it) and C901 (complexity; several
    functions exceed the limit).
  - `pylint` disables the checks listed under `disable=` in `.pylintrc`, and requires a score of at least the
    `fail-under` value there. Raise it whenever warnings are fixed, so the score cannot drop again.

### Code coverage

Dependencies which can be installed with `pip`:

- `coverage`
- `genbadge[coverage]`

Check the shell script `scripts/test.sh`.

## Testing

- Writing unit tests and integration tests ensure long-term stability of the repository. Thus, try to write as many tests as the number of functions and files you create and merge to the `main` branch
- Reading tests also show the inner working of functions, classes, etc.
- If you are new to writing tests, check existing test files and use the template.
- When you change an example's graph, regenerate its `spec.yaml` with `python -m examples.spec_files`;
  `tests/test_spec_files.py` fails while a spec file is outdated.
- Naming convention:
  - Unit test for individual file: `test_filename.py`
  - Integration test: `test_aicon_[setupA_or_taskB].py`
- Run the test locally with

  ```bash
  pytest                # The simplist way
  ./scripts/test.sh -p  # Include also coverage check to update test badges
  ```

## Pull requests

- Don't include unnecessary or unrelated changes in your pull request. Make another pull request instead, in which the changes are relevant.
- Format and test your code locally before pushing to the remote branch.

```bash
./scripts/refactor.sh
./scripts/test.sh --help
```

## Maintainers

- Vito Mengers - <v.mengers@tu-berlin.de>
- Adrian Pfisterer - <adrian.pfisterer@tu-berlin.de>
- Patrick Lowin - <lowin@tu-berlin.de>
