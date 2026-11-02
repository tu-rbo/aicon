# AICONLS

## About
AICONLS is a linter and Language Server for the [AICON](https://rbo.gitlab-pages.tu-berlin.de/pages/aicon/) framework.  
It uses a project-local `spec.yaml` as source of truth and validates Python components against that specification.

## Requirements
- Python 3.10+
- AICON project with a `spec.yaml` in the project root
- For `--dynamic` checks: `torch` and `tensordict` installed (`pip install -e ".[dynamic]"`) and the AICON framework (`aicon_base`) importable in the same environment. The static checks do not need them.

## `spec.yaml` structure

`spec.yaml` is the source of truth for AICONLS. It usually contains top-level `components`, `goals`, and `connections` mappings.

Top-level fields:
- `components` (optional, recommended): mapping of component instance names to component definitions
- `goals` (optional): mapping of goal instance names to goal definitions
- `connections` (optional): mapping of connection instance names to connection definitions

### Components

Each component entry supports:
- `component_type` (required): one of `Sensor`, `Action`, `Estimator`
- `class_name` (optional): implementation class name; if omitted, the component key is used
- `dtype` (optional): tensor dtype string, e.g. `float32`, `float64`, `double`, `int64`
- `quantities` (optional): mapping of quantity names to quantity definitions
- `goals` (optional): list of goal references (by goal instance name or goal class name)

Each quantity definition supports:
- `q_type` (optional): one of `Generic`, `Gaussian`, `Categorical`, `Particle` (defaults to `Generic`)

`q_type: Generic` (default):
- `shape` (required): tuple-like string of integers, e.g. `(6,)`, `(3, 3)`, `(1, 4)`

`q_type: Gaussian`:
- `dim` (required): positive integer dimensionality
- `uncertainty_name` (required): name of the covariance quantity
- shape is derived as `(dim,)`
- a reference uncertainty quantity named by `uncertainty_name` is added with shape `(dim, dim)`

`q_type: Categorical`:
- `dim` (required): positive integer dimensionality
- shape is derived as `(dim,)`

`q_type: Particle`:
- `dim` (required): positive integer dimensionality per particle
- `particle_count` (required): positive integer number of particles
- `particle_weights_name` (required): name of the weights quantity
- shape is derived as `(particle_count, dim)`
- a reference weights quantity named by `particle_weights_name` is added with shape `(particle_count,)`

### Goals

Each goal entry supports:
- `class_name` (optional): implementation class name; if omitted, the goal key is used

### Connections

Each connection entry supports:
- `class_name` (optional): implementation class name; if omitted, the connection key is used
- `components` (optional): list of component instance names participating in that connection

### Example

```yaml
components:
  GripperAction:
    component_type: Action
    quantities:
      gripper_activation:
        shape: (1,)
  DrawerPosEstimator:
    component_type: Estimator
    class_name: DrawerPositionEstimator
    quantities:
      position_drawer:
        q_type: Gaussian
        dim: 3
        uncertainty_name: uncertainty_drawer
      drawer_state:
        q_type: Categorical
        dim: 3
  KinematicJointEstimator:
    component_type: Estimator
    goals:
      - ReduceJointStateDifference
    quantities:
      kinematic_joint:
        q_type: Particle
        dim: 6
        particle_count: 64
        particle_weights_name: uncertainty_joint
connections:
  GraspedDrawerKinematics:
    class_name: EEDrawerGraspedConnection
    components:
      - DrawerPosEstimator
      - GripperAction
      - KinematicJointEstimator
goals:
  ReduceJointStateDifference:
    class_name: DrawerOpenViaJointGoal
```

## Installation
### Install from source

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install AICONLS in editable mode:

```bash
python -m pip install --upgrade pip
pip install -e .
```

3. Verify the installation:

```bash
aiconls --version
```

This installs the `aiconls` CLI entrypoint in the active virtual environment.

### Development setup

If you want to run all local quality checks used by this repository, install the dev tools too:

```bash
pip install -e ".[test]"
pip install pyright ruff pre-commit
```

## CLI usage
Scan an AICON project and print diagnostics:

```bash
aiconls scan /path/to/aicon-project
```

Useful options:
- `--version` (print CLI version)
- `--format stylish|json` (default: `stylish`)
- `--pretty` (pretty JSON formatting)
- `--output <file>` (write report to file)
- `--fail-on none|error|warning` (controls exit code; default: `warning`)
- `--color auto|always|never` (ANSI colors for `stylish`; default: `auto`)
- `--fix` (insert missing `cls_quantity_names_and_shapes` definitions from `spec.yaml` before scanning)
- `--dynamic` (enable dynamic linting mode; instantiates the parsed network model from `spec.yaml`, then runs 3 dynamic-check cycles in this order: validate runtime quantity shapes (`after initialization`/`after runtime pass N`), validate estimator `f_func` and connection `connection_func` via `torch.func.jacfwd` including all-zero Jacobian checks, and run one network sequence pass with randomized sensor values)

Examples:

```bash
aiconls scan . --format json --pretty
aiconls scan . --fail-on error
aiconls scan . --color always
aiconls scan . --fix
```

### Ignoring diagnostics with annotations
You can suppress warning and error diagnostics in both `spec.yaml` and Python files with `#` comments:

- Warning file-level (must be in the file header comments):  
  `# aiconls: ignore-warnings`  
  or only selected codes, e.g. `# aiconls: ignore-warnings W011,W013`
  (compatibility aliases also work: `ignore-warning` and numeric shorthand like `202` -> `W202`)
- Warning inline (same line as the reported warning):  
  `# aiconls: ignore-warning`  
  or only selected codes, e.g. `# aiconls: ignore-warning W011`
- Error file-level (must be in the file header comments):  
  `# aiconls: ignore-errors`  
  or only selected codes, e.g. `# aiconls: ignore-errors E012,E013`
  (compatibility aliases also work: `ignore-error` and numeric shorthand like `012` -> `E012`)
- Error inline (same line as the reported error):  
  `# aiconls: ignore-error`  
  or only selected codes, e.g. `# aiconls: ignore-error E012`

When a file-level ignore is declared in `spec.yaml` header comments, it also applies globally to diagnostics reported in other project files.

## Language Server usage
The LSP server expects the AICON project root via `AICONLS_PROJECT_ROOT`:

```bash
export AICONLS_PROJECT_ROOT=/path/to/aicon-project
aiconls lsp
```

The server runs over stdio (via `pygls`) and can be wired into an editor/client.
Transport options are passed through to `pygls`, e.g. `aiconls lsp --tcp --port 8888`.
From a source checkout, `python linter.py` starts the same server.

## Features
- Validates `spec.yaml` structure and consistency
- Reports spec errors for unexpected properties in component/goal/connection entries
- Parses AICON Python components and checks spec/code mismatches
- Resolves component kind through inheritance chains, including imported base classes from the current Python environment, project-local modules, and relative imports
- Emits diagnostics for missing implementations and shape/connection issues
- Provides code actions to:
  - generate missing component/goal/connection skeletons
  - fix spec mismatch issues (including "Fix all spec mismatches")
  - add inline warning-ignore annotations (`# aiconls: ignore-warning <CODE>`)
  - skip unavailable quick fixes without showing error popups (failures are logged)
- Exposes an editor command (`aiconls.runDynamicCheck`) to run dynamic lint checks project-wide, publish snapshot diagnostics for all affected files, and log a summary
- Supports completions for quantities/connections in component code
- Supports go-to-definition / go-to-implementation between `spec.yaml` and Python files

## Development
Run the test suite:

```bash
pytest -q
```

Run type checking and linting:

```bash
pyright
ruff check .
ruff format --check .
```

Install pre-commit hooks (includes automatic patch-version bump on commit):

```bash
pre-commit install
```

## Context
This project is part of my bachelor thesis at TU-Berlin at Compilers & Programming Languages with Michel Steuwer together with the Robotics group of Oliver Brock
