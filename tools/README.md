# Tools

- [`aiconls/`](aiconls): AICONLS, a linter and language server for AICON by Ole Liebich. It checks the code of a system
  against the `spec.yaml` next to it: the components with their quantities and shapes, the connections between them and
  the goals, and, with `--dynamic`, whether the gradients through each connection and estimator are non-zero.
- [`vscode-aiconls/`](vscode-aiconls): the VS Code extension for it, also by Ole Liebich.

Both are copies of their upstream versions ([aiconls](https://git.tu-berlin.de/ole_l/aiconls),
[vscode-aiconls](https://git.tu-berlin.de/ole_l/vscode-aiconls)), adapted to this repository (see
[Changes](#changes-to-upstream)). As part of this repository, they are covered by its MIT license
([LICENSE.md](../LICENSE.md)).

## Linter

The linter needs four packages in addition to AICON, the `lint` extra:

```bash
uv sync --extra pusht --extra nav2d --extra test --extra lint   # your usual extras plus lint
```

The default and `cpu` pixi environments already include it.

Each example directory has its `spec.yaml`. The scan checks one of them:

```bash
python tools/aiconls/src/aiconls.py scan examples/nav_2d             # or: pixi run lint examples/nav_2d
python tools/aiconls/src/aiconls.py scan examples/nav_2d --dynamic   # also builds the graph and checks the gradients
```

`python -m examples.spec_files` regenerates the `spec.yaml` files from the experiment specifications, and
`aicon_base.experiment_util.spec_yaml` writes one for your own system. For the format of `spec.yaml`, the diagnostics
and how to ignore single ones, see [aiconls/README.md](aiconls/README.md).

## VS Code extension

Building the extension needs Node.js (e.g. `pixi exec -s nodejs -- bash` gives a shell with it):

```bash
cd tools/vscode-aiconls
npm ci
npx @vscode/vsce package --no-dependencies --skip-license -o aiconls.vsix
```

Then, in VS Code:

1. Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and
   `aiconls.vsix` (Command Palette → **Extensions: Install from VSIX...**).
2. Open the repository folder and select its environment (`.venv` or the pixi environment) as the Python interpreter.
3. Open a file of an example. The extension uses the server in `tools/aiconls`; the example whose `spec.yaml` is
   nearest above the active file is the project, and switching to a file of another example switches the project.

**aiconls: Run dynamic parsing check** in the Command Palette runs the dynamic check, and **aiconls: Visualize
spec.yaml** draws the graph of the `spec.yaml`. See [vscode-aiconls/README.md](vscode-aiconls/README.md) for the
settings.

## Changes to upstream

Based on aiconls 0.0.71 (commit `a87b5b9`) and vscode-aiconls 0.0.9 (commit `b27044a`).

aiconls:

- Shapes: a dimension of `0` in the code counts as of variable size, like `-1` and named constants (e.g.
  `(NUM_BLOCKS, 2)`).
- Quantity definitions: annotated `cls_quantity_names_and_shapes`, definitions inherited from a base class (also from
  another package) or set when the component is built; `c = self.connections[...]` aliases in components.
- Classes that an example imports from another package of the repository (a sibling example, a shared module),
  including relative imports and re-exports, are found; base classes defined next to their subclasses are not
  reported as unused.
- Connections: quantities of switching conditions (`required_signs_dict`), `<quantity>_batched` inputs and quantities
  used by a component on the connection (its `f_func` or `connected_quantities`) count as used; connection builders
  that return their dictionary directly are read.
- Goals that no component holds (other goals an experiment can be run with) are not reported as errors.
- Dynamic check: modules are imported as part of their package; constructor parameters without default get
  placeholders; the derivative check draws up to 1024 seeded random inputs instead of one, so that functions whose
  gradient is non-zero only for some inputs pass; it runs with the dtype of the `spec.yaml` as PyTorch's default
  dtype; `differentiate_against` is respected; floating-point inputs are converted to the dtype of the component or
  connection that uses them, as the ROS middleware does (the real pushing example mixes float32 and float64).
- The language server reports the same as the scan.
- Package metadata: the license note above and links to the upstream repository and this repository's issues.

vscode-aiconls:

- Uses the server in `tools/aiconls` of the workspace if there is one, instead of installing one from Git.
- Without `aiconls.project.root`, the folder of the nearest `spec.yaml` above the active file is the project root.
- No welcome message on start.
- Package metadata: Ole Liebich as author and the MIT license.
