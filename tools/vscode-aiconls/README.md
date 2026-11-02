# AICON Linter VS Code Extension

VS Code extension for linting and tooling around AICON projects.

## Fastest way for testers to install

1. Install the `.vsix` file in VS Code:
   - Command Palette → **Extensions: Install from VSIX...**
2. Set up AICON as described in its [README](https://github.com/tu-rbo/aicon#without-ros-simulated-examples), with the
   `lint` extra, for example:
   - uv: `uv sync --extra pusht --extra nav2d --extra test --extra lint` (creates `.venv`)
   - pixi: `pixi install` (creates `.pixi/envs/default`, which includes the `lint` extra)
3. Open the AICON repository as the workspace.
4. Ensure the Python extension (`ms-python.python`) is installed and select the AICON environment as interpreter (Python >= 3.10):
   - uv: `.venv/bin/python`
   - pixi: `.pixi/envs/default/bin/python`
5. Open a Python or YAML file. A workspace that contains the server itself in `tools/aiconls` (as the AICON repository
   does) uses that one, without installing anything; its dependencies come from the `lint` extra.
6. In other workspaces, the extension prompts on first start to install the language server from Git. Click
   **Install from Git**.

From Git, the language server is installed into the selected environment with `uv pip install`, because uv and pixi environments do not contain pip. Make sure [uv](https://docs.astral.sh/uv/) is installed (it is also needed for the uv setup above); pip is only used as a fallback if uv is not found.

`uv sync` and `pixi install` remove packages that AICON does not declare, including the dependencies of a server installed from Git. The extension verifies the server dependencies (`pygls`, `PyYAML`, `lsprotocol`, `libcst`) before every start and reinstalls them when needed, so this also covers recreating or switching the environment.

The extension clones and updates the server from:
- `https://git.tu-berlin.de/ole_l/aiconls`
  (default; configurable via `aiconls.server.gitRepository`)

## Commands

- **AICONLS: Install/Update AICON Language Server**  
  Updates the checked-out server and reinstalls it into the selected Python environment.

- **AICONLS: Restart Language Server**  
  Restarts the active language server process.

- **AICONLS: Generate AICON files from spec**  
  Runs server-side code generation from the current specification.

- **AICONLS: Generate AICON specification from spec**  
  Generates an experiment specification from the current `spec.yaml`.

- **AICONLS: Fix all spec mismatches**  
  Applies the server's bulk mismatch-fix flow for the specification/project.

- **AICONLS: Run dynamic parsing check**  
  Runs language server dynamic checks across project files and publishes dynamic diagnostics in the editor.

- **AICONLS: Set AICON project root to current folder**  
  Sets `aiconls.project.root` (workspace setting) to the folder of the open file and restarts the language server. Also available in the Explorer context menu, where it uses the clicked folder (or the folder of the clicked file). Folders inside the workspace are stored relative to `${workspace}`.

- **AICONLS: Visualize spec.yaml**  
  Opens an interactive webview graph based on `spec.yaml` (`components`, `connections`, `goals`) and lets you inspect and drag nodes, pan/zoom with mouse or trackpad (including deeper zoom-out), and recalculate ELK-based 2D auto-layout positions, with live refresh on save.
  When a `spec.yaml` editor is active, a **Visualize spec.yaml** text button appears in the bottom-right status bar for one-click access.
  The top of `spec.yaml` also shows a **Visualize spec.yaml** CodeLens action for in-editor access.
  The webview includes advanced layout controls for dense undirected graphs: algorithm switching (`stress`, `force`, `layered`, `radial`, `mrtree`) and tuning for routing, spacing, iterations, force repulsion, and radial compaction.
  You can edit selected node properties as JSON and persist them to `spec.yaml`, and create/delete components and connections directly from the webview.

## Settings

- `aiconls.project.root`  
  Set the AICON project root (supports `${workspace}`). If empty (the default), the folder of the nearest `spec.yaml`
  above the active file is the project, and the server restarts when a file of another project becomes active. This
  way each example of a repository with several examples is checked against its own `spec.yaml`.

- `aiconsls.trace.server`  
  Controls LSP protocol trace logging.

- `aiconls.server.path`  
  Optional absolute path to a custom `linter.py`. If set, auto Git install is skipped. If unset, the server in
  `tools/aiconls` of the workspace is used if there is one.

- `aiconls.server.debug.enabled`  
  Enables debugpy for the language server process to allow attaching a Python debugger.

- `aiconls.server.debug.port`  
  TCP port used by debugpy (default: `5678`).

- `aiconls.server.debug.waitForClient`  
  When enabled, the server waits for a debugger to attach before continuing startup.

- `aiconls.server.gitRepository`  
  Git repository URL used by the automatic install/update flow.

## Debugging the language server

To debug a custom server path such as
`/Users/oleliebich/Uni/Bachelor/linter/aiconls/linter.py`:

1. In VS Code settings, set:
   - `"aiconls.server.path": "/Users/oleliebich/Uni/Bachelor/linter/aiconls/linter.py"`
   - `"aiconls.server.debug.enabled": true`
2. Start **Run Extension** from the extension workspace.
3. In the extension host window, open a Python or YAML file to trigger server startup.
4. Start **Attach AICONLS Server (debugpy)**.
5. Set breakpoints in the Python server code and continue.

## Packaging for sharing

Build and package:

```bash
npm run package
npx @vscode/vsce package
```

Distribute the generated `.vsix` file to testers.

## GitLab release automation

This repository now includes a GitLab pipeline in `.gitlab-ci.yml`.

When you push a Git tag:

1. `build_vsix` builds the extension and creates the `.vsix`.
2. `release_gitlab` uploads the `.vsix` to GitLab Package Registry.
3. `release_gitlab` creates (or updates) a GitLab Release for the tag and attaches the `.vsix` as a release asset link.

If your GitLab instance does not allow release creation with `CI_JOB_TOKEN`, add a protected CI/CD variable `GITLAB_TOKEN` (API scope) and the pipeline will use it automatically.

Create a release by tagging and pushing:

```bash
git tag v0.0.2
git push origin v0.0.2
```
