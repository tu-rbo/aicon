# AICON Extension + Linter: User Capabilities

This list is compiled from the extension code in [src/](/Users/oleliebich/Uni/Bachelor/linter/vscode-extension/aiconsls/src) and the linter/language-server code in [/Users/oleliebich/Uni/Bachelor/linter/aiconls](/Users/oleliebich/Uni/Bachelor/linter/aiconls).

## VS Code extension capabilities

- Starts and manages the AICON language server for Python and YAML files.
- Auto-starts the server when a document is opened.
- Restarts the server when the active Python environment changes.
- Lets users manually restart the server (`aiconls.restart`).
- Lets users set the AICON project root to the open file's folder (`aiconls.setProjectRootToCurrentFolder`).
- Can install/update the language server from Git (`aiconls.installServer`).
- Lets users trigger code generation from the extension (`aiconls.runGenerateFiles`).
- Lets users trigger experiment spec generation from the extension (`aiconls.runGenerateExperimentSpecification`).
- Lets users run project-wide spec mismatch fixes from the extension (`aiconls.runFixAllSpecMismatches`).
- Lets users run dynamic parsing checks from the extension (`aiconls.runDynamicParsing`).
- Supports using a custom `linter.py` path instead of auto-installed server.
- Installs the server into uv/pixi-managed AICON environments via `uv pip install` (pip only as fallback).
- Verifies required Python server dependencies and reinstalls if missing (e.g. after `uv sync`/`pixi install`).
- Enforces minimum Python version (3.10+) and prompts environment change if needed.
- Exposes debug attach settings (debugpy host/port/wait) for server debugging.
- Provides LSP trace logging and server logging output channels.
- Supports configurable AICON project root and server Git repository URL.

### `spec.yaml` visualizer features

- Opens an interactive graph view for `spec.yaml` (`aiconls.visualizeSpec`).
- Shows a bottom-right status-bar text button for one-click visualizer access when `spec.yaml` is active.
- Shows a top-of-file CodeLens action in `spec.yaml` for in-editor visualizer access.
- Resolves `spec.yaml` from active editor, configured project root, or workspace root.
- Visualizes `components`, `connections`, and `goals` as nodes/edges.
- Shows spec-derived diagnostics in the panel.
- Supports pan/zoom (including deeper zoom-out) and node dragging in the graph.
- Supports ELK auto-layout with algorithm/routing/spacing tuning.
- Falls back to deterministic layout if ELK layout fails/incomplete.
- Live refreshes when `spec.yaml` is saved.
- Allows editing selected node properties as JSON and writing back to `spec.yaml`.
- Allows creating components and connections from the UI.
- Allows deleting components and connections from the UI (goal deletion intentionally not supported there).

## Linter / language-server capabilities

- Runs as an LSP server over stdio (`linter.py` + `pygls`).
- Requires and uses `AICONLS_PROJECT_ROOT` to locate project `spec.yaml`.
- Parses/validates on open/change and publishes diagnostics.
- Parses all project Python files when spec is opened/updated.

### CLI capabilities (`aiconls`)

- `aiconls scan <project>` scans project and emits diagnostics.
- Supports output formats: `stylish` and `json`.
- Supports pretty JSON output.
- Supports writing output to file.
- Supports configurable exit behavior (`--fail-on none|error|warning`).
- Supports ANSI color control for stylish output.
- Produces report summaries (counts by errors/warnings/files).

### Validation and diagnostics capabilities

- Validates malformed spec structure/content (parser and schema-like checks).
- Detects unexpected properties in component/goal/connection entries.
- Detects missing implementations declared in spec (components/goals/connections).
- Detects undefined goals in components.
- Detects undefined/duplicate components in connection component lists.
- Detects components/goals/connections implemented in code but not present in spec.
- Validates component quantities against spec (missing/unexpected/shape mismatch).
- Validates connection quantity mappings and shape compatibility.
- Validates invalid/unused connection function parameters.
- Validates goal function parameters against allowed quantities.
- Validates estimator `f_func` parameters against allowed local/connected quantities.
- Supports warning/error suppression annotations (file-level and inline, with optional code filters).

### Editor intelligence capabilities

- Completion support for:
  - component quantities,
  - allowed connection names for a component,
  - `connected_quantities["..."]` scaffolding,
  - connected quantity names available through a selected connection.
- Go-to-implementation from `spec.yaml` symbols to Python implementations.
- Go-to-definition from Python class references back to `spec.yaml` entries.
- Rename support for quantities across Python files and `spec.yaml` with validation.

### Code action / autofix capabilities

- Auto-generate missing component skeletons.
- Auto-generate missing goal skeletons.
- Auto-generate missing connection skeletons.
- Update component quantities in Python code to match spec.
- Update connection quantities in Python code to match spec.
- Generate/update `build_connections()` implementation to match spec.
- Update spec quantities from component implementation.
- Update spec quantities from component and propagate changes to affected project code.
- “Fix all spec mismatches” across one file or broader project scope.
- Quick-fix action to append inline warning ignore annotation.

### Server commands exposed to clients

- `aiconls.generateFiles`
- `aiconls.generateExperimentSpecification`
- `aiconls.revalidateGeneratedComponent`
- `aiconls.fixAllSpecMismatches`
- `aiconls.runDynamicCheck`
