# AICONLS LSP: Validations, Capabilities, and Features

This document summarizes the Language Server Protocol (LSP) functionality implemented in AICONLS.

## 1) LSP capabilities

### Registered LSP requests/notifications

| LSP method | Purpose |
| --- | --- |
| `initialized` | Loads/parses `spec.yaml` on server initialization. |
| `textDocument/didOpen` | Parses opened file; if `spec.yaml` is opened, reparses the spec, reparses all project Python files, and revalidates. |
| `textDocument/didChange` | Re-parses the changed document and refreshes diagnostics. |
| `textDocument/implementation` | From `spec.yaml`, jumps to Python implementation of component/goal/connection classes or instances. |
| `textDocument/definition` | From Python class references, jumps to matching definitions in `spec.yaml`. |
| `textDocument/completion` | Context-aware completions for quantities and connections in component code. |
| `textDocument/rename` | Cross-project rename for quantity names across Python files and `spec.yaml`. |
| `textDocument/codeAction` | Provides quick fixes and source-level fix-all actions tied to diagnostics. |

### Completion support

Completions are provided with trigger characters `"` and `.`:

- `self.quantities["..."]` -> suggests local quantity names.
- `self.connections["..."]` -> suggests valid connection names for the current component.
- `self.connections["conn"].` -> suggests `connected_quantities[]` snippet.
- `self.connections["conn"].connected_quantities["..."]` -> suggests quantities available through connected components.
- `self.connected_quantities["..."]` -> in connection classes, suggests available connection quantities.

### Rename support

`textDocument/rename` supports quantity renaming across:

- open Python documents,
- Python files in project (excluding ignored directories),
- `spec.yaml`.

Safety behavior:

- If a rename touches identifier usages (e.g., function parameters), the new name must be a valid non-keyword Python identifier.
- String literal usages preserve quote style.

## 2) Code actions and commands

### Quick fixes / source actions

| Action | Trigger |
| --- | --- |
| Auto-generate component skeleton | `W011` (`COMPONENT_NOT_IMPLEMENTED`) |
| Auto-generate goal skeleton | `W012` (`GOAL_NOT_IMPLEMENTED`) |
| Auto-generate connection skeleton | `W013` (`CONNECTION_NOT_IMPLEMENTED`) |
| Update component quantities to match spec | `E101`, `E102` |
| Update connection quantities to match spec | `E201` |
| Fix all spec mismatches (`source.fixAll`) | Shown when mismatch diagnostics exist (`E102` and/or `E201`) |
| Ignore this warning (inline annotation) | Any warning diagnostic |

If a quick fix cannot be generated, the action is skipped and the failure is logged via `window/logMessage`; no user-facing error popup is shown.

### Registered server commands

| Command ID | Purpose |
| --- | --- |
| `aiconls.generateFiles` | Generate project skeleton files from `spec.yaml`. |
| `aiconls.generateExperimentSpecification` | Generate experiment specification artifacts from `spec.yaml`. |
| `aiconls.revalidateGeneratedComponent` | Re-parse/validate after generated edits. |
| `aiconls.fixAllSpecMismatches` | Apply workspace-wide or file-scoped mismatch fixes. |
| `aiconls.runDynamicCheck` | Run dynamic lint checks across project files, publish snapshot diagnostics (`source: aiconls.dynamic`), and log a results summary. |

> Generation/fix flows are blocked when spec parsing/validation contains errors; the server emits "Spec malformed" messages.

## 3) Validations and diagnostics

Diagnostics are emitted with severity and code. Current set:

### Spec parse/structure diagnostics

| Code | Severity | Meaning |
| --- | --- | --- |
| `E000` | Error | Malformed spec. |
| `E001` | Error | Malformed component type. |
| `E002` | Error | Malformed data type. |
| `E003` | Error | Malformed shape. |
| `E004` | Error | Duplicated component in spec. |
| `E005` | Error | Duplicated goal in spec. |
| `E006` | Error | Duplicated connection in spec. |
| `E007` | Error | Unexpected component property in spec. |
| `E008` | Error | Unexpected goal property in spec. |
| `E009` | Error | Unexpected connection property in spec. |

### Spec-to-project consistency diagnostics

| Code | Severity | Meaning |
| --- | --- | --- |
| `W011` | Warning | Component declared in spec but not implemented. |
| `W012` | Warning | Goal declared in spec but not implemented. |
| `W013` | Warning | Connection declared in spec but not implemented. |
| `E012` | Error | Undefined component referenced in a spec connection. |
| `E013` | Error | Component references goal not defined in spec. |
| `E014` | Error | Duplicate component listed inside one connection. |

### Component-level diagnostics

| Code | Severity | Meaning |
| --- | --- | --- |
| `E101` | Warning | Missing local quantity definitions relative to spec. |
| `E102` | Error | Quantity shape/content mismatch against spec. |
| `W101` | Warning | Parsed component not found in spec. |
| `W102` | Warning | Parsed goal not found in spec. |
| `E103` | Error | `goal_func` parameter not available from connected quantities. |
| `E104` | Error | Component references connection not allowed by spec. |
| `E105` | Error | Accessed connected quantity not available via that connection. |
| `E106` | Error | Estimator `f_func` parameter not available from local or directly connected quantities (`dt` allowed). |
| `E107` | Error | Component accesses local `self.quantities[...]` key not defined in spec quantities. |
| `E108` | Error | Estimator `f_func` is not differentiable with `torch.func.jacfwd`. |
| `E109` | Error | Estimator output-signature name is not defined as a local quantity in spec. |

### Connection-level diagnostics

| Code | Severity | Meaning |
| --- | --- | --- |
| `W201` | Warning | Parsed connection not found in spec. |
| `E201` | Error | Connection has invalid quantity set vs. expected spec-derived quantities. |
| `E202` | Error | `connection_func` parameter not defined in super-constructor quantities. |
| `W202` | Warning | Quantity provided in connection super-constructor is unused in `connection_func`. |

## 4) Diagnostic suppression features

Supported in both `spec.yaml` and Python files:

- File-level warning suppression in file header comments:
  - `# aiconls: ignore-warnings`
  - `# aiconls: ignore-warnings W011,W013`
  - Compatibility aliases: `ignore-warning` and numeric shorthand (`202` -> `W202`)
- Inline warning suppression:
  - `# aiconls: ignore-warning`
  - `# aiconls: ignore-warning W011`
- File-level error suppression in file header comments:
  - `# aiconls: ignore-errors`
  - `# aiconls: ignore-errors E012,E013`
  - Compatibility aliases: `ignore-error` and numeric shorthand (`012` -> `E012`)
- Inline error suppression:
  - `# aiconls: ignore-error`
  - `# aiconls: ignore-error E012`

File-level suppressions declared in `spec.yaml` header comments are applied globally to diagnostics from all project files.

## 5) Additional behavior implemented by the server

- Requires `AICONLS_PROJECT_ROOT` to point to the AICON project root.
- Publishes diagnostics per document/version through LSP `publishDiagnostics`.
- Tracks open files and marks Python documents stale after spec changes, then refreshes them on demand.
- Parses all project Python files on spec open/update while skipping directories like `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`.
- Clears previously published dynamic-check diagnostics when a document changes to avoid stale dynamic results after edits.
