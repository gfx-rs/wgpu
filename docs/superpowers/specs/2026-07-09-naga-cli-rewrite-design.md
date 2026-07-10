# naga-cli rewrite — design

Date: 2026-07-09
Branch: `naga-cli-rewrite`
Status: approved (pending written-spec review)

## Goals

Rewrite `naga-cli` to serve four drivers (all first-class):

1. **IDE / tool integration** — structured JSON on stdout (diagnostics + reflection), stdin input.
2. **Full option coverage** — every frontend/backend option reachable at all times, with an always-current help surface.
3. **External tool hooks** — chain DXC (→ DXIL), `spirv-opt`, `spirv-val` from a single invocation.
4. **Robustness** — reduce panics, typed errors, tests.

## Decisions (locked during brainstorming)

- **Clean break** on the current CLI surface. No obligation to preserve existing flags (naga's own test harness will be migrated).
- **Flat CLI** (no subcommands). Output format inferred from output-file extension, as today.
- **clap** replaces `argh` (rich auto-generated `--help`, derives, `ArgGroup`, `ValueEnum`).
- Config dialect is **JSON**, identical to the `--format json` output dialect (one serde dialect for in and out).
- Tool hooks are **subprocesses discovered on `PATH`**, degrading gracefully when absent (no new link-time deps).

## Current state (baseline)

- Crate: `naga-cli/`, binary `naga`, single file `naga-cli/src/bin/naga.rs` (~1080 lines), parsing via `argh`.
- Most backend `Options` already derive `Serialize`/`Deserialize` behind naga's existing `serialize` / `deserialize` features: `msl::Options`, `hlsl::Options`, `glsl::Options`, `spv::PipelineOptions`, `msl::PipelineOptions`, `wgsl::WriterFlags`, `BoundsCheckPolicy`/`BoundsCheckPolicies`.
- Serde gaps (derive nothing today): `back::spv::Options`, `back::dot::Options`, and all three frontend Options (`front::wgsl::Options`, `front::glsl::Options`, `front::spv::Options`).
- No clap traits anywhere. No JSON output. No DXC / spirv-opt / spirv-val hooks. stdin only partial (`--stdin-file-path`). Line ~507 carries a pre-existing `TODO: read the parameters from RON?`.

## Approach

Chosen approach is a **hybrid of config-driven exposure and auto-derived flags** ("A+B"): every option is reachable either as an auto-derived clap flag (scalars) or through a JSON config (everything, including complex fields). clap coupling into naga core is acceptable to the maintainer and gated behind a feature.

### 1. Crate layout

Split responsibilities so the core is testable and panic-free:

- **`naga-cli` library** — pure core. Functions along the lines of `parse → validate → compile / reflect`, each returning `Result<T, Diagnostic>`. No file IO, no `process::exit`, no panics on user input. All input/output types are serde-serializable.
- **`naga` binary** — thin shell. clap parsing, file/stdin IO, subprocess invocation, and rendering (human text vs JSON). The binary is the *only* site that exits the process or prints.

Rationale: robustness (§6), JSON output (§4), and testing (§6) all fall out of a pure, serde-able library boundary.

### 2. Options exposure (config + auto flags)

- Derive `clap::Args` (and `clap::ValueEnum` on option enums such as `BoundsCheckPolicy`) on naga's Options structs, gated behind a naga feature (new `clap` feature, or folded into the existing serialize-adjacent feature set).
- The CLI `#[command(flatten)]`s these into its arg struct. New naga fields therefore appear as flags automatically.
- **Name collisions** (e.g. `lang_version` in both `msl` and `spv`) are resolved by renaming *only the colliding* flags: `#[arg(long = "msl-lang-version")]`, etc. Non-colliding fields stay automatic.
- **Complex fields** that do not map to scalar flags — `binding_map`, `inline_samplers`, `per_entry_point_map`, `sampler_buffer_binding_map`, `dynamic_storage_buffer_offsets_targets`, `external_texture_binding_map`, etc. — are marked `#[arg(skip)]` and are reachable only via config. Individual flag exposure for these can be added later as demand appears.

#### Config input

- `--config <path>` — read JSON from a file, deserialize directly into the Options structs.
- `--config-json '<string>'` — same, but the JSON literal is given inline as the argument value (no temp file needed).
- **Exclusivity:** `--config` and `--config-json` are mutually exclusive with each other, and both are mutually exclusive with the entire flattened option-flag group, enforced declaratively via a clap `ArgGroup`. If a config is supplied, passing any option flag is a hard parse error.
- **IO is orthogonal** and always accepted alongside config: input/output paths, `--stdin-file-path`, and `--format {text|json}`. Config describes *how* to compile; the CLI still says *what* file and *what* output shape.

Rationale for exclusivity: eliminates flag-vs-config merge ambiguity entirely (no need to detect explicitly-set flags, no partial-mirror struct, no `value_source` plumbing).

### 3. Shared common-options refactor (phased, not a gate)

Many Options structs repeat the same fields with slightly different docs/names:
`bounds_check_policies`, `zero_initialize_workgroup_memory`, `force_loop_bounding`, `task_dispatch_limits`, `mesh_shader_primitive_indices_clamp`, `ray_query_initialization_tracking`, `emit_int_div_checks`, `fake_missing_bindings`, `lang_version`.

Extract these into a shared `CommonOptions` struct that backends embed, so clap derives and docs are declared once. This also shrinks the surface that must be annotated for flags.

**Scope caveat:** this is a naga *core* refactor with a wide blast radius (every backend `Options`, plus callers in `wgpu`, `wgpu-hal`, and tests). It is a **separate, optional phase** and must **not gate** the CLI rewrite. If deferred, the CLI still works with per-backend duplicated fields.

### 4. Structured IO

- `--format {text|json}`, default `text`.
- `json` mode emits, on stdout, a machine-consumable document containing:
  - **Diagnostics** — errors and warnings with source spans (byte offsets / line-col), severity, and message. Replaces today's stderr-only human errors.
  - **Reflection info** — entry points, bindings, workgroup sizes, and related module metadata (exact schema defined during planning).
- Human `text` mode remains the default for interactive use.
- **stdin:** generalize the existing `--stdin-file-path` mechanism so input can always come from stdin with an explicit virtual path (used for extension-based format detection).

### 5. External tool hooks (subprocess seam)

- The library defines a **hook seam**; the binary runs the processes.
- `--dxc` — after SPIR-V/HLSL generation, invoke DXC to produce DXIL, so a single command yields DXIL.
- `--spirv-opt` — pipe SPIR-V output through `spirv-opt`.
- `--spirv-val` — validate SPIR-V output through `spirv-val`.
- Binaries are discovered on `PATH`. When a requested tool is absent, fail with a clear, actionable error; when not requested, no dependency is imposed.

### 6. Robustness & testing

- Library returns typed errors (`Diagnostic`); the binary is the sole panic/exit/print site. Audit current `unwrap`/`expect`/`panic!` on user-controlled paths and convert to errors.
- Tests:
  - **Library unit tests** — parse / validate / compile / reflect happy + error paths; config round-trip; config/flag exclusivity rejection.
  - **Help-output snapshot test** — guards that `--help` stays current (fulfils "up-to-date help menu").
  - **Golden-file tests** — representative shaders compiled to each backend, output compared.

### 7. "All options exposed, always current" — help & discoverability

- clap auto-generates `--help` from the derives; because flags derive from naga Options, help **auto-syncs** when fields are added. This covers scalar flags.
- For config-only (`#[arg(skip)]`) fields, derive `schemars::JsonSchema` on the Options (same gated pattern as serde/clap) and add **`--print-config-schema`**, which dumps the JSON Schema for the config document. Every option is then either a visible flag or a documented schema entry — honestly "all options exposed all the time."
- The help-snapshot test (§6) guards against drift.

### 8. Examples & docs

- `examples/` directory with runnable invocations: flags form, config-file form, `--config-json` inline form, stdin input, `--format json` output, and a DXC chain.
- Doc comments on every flag (these feed clap `--help`).
- README rewrite covering the config schema, JSON output shape, and tool-hook prerequisites.

## Requirement → design traceability

| Original requirement | Covered by |
|---|---|
| Testing | §1 (pure lib), §6 (unit + snapshot + golden + config tests) |
| All options exposed all the time | §2 (auto flags), §7 (`--print-config-schema`, snapshot guard) |
| Up-to-date help menu | §7 (clap auto-help + schema), §8 (doc comments) |
| Reduce panics | §1, §6 (typed errors; bin is sole exit site) |
| Structured stdout (JSON) incl. error messages | §4 (`--format json`, diagnostics with spans) |
| Possibly reflection info | §4 (reflection block in JSON) |
| Parse from stdin | §4 (generalized `--stdin-file-path`) |
| Examples & better docs | §8 |
| Hook into DXC (single-command DXIL) | §5 (`--dxc`) |
| Hook into spirv-opt / spirv-val | §5 (`--spirv-opt`, `--spirv-val`) |

## Non-goals / deferred

- Subcommand structure (flat CLI chosen).
- Individual flags for complex map/vec fields (config-only for now; add later as needed).
- Linking DXC / SPIRV-Tools as libraries (subprocess only).
- The `CommonOptions` extraction may ship as a later phase without blocking the CLI rewrite.

## Open items for the planning phase

- Exact JSON schema for diagnostics and reflection output.
- Whether the naga clap-derive feature is new or folded into existing features.
- Migration plan for naga's existing test harness that shells out to the CLI.
