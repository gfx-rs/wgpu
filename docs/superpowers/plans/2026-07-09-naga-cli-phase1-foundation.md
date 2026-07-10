# naga-cli Rewrite — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `naga-cli` from `argh` to `clap`, split the single 1080-line binary into focused modules, remove all `process::exit`/panic paths from translation logic, and add integration tests — all behavior-preserving, with no changes to the `naga` core crate.

**Architecture:** Single `naga` binary (no separate library crate). Logic is split into modules of the binary crate: `cli` (clap arg definitions + value parsers), `params` (build a `Parameters` from parsed args), `core` (pure `parse → validate → write_output` returning `anyhow::Result`, no exits/prints of its own for control flow), and `main` (the sole process-exit and error-render site). Tests are integration tests that exec the compiled binary via the cargo-provided `CARGO_BIN_EXE_naga` path.

**Tech Stack:** Rust, `clap` v4 (derive), `anyhow`, `naga` (unchanged), cargo integration tests (no new test-only crates).

## Global Constraints

- MSRV for `naga-cli`: `rust-version = "1.76"` (do not use APIs newer than this). Copied verbatim from `naga-cli/Cargo.toml:15`.
- `naga-cli/Cargo.toml` `[[bin]]` MUST keep `doc = false` (conflicts with `naga`'s docs — see gfx-rs/wgpu#4997).
- No changes to any file under `naga/` in this phase. naga-core derive work is Phase 2.
- Behavior-preserving: every flag keeps its current long name and semantics. The only intentional behavior change is exit codes for internal-error paths (previously `-1`/`process::exit`) now uniformly exit `1` via `main`.
- Use `clap = { workspace = true }`; add the workspace entry `clap = { version = "4", features = ["derive"] }`.

---

## File Structure

- `Cargo.toml` (workspace root) — add `clap` to `[workspace.dependencies]`.
- `naga-cli/Cargo.toml` — swap `argh`→`clap`; point `[[bin]]` at `src/main.rs`.
- `naga-cli/src/main.rs` — NEW entry point (thin): init logging, parse args, dispatch, render errors, exit. Replaces `src/bin/naga.rs`.
- `naga-cli/src/cli.rs` — NEW: clap `Args` struct + value-parser fns + arg enums. All argument definitions live here.
- `naga-cli/src/params.rs` — NEW: `Parameters` struct + `build_parameters(&Args) -> anyhow::Result<Parameters>`.
- `naga-cli/src/core.rs` — NEW: `parse_input`, `validate_module`, `write_output`, `bulk_validate` as `anyhow::Result`-returning functions. No `process::exit`, no `unwrap_pretty`.
- `naga-cli/src/error.rs` — NEW: `CliError` type + `print_err` render helper (moved out of the entry point).
- `naga-cli/tests/cli.rs` — NEW: integration tests execing the built binary.
- `naga-cli/tests/snapshots/help.txt` — NEW: golden `--help` output.
- `naga-cli/src/bin/naga.rs` — DELETED at the end of Task 6.

---

### Task 1: Cargo scaffolding + module skeleton

Establish the crate's new shape with a compiling stub so later tasks fill modules in.

**Files:**
- Modify: `Cargo.toml` (workspace root, `[workspace.dependencies]`)
- Modify: `naga-cli/Cargo.toml`
- Create: `naga-cli/src/main.rs`
- Create: `naga-cli/src/cli.rs`, `naga-cli/src/params.rs`, `naga-cli/src/core.rs`, `naga-cli/src/error.rs` (stubs)
- Keep (untouched for now): `naga-cli/src/bin/naga.rs`

**Interfaces:**
- Produces: a buildable `naga` binary from `src/main.rs`; empty modules `cli`, `params`, `core`, `error`.

- [ ] **Step 1: Add clap to workspace dependencies**

In `Cargo.toml` (workspace root), insert after line 116 (`argh = "0.1.13"`), keeping alphabetical order (after `bit-vec`/before is fine; place near other deps):

```toml
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 2: Rewrite `naga-cli/Cargo.toml`**

Replace the `[[bin]]` block and the `argh` line. New `[[bin]]`:

```toml
[[bin]]
name = "naga"
path = "src/main.rs"
# This _must_ be false, as this conflicts with `naga`'s docs.
#
# See https://github.com/gfx-rs/wgpu/issues/4997
doc = false
```

In `[dependencies]`, remove the line `argh.workspace = true` and add:

```toml
clap = { workspace = true }
```

(Leave `test` unset on `[[bin]]` — integration tests in `tests/` build the binary regardless.)

- [ ] **Step 3: Create stub modules**

`naga-cli/src/cli.rs`:
```rust
//! Command-line argument definitions.
```
`naga-cli/src/params.rs`:
```rust
//! Translation parameters assembled from parsed CLI arguments.
```
`naga-cli/src/core.rs`:
```rust
//! Pure translation core: parse, validate, and emit output.
```
`naga-cli/src/error.rs`:
```rust
//! CLI error type and human-readable rendering.
```

- [ ] **Step 4: Create stub `naga-cli/src/main.rs`**

```rust
mod cli;
mod core;
mod error;
mod params;

fn main() {
    eprintln!("naga-cli scaffolding");
    std::process::exit(2);
}
```

- [ ] **Step 5: Verify it builds**

Run: `cargo build -p naga-cli`
Expected: builds successfully (warnings about unused modules are acceptable at this stage).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml naga-cli/Cargo.toml naga-cli/src/main.rs naga-cli/src/cli.rs naga-cli/src/params.rs naga-cli/src/core.rs naga-cli/src/error.rs
git commit -m "refactor(naga-cli): scaffold clap-based module layout"
```

---

### Task 2: `cli.rs` — clap argument definitions

Port every argument from the current `argh` `Args` struct (naga.rs:6-163) and the associated value newtypes to clap. Flag long-names are unchanged.

**Files:**
- Modify: `naga-cli/src/cli.rs`
- Test: `naga-cli/src/cli.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Produces:
  - `pub struct Args { .. }` deriving `clap::Parser`, with fields mirroring naga.rs:6-163 (same long names).
  - `pub enum BoundsCheckPolicyArg` (clap `ValueEnum`): `Restrict | ReadZeroSkipWrite | Unchecked`, with `pub fn to_policy(self) -> naga::proc::BoundsCheckPolicy`.
  - `pub enum ShaderStageArg` (clap `ValueEnum`): `Vert | Frag | Comp`, with `pub fn to_stage(self) -> naga::ShaderStage`.
  - `pub enum InputKind` (clap `ValueEnum`): `Bin | Glsl | Spv | Wgsl`.
  - value-parser fns (each `fn(&str) -> Result<T, String>`): `parse_shader_model`, `parse_spirv_version`, `parse_metal_version`, `parse_glsl_profile`, `parse_overrides`, `parse_defines`, `parse_capabilities`, `parse_task_limits`.
  - `pub struct Overrides { pub pairs: Vec<(String, f64)> }`, `pub struct Defines { pub pairs: Vec<(String, String)> }`.

- [ ] **Step 1: Write the failing test**

Append to `naga-cli/src/cli.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_core_flags() {
        let args = Args::try_parse_from([
            "naga",
            "--validate",
            "1",
            "--entry-point",
            "main",
            "--index-bounds-check-policy",
            "restrict",
            "in.wgsl",
            "out.spv",
        ])
        .unwrap();
        assert_eq!(args.validate, Some(1));
        assert_eq!(args.entry_point.as_deref(), Some("main"));
        assert_eq!(args.index_bounds_check_policy, Some(BoundsCheckPolicyArg::Restrict));
        assert_eq!(args.files, vec!["in.wgsl".to_string(), "out.spv".to_string()]);
    }

    #[test]
    fn parses_repeated_overrides() {
        let args =
            Args::try_parse_from(["naga", "--override", "a=1,b=2", "--override", "c=3", "x.wgsl"])
                .unwrap();
        let flat: Vec<_> = args.overrides.iter().flat_map(|o| o.pairs.clone()).collect();
        assert_eq!(
            flat,
            vec![
                ("a".to_string(), 1.0),
                ("b".to_string(), 2.0),
                ("c".to_string(), 3.0)
            ]
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p naga-cli --bin naga cli::tests`
Expected: FAIL to compile — `Args`, `BoundsCheckPolicyArg` not found.

- [ ] **Step 3: Implement `cli.rs`**

Replace the contents of `naga-cli/src/cli.rs` with:

```rust
//! Command-line argument definitions.

use clap::{Parser, ValueEnum};

/// Translate shaders to different formats.
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Bitmask of the ValidationFlags to be used; use 0 to disable validation.
    #[arg(long)]
    pub validate: Option<u8>,

    /// Policy for index bounds checking of arrays, vectors, and matrices.
    #[arg(long)]
    pub index_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Bounds-check policy for arrays/vectors/matrices in `storage`/`uniform` globals.
    /// Defaults to the index bounds check policy.
    #[arg(long)]
    pub buffer_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Bounds-check policy for texture loads. Defaults to the index bounds check policy.
    #[arg(long)]
    pub image_load_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Directory to dump the SPIR-V block context dump to.
    #[arg(long)]
    pub block_ctx_dir: Option<String>,

    /// The shader entrypoint. With `--compact`, anything unreachable from it is dropped.
    #[arg(long)]
    pub entry_point: Option<String>,

    /// GLSL profile to target, e.g. `es`, `core`, `es330`.
    #[arg(long, value_parser = parse_glsl_profile)]
    pub profile: Option<GlslProfile>,

    /// HLSL shader model, e.g. `50`, `51`, `60`..`67`.
    #[arg(long, value_parser = parse_shader_model)]
    pub shader_model: Option<naga::back::hlsl::ShaderModel>,

    /// SPIR-V version, e.g. `1.0`, `1.4`.
    #[arg(long, value_parser = parse_spirv_version)]
    pub spirv_version: Option<(u8, u8)>,

    /// Shader stage; derived from the file extension if unspecified.
    #[arg(long)]
    pub shader_stage: Option<ShaderStageArg>,

    /// Kind of input: `glsl`, `wgsl`, `spv`, or `bin`.
    #[arg(long)]
    pub input_kind: Option<InputKind>,

    /// Metal language version, e.g. `1.0`, `1.1`, `1.2`.
    #[arg(long, value_parser = parse_metal_version)]
    pub metal_version: Option<(u8, u8)>,

    /// Disable coordinate-space conversions where the frontend/backend supports them.
    #[arg(long)]
    pub keep_coordinate_space: bool,

    /// In dot output, include only the control flow graph.
    #[arg(long)]
    pub dot_cfg_only: bool,

    /// Treat STDIN as if it were this file path (needed for extension-based detection).
    #[arg(long)]
    pub stdin_file_path: Option<String>,

    /// Generate debug symbols (spv-out only, for now).
    #[arg(short = 'g', long)]
    pub generate_debug_symbols: bool,

    /// Compact the module's IR and revalidate.
    #[arg(long)]
    pub compact: bool,

    /// Write the module's IR before compaction to the given file. Implies `--compact`.
    #[arg(long)]
    pub before_compaction: Option<String>,

    /// Bulk validation mode: all filenames are inputs to read and validate.
    #[arg(long)]
    pub bulk_validate: bool,

    /// Pipeline-constant override, of the form "foo=N,bar=M"; repeatable.
    #[arg(long = "override", value_parser = parse_overrides)]
    pub overrides: Vec<Overrides>,

    /// Preprocessor defines for the GLSL frontend, "KEY=VALUE"; repeatable.
    #[arg(short = 'D', long = "defines", value_parser = parse_defines)]
    pub defines: Vec<Defines>,

    /// Capabilities filter: comma-separated names, a numeric bitflags value, "none", or "all".
    #[arg(long, default_value = "all", value_parser = parse_capabilities)]
    pub capabilities: naga::valid::Capabilities,

    /// Mesh shader task dispatch limits, as "X,Y".
    #[arg(long, value_parser = parse_task_limits)]
    pub task_limits: Option<naga::back::TaskDispatchLimits>,

    /// Whether the mesh shader output should be validated.
    #[arg(long, default_value_t = true)]
    pub validate_mesh_output: bool,

    /// Input file (stdin if omitted), then output files. In bulk mode, all are inputs.
    pub files: Vec<String>,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsCheckPolicyArg {
    Restrict,
    ReadZeroSkipWrite,
    Unchecked,
}

impl BoundsCheckPolicyArg {
    pub fn to_policy(self) -> naga::proc::BoundsCheckPolicy {
        use naga::proc::BoundsCheckPolicy as P;
        match self {
            BoundsCheckPolicyArg::Restrict => P::Restrict,
            BoundsCheckPolicyArg::ReadZeroSkipWrite => P::ReadZeroSkipWrite,
            BoundsCheckPolicyArg::Unchecked => P::Unchecked,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStageArg {
    Vert,
    Frag,
    Comp,
}

impl ShaderStageArg {
    pub fn to_stage(self) -> naga::ShaderStage {
        match self {
            ShaderStageArg::Vert => naga::ShaderStage::Vertex,
            ShaderStageArg::Frag => naga::ShaderStage::Fragment,
            ShaderStageArg::Comp => naga::ShaderStage::Compute,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Bin,
    Glsl,
    Spv,
    Wgsl,
}

/// Newtype wrapper so clap can collect repeated `--override` values.
#[derive(Clone, Debug)]
pub struct Overrides {
    pub pairs: Vec<(String, f64)>,
}

/// Newtype wrapper so clap can collect repeated `-D`/`--defines` values.
#[derive(Clone, Debug)]
pub struct Defines {
    pub pairs: Vec<(String, String)>,
}

/// Re-exported so `params.rs` can name the parsed GLSL version.
pub use naga::back::glsl::Version as GlslProfile;

fn parse_glsl_profile(s: &str) -> Result<GlslProfile, String> {
    use naga::back::glsl::Version;
    if let Some(rest) = s.strip_prefix("core") {
        Ok(Version::Desktop(rest.parse().unwrap_or(330)))
    } else if let Some(rest) = s.strip_prefix("es") {
        Ok(Version::new_gles(rest.parse().unwrap_or(310)))
    } else {
        Err(format!("Unknown profile: {s}"))
    }
}

fn parse_shader_model(s: &str) -> Result<naga::back::hlsl::ShaderModel, String> {
    use naga::back::hlsl::ShaderModel as M;
    Ok(match s.to_lowercase().as_str() {
        "50" => M::V5_0,
        "51" => M::V5_1,
        "60" => M::V6_0,
        "61" => M::V6_1,
        "62" => M::V6_2,
        "63" => M::V6_3,
        "64" => M::V6_4,
        "65" => M::V6_5,
        "66" => M::V6_6,
        "67" => M::V6_7,
        _ => return Err(format!("Invalid value for --shader-model: {s}")),
    })
}

fn parse_spirv_version(s: &str) -> Result<(u8, u8), String> {
    let dot = s.find('.').ok_or_else(|| "Missing dot separator".to_owned())?;
    let major = s[..dot].parse::<u8>().map_err(|e| e.to_string())?;
    let minor = s[dot + 1..].parse::<u8>().map_err(|e| e.to_string())?;
    Ok((major, minor))
}

fn parse_metal_version(s: &str) -> Result<(u8, u8), String> {
    let mut iter = s.split('.');
    let mut next = |iter: &mut core::str::Split<char>| {
        iter.next()
            .ok_or_else(|| format!("Invalid value for --metal-version: {s}"))?
            .parse::<u8>()
            .map_err(|err| format!("Invalid value for --metal-version: '{s}': {err}"))
    };
    let major = next(&mut iter)?;
    let minor = next(&mut iter)?;
    Ok((major, minor))
}

fn parse_overrides(s: &str) -> Result<Overrides, String> {
    let mut pairs = vec![];
    for pair in s.split(',') {
        let Some((name, value)) = pair.split_once('=') else {
            return Err(format!("value needs a `=`: {pair:?}"));
        };
        let value = value
            .trim()
            .parse::<f64>()
            .map_err(|err| format!("{err}: {value:?}"))?;
        pairs.push((name.trim().to_string(), value));
    }
    Ok(Overrides { pairs })
}

fn parse_defines(s: &str) -> Result<Defines, String> {
    let mut pairs = vec![];
    for pair in s.split(',') {
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        pairs.push((name.trim().to_string(), value.trim().to_string()));
    }
    Ok(Defines { pairs })
}

fn parse_capabilities(s: &str) -> Result<naga::valid::Capabilities, String> {
    use naga::valid::Capabilities;
    let s = s.to_uppercase();
    if s == "NONE" {
        Ok(Capabilities::empty())
    } else if s == "ALL" {
        Ok(Capabilities::all())
    } else if let Ok(bits) = s.parse::<u64>() {
        Capabilities::from_bits(bits)
            .ok_or_else(|| format!("Invalid capabilities bitflags value: {bits}"))
    } else {
        s.split(',').try_fold(Capabilities::empty(), |acc, name| {
            Capabilities::from_name(name.trim())
                .map(|cap| acc | cap)
                .ok_or_else(|| format!("Unknown capability {}", name.trim()))
        })
    }
}

fn parse_task_limits(s: &str) -> Result<naga::back::TaskDispatchLimits, String> {
    let (x, y) = s
        .split_once(',')
        .ok_or_else(|| format!("No comma present for --task-limits value: {s}"))?;
    Ok(naga::back::TaskDispatchLimits {
        max_mesh_workgroups_per_dim: x.parse().map_err(|e: core::num::ParseIntError| e.to_string())?,
        max_mesh_workgroups_total: y.parse().map_err(|e: core::num::ParseIntError| e.to_string())?,
    })
}
```

Note: `--capabilities` defaults to `all` via `default_value = "all"`, preserving the current `Capabilities::all()` default. `--task-limits` becomes `Option` (default `None`), matching `TaskDispatchLimitsArg(None)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p naga-cli --bin naga cli::tests`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add naga-cli/src/cli.rs
git commit -m "refactor(naga-cli): port arguments to clap"
```

---

### Task 3: `params.rs` — build `Parameters` from `Args`

Move the `Parameters` struct (naga.rs:404-429) here and extract the arg→params mapping (naga.rs:508-581) into a pure function.

**Files:**
- Modify: `naga-cli/src/params.rs`
- Test: `naga-cli/src/params.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Consumes: `crate::cli::{Args, BoundsCheckPolicyArg, ShaderStageArg, InputKind}`.
- Produces:
  - `pub struct Parameters<'a>` — identical fields to naga.rs:404-429 (`validation_flags`, `bounds_check_policies`, `entry_point`, `keep_coordinate_space`, `overrides`, `spv_in`, `spv_out`, `dot`, `msl`, `glsl`, `hlsl`, `input_kind: Option<InputKind>`, `shader_stage: Option<ShaderStageArg>`, `defines`, `capabilities`, `compact`), deriving `Default`.
  - `pub fn build_parameters(args: &Args) -> anyhow::Result<Parameters<'static>>`.

- [ ] **Step 1: Write the failing test**

Append to `naga-cli/src/params.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Args;
    use clap::Parser;

    #[test]
    fn buffer_policy_defaults_to_index_policy() {
        let args = Args::try_parse_from([
            "naga",
            "--index-bounds-check-policy",
            "restrict",
            "in.wgsl",
        ])
        .unwrap();
        let params = build_parameters(&args).unwrap();
        assert_eq!(
            params.bounds_check_policies.buffer,
            naga::proc::BoundsCheckPolicy::Restrict
        );
        assert_eq!(
            params.bounds_check_policies.image_load,
            naga::proc::BoundsCheckPolicy::Restrict
        );
    }

    #[test]
    fn invalid_validate_bits_error() {
        let args = Args::try_parse_from(["naga", "--validate", "255", "in.wgsl"]).unwrap();
        // 255 has bits outside ValidationFlags; build should error rather than panic.
        assert!(build_parameters(&args).is_err());
    }
}
```

(If `--validate 255` happens to be valid on the current `ValidationFlags`, change the second test to use a value known to be out of range; check `naga::valid::ValidationFlags::all().bits()` and pick `all_bits + 1`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p naga-cli --bin naga params::tests`
Expected: FAIL to compile — `build_parameters`/`Parameters` not found.

- [ ] **Step 3: Implement `params.rs`**

Replace the contents of `naga-cli/src/params.rs` with:

```rust
//! Translation parameters assembled from parsed CLI arguments.

use crate::cli::{Args, InputKind, ShaderStageArg};
use anyhow::anyhow;
use naga::FastHashMap;

#[derive(Default)]
pub struct Parameters<'a> {
    pub validation_flags: naga::valid::ValidationFlags,
    pub bounds_check_policies: naga::proc::BoundsCheckPolicies,
    pub entry_point: Option<String>,
    pub keep_coordinate_space: bool,
    pub overrides: naga::back::PipelineConstants,
    pub spv_in: naga::front::spv::Options,
    pub spv_out: naga::back::spv::Options<'a>,
    pub dot: naga::back::dot::Options,
    pub msl: naga::back::msl::Options,
    pub glsl: naga::back::glsl::Options,
    pub hlsl: naga::back::hlsl::Options,
    pub input_kind: Option<InputKind>,
    pub shader_stage: Option<ShaderStageArg>,
    pub defines: FastHashMap<String, String>,
    pub capabilities: naga::valid::Capabilities,
    /// Whether to pass the entry point to `process_overrides` (drops unreachable items).
    pub compact: bool,
}

pub fn build_parameters(args: &Args) -> anyhow::Result<Parameters<'static>> {
    let mut params = Parameters::default();

    if let Some(bits) = args.validate {
        params.validation_flags = naga::valid::ValidationFlags::from_bits(bits)
            .ok_or_else(|| anyhow!("Invalid validation flags: {bits}"))?;
    }

    if let Some(policy) = args.index_bounds_check_policy {
        params.bounds_check_policies.index = policy.to_policy();
    }
    params.bounds_check_policies.buffer = match args.buffer_bounds_check_policy {
        Some(p) => p.to_policy(),
        None => params.bounds_check_policies.index,
    };
    params.bounds_check_policies.image_load = match args.image_load_bounds_check_policy {
        Some(p) => p.to_policy(),
        None => params.bounds_check_policies.index,
    };

    params.overrides = args
        .overrides
        .iter()
        .flat_map(|o| o.pairs.iter().cloned())
        .collect();
    params.defines = args
        .defines
        .iter()
        .flat_map(|o| o.pairs.iter().cloned())
        .collect();

    params.spv_in = naga::front::spv::Options {
        adjust_coordinate_space: !args.keep_coordinate_space,
        strict_capabilities: false,
        block_ctx_dump_prefix: args.block_ctx_dir.clone(),
    };

    params.entry_point.clone_from(&args.entry_point);
    if let Some(version) = args.profile {
        params.glsl.version = version;
    }
    if let Some(ref model) = args.shader_model {
        params.hlsl.shader_model = model.clone();
    }
    if let Some(version) = args.metal_version {
        params.msl.lang_version = version;
    }
    if let Some(version) = args.spirv_version {
        params.spv_out.lang_version = version;
    }
    params.keep_coordinate_space = args.keep_coordinate_space;
    params.dot.cfg_only = args.dot_cfg_only;

    params.spv_out.bounds_check_policies = params.bounds_check_policies;
    params.spv_out.flags.set(
        naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE,
        !params.keep_coordinate_space,
    );
    params.glsl.writer_flags.set(
        naga::back::glsl::WriterFlags::ADJUST_COORDINATE_SPACE,
        !params.keep_coordinate_space,
    );

    params.compact = args.compact || args.before_compaction.is_some();
    params.capabilities = args.capabilities;

    params.spv_out.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.spv_out.task_dispatch_limits = args.task_limits;
    params.msl.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.msl.task_dispatch_limits = args.task_limits;
    params.hlsl.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.hlsl.task_dispatch_limits = args.task_limits;

    params.input_kind = args.input_kind;
    params.shader_stage = args.shader_stage;

    Ok(params)
}
```

Note: `build_parameters` folds in the `before_compaction ⇒ compact` implication (was naga.rs:494-496), and moves `input_kind`/`shader_stage` assignment (was naga.rs:602-603) in here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p naga-cli --bin naga params::tests`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add naga-cli/src/params.rs
git commit -m "refactor(naga-cli): extract Parameters builder"
```

---

### Task 4: `error.rs` — error type + renderer

Move `CliError` and `print_err` (naga.rs:436-482) here. Drop the `PrettyResult`/`unwrap_pretty` trait entirely — Task 5's core functions propagate errors instead of exiting.

**Files:**
- Modify: `naga-cli/src/error.rs`

**Interfaces:**
- Produces:
  - `pub struct CliError(pub &'static str)` implementing `Display` + `std::error::Error`.
  - `pub fn print_err(error: &dyn std::error::Error)`.

- [ ] **Step 1: Implement `error.rs`**

Replace the contents of `naga-cli/src/error.rs` with:

```rust
//! CLI error type and human-readable rendering.

use std::error::Error;
use std::fmt;

/// A simple static-message CLI error.
#[derive(Debug, Clone)]
pub struct CliError(pub &'static str);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for CliError {}

/// Print an error and its source chain to stderr.
#[cold]
#[inline(never)]
pub fn print_err(error: &dyn Error) {
    eprint!("{error}");

    let mut e = error.source();
    if e.is_some() {
        eprintln!(": ");
    } else {
        eprintln!();
    }

    while let Some(source) = e {
        eprintln!("\t{source}");
        e = source.source();
    }
}
```

- [ ] **Step 2: Verify it builds**

Run: `cargo build -p naga-cli`
Expected: builds (unused-warnings acceptable; `bin/naga.rs` still present and separate).

- [ ] **Step 3: Commit**

```bash
git add naga-cli/src/error.rs
git commit -m "refactor(naga-cli): move error type and renderer"
```

---

### Task 5: `core.rs` — de-paniced translation functions

Move `parse_input`, `write_output`, `bulk_validate` and the `run()` orchestration (minus arg parsing) into `core.rs`, converting every `process::exit`/`unwrap_pretty`/`expect` into a returned `anyhow::Error`.

**Files:**
- Modify: `naga-cli/src/core.rs`
- Test: `naga-cli/tests/cli.rs` (added in Task 6; this task is verified by build + Task 6 tests)

**Interfaces:**
- Consumes: `crate::cli::{Args, InputKind}`, `crate::params::Parameters`, `crate::error::CliError`.
- Produces:
  - `pub fn run(args: &Args, params: &mut Parameters) -> anyhow::Result<()>` — the top-level flow from naga.rs:583-727 (bulk dispatch, input read, parse, validate, compact, output loop).
  - `pub struct Parsed { pub module: naga::Module, pub input_text: Option<String>, pub language: naga::back::spv::SourceLanguage }`.
  - `fn parse_input(input_path: &std::path::Path, input: Vec<u8>, params: &Parameters) -> anyhow::Result<Parsed>`.
  - `fn write_output(module: &naga::Module, info: &Option<naga::valid::ModuleInfo>, params: &Parameters, output_path: &str) -> anyhow::Result<()>`.
  - `fn bulk_validate(files: &[String], params: &Parameters) -> anyhow::Result<()>`.

- [ ] **Step 1: Implement `core.rs` — move `parse_input`**

Copy `parse_input` verbatim from naga.rs:736-824 into `core.rs`, with these exact changes:
- Add imports at top of file (see Step 4 for the full import block).
- In the `InputKind` match, rename variants to the clap enum: `InputKind::Bincode` → `InputKind::Bin`, `InputKind::SpirV` → `InputKind::Spv`, `InputKind::Glsl` → `InputKind::Glsl`, `InputKind::Wgsl` → `InputKind::Wgsl`.
- Replace the GLSL error path (naga.rs:810-818) that calls `std::process::exit(1)`:

```rust
InputKind::Glsl => {
    let shader_stage = match params.shader_stage {
        Some(stage) => stage.to_stage(),
        None => {
            let file_stem = input_path
                .file_stem()
                .context("Unable to determine file stem from input filename.")?;
            let inner_ext = Path::new(file_stem)
                .extension()
                .context("Unable to determine inner extension from input filename.")?
                .to_str()
                .context("Input filename not valid unicode")?;
            match inner_ext {
                "vert" => naga::ShaderStage::Vertex,
                "frag" => naga::ShaderStage::Fragment,
                "comp" => naga::ShaderStage::Compute,
                other => return Err(anyhow!("Unknown GLSL stage extension: {other}")),
            }
        }
    };
    let input = String::from_utf8(input)?;
    let mut parser = naga::front::glsl::Frontend::default();
    let module = parser
        .parse(
            &naga::front::glsl::Options {
                stage: shader_stage,
                defines: params.defines.clone(),
            },
            &input,
        )
        .map_err(|error| {
            let filename = input_path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or("glsl");
            anyhow!(
                "Could not parse GLSL:\n{}",
                error.emit_to_string_with_path(&input, filename)
            )
        })?;
    Parsed {
        module,
        input_text: Some(input),
        language: naga::back::spv::SourceLanguage::GLSL,
    }
}
```

(`naga::front::glsl::ParseErrors` exposes `emit_to_string_with_path`; if the exact method name differs, use the same signature the WGSL path uses at naga.rs:776. Verify against `naga/src/front/glsl/error.rs`.)

- The `SpirV`/`Bincode`/`Wgsl` arms are copied verbatim, only the variant names changed.

- [ ] **Step 2: Implement `core.rs` — move `write_output`, de-panic it**

Copy `write_output` verbatim from naga.rs:826-1022 into `core.rs` with these exact changes:
- The entry-point lookup (naga.rs:832-840) `.expect("Unable to find the entry point")` becomes an error:

```rust
let entry_point = match params.entry_point.as_deref() {
    Some(name) => {
        let ep_index = module
            .entry_points
            .iter()
            .position(|ep| ep.name == *name)
            .ok_or_else(|| anyhow!("Unable to find the entry point: {name}"))?;
        Some((module.entry_points[ep_index].stage, name))
    }
    None => None,
};
```

- Replace every `.unwrap_pretty()` with `?`. Specifically:
  - naga.rs:879 `...process_overrides(...).unwrap_pretty();` → `...process_overrides(...)?;`
  - naga.rs:883 `msl::write_string(...).unwrap_pretty();` → `msl::write_string(...)?;`
  - naga.rs:905, 907-908 (spv): `.unwrap_pretty()` → `?`
  - naga.rs:959 (glsl process_overrides), 970 (`glsl::Writer::new(...).unwrap_pretty()`) → `?`
  - naga.rs:994 (hlsl process_overrides), 999 (`writer.write(...).unwrap_pretty()`) → `?`
  - naga.rs:1013 (wgsl `write_string(...).unwrap_pretty()`) → `?`
- Everything else (the `match` on extension, byte packing, `fs::write`) is copied verbatim.

- [ ] **Step 3: Implement `core.rs` — move `run` + `bulk_validate`, de-panic them**

Copy the body of the current `run()` from naga.rs:583-727 into a new `pub fn run(args: &Args, params: &mut Parameters) -> anyhow::Result<()>`, with these exact changes:
- Delete the arg-parsing / logging / `--version` prelude (naga.rs:485-508) — that stays in `main.rs`.
- Delete the `Parameters::default()` + all arg→params mapping (naga.rs:508-581) — now done by `build_parameters`.
- `bulk_validate(&args, &params)` (naga.rs:584) → `bulk_validate(&args.files, params)`.
- The no-output validation-failure branch (naga.rs:714-721):

```rust
if output_paths.clone().next().is_none() {
    if info.is_some() {
        println!("Validation successful");
        return Ok(());
    } else {
        return Err(CliError("Validation failed").into());
    }
}
```

(was `std::process::exit(-1)`.)
- The validation-diagnostic emission blocks (naga.rs:658-668, 690-701) are copied verbatim — they emit source diagnostics to stderr and set `info = None`. This is interim; Phase 4 routes diagnostics through structured output. Add a `// TODO(phase 4): route through structured diagnostics` comment above each.

Then copy `bulk_validate` from naga.rs:1024-1076 with the signature change `fn bulk_validate(files: &[String], params: &Parameters) -> anyhow::Result<()>` and iterate `for input_path in files`.

- [ ] **Step 4: Add the module's import block**

At the top of `core.rs` (below the doc comment), add:

```rust
use crate::cli::{Args, InputKind};
use crate::error::CliError;
use crate::params::Parameters;
use anyhow::{anyhow, Context as _};
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use naga::compact::KeepUnused;
use std::fs;
use std::io::Read as _;
use std::path::Path;
```

Remove any now-unused imports flagged by the compiler (e.g. `StandardStream`/`ColorChoice` if the GLSL path no longer uses them — the new GLSL error path uses `emit_to_string_with_path`, so those imports may be droppable; let the compiler guide you).

- [ ] **Step 5: Verify it builds**

Run: `cargo build -p naga-cli`
Expected: builds. Resolve any unused-import warnings.

- [ ] **Step 6: Commit**

```bash
git add naga-cli/src/core.rs
git commit -m "refactor(naga-cli): move translation core, replace exits with errors"
```

---

### Task 6: `main.rs` wiring + integration tests + delete old binary

Wire the modules together in a thin `main`, add integration tests that exec the binary, and remove `src/bin/naga.rs`.

**Files:**
- Modify: `naga-cli/src/main.rs`
- Create: `naga-cli/tests/cli.rs`
- Create: `naga-cli/tests/snapshots/help.txt`
- Delete: `naga-cli/src/bin/naga.rs`

**Interfaces:**
- Consumes: `crate::cli::Args`, `crate::params::build_parameters`, `crate::core::run`, `crate::error::print_err`.

- [ ] **Step 1: Implement `main.rs`**

Replace the contents of `naga-cli/src/main.rs` with:

```rust
mod cli;
mod core;
mod error;
mod params;

use clap::Parser as _;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = cli::Args::parse();

    if let Err(e) = real_main(&args) {
        error::print_err(e.as_ref());
        std::process::exit(1);
    }
}

fn real_main(args: &cli::Args) -> anyhow::Result<()> {
    let mut params = params::build_parameters(args)?;
    core::run(args, &mut params)
}
```

Note: `--version` is now handled by clap's `#[command(version)]` derive (Task 2), replacing the manual `args.version` branch.

- [ ] **Step 2: Delete the old binary**

Run: `git rm naga-cli/src/bin/naga.rs`

- [ ] **Step 3: Verify the whole crate builds and unit tests pass**

Run: `cargo build -p naga-cli && cargo test -p naga-cli --bin naga`
Expected: builds; `cli::tests` and `params::tests` PASS.

- [ ] **Step 4: Write the failing integration test**

Create `naga-cli/tests/cli.rs`:
```rust
use std::process::Command;

/// Path to the built `naga` binary, provided by cargo to integration tests.
fn naga() -> Command {
    Command::new(env!("CARGO_BIN_EXE_naga"))
}

#[test]
fn help_lists_all_options() {
    let out = naga().arg("--help").output().unwrap();
    assert!(out.status.success());
    let help = String::from_utf8(out.stdout).unwrap();
    for flag in [
        "--validate",
        "--index-bounds-check-policy",
        "--buffer-bounds-check-policy",
        "--image-load-bounds-check-policy",
        "--entry-point",
        "--profile",
        "--shader-model",
        "--spirv-version",
        "--shader-stage",
        "--input-kind",
        "--metal-version",
        "--keep-coordinate-space",
        "--stdin-file-path",
        "--compact",
        "--bulk-validate",
        "--override",
        "--defines",
        "--capabilities",
        "--task-limits",
        "--validate-mesh-output",
    ] {
        assert!(help.contains(flag), "help missing {flag}\n---\n{help}");
    }
}

#[test]
fn validates_wgsl_from_file() {
    let dir = std::env::temp_dir().join("naga_cli_phase1_validate");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("ok.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();

    let out = naga().arg(&src).output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Validation successful"));
}

#[test]
fn compiles_wgsl_to_spv() {
    let dir = std::env::temp_dir().join("naga_cli_phase1_spv");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    let dst = dir.join("s.spv");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();

    let out = naga().arg(&src).arg(&dst).output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let bytes = std::fs::read(&dst).unwrap();
    // SPIR-V magic number 0x07230203, little-endian.
    assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07]);
}

#[test]
fn reads_wgsl_from_stdin() {
    use std::io::Write;
    let mut child = naga()
        .args(["--stdin-file-path", "in.wgsl"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Validation successful"));
}
```

- [ ] **Step 5: Run integration tests to verify they pass**

Run: `cargo test -p naga-cli --test cli`
Expected: PASS (4 tests). If `help_lists_all_options` fails, the missing flag names identify gaps against Task 2 — fix `cli.rs`.

- [ ] **Step 6: Capture the help snapshot (guard against drift)**

Run and save:
```bash
cargo run -q -p naga-cli -- --help > naga-cli/tests/snapshots/help.txt
```
Then append this test to `naga-cli/tests/cli.rs`:
```rust
#[test]
fn help_matches_snapshot() {
    let expected = include_str!("snapshots/help.txt");
    let out = naga().arg("--help").output().unwrap();
    let actual = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "--help output changed. If intentional, regenerate:\n\
         cargo run -q -p naga-cli -- --help > naga-cli/tests/snapshots/help.txt"
    );
}
```

- [ ] **Step 7: Run the full test suite**

Run: `cargo test -p naga-cli`
Expected: all unit + integration tests PASS (5 integration tests, 4 unit tests).

- [ ] **Step 8: Commit**

```bash
git add naga-cli/src/main.rs naga-cli/tests/cli.rs naga-cli/tests/snapshots/help.txt
git commit -m "refactor(naga-cli): wire clap main, add integration tests, drop argh binary"
```

---

### Task 7: Panic-path audit sweep

Confirm no user-input-reachable panic/exit remains outside `main`.

**Files:**
- Review: `naga-cli/src/*.rs`

- [ ] **Step 1: Grep for remaining exit/panic paths**

Run: `grep -rnE "process::exit|unwrap_pretty|\.unwrap\(\)|\.expect\(|panic!|unreachable!" naga-cli/src`
Expected findings and required dispositions:
- `std::process::exit(1)` — allowed ONLY in `main.rs` `main()`.
- `unwrap_pretty` — MUST have zero matches (trait deleted).
- `.unwrap()`/`.expect(` — allowed only on values that cannot depend on user input (e.g. writing to an in-memory `String` via `write!`). Any that can fail on user input must become `?`/`anyhow!`. Document each remaining one with a trailing `// infallible: <reason>` comment.
- `unreachable!()` — the `vert|frag|comp` match arm in `write_output` (naga.rs:925) is reachable only for those three literals matched just above; leave it, add `// exhaustive: guarded by outer match`.

- [ ] **Step 2: Fix any user-reachable unwrap/expect found**

For each offending site, convert to `?` with context, e.g.:
```rust
let x = something().context("describe what failed")?;
```

- [ ] **Step 3: Verify build + tests still pass**

Run: `cargo test -p naga-cli`
Expected: all PASS.

- [ ] **Step 4: Verify clippy is clean**

Run: `cargo clippy -p naga-cli --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add naga-cli/src
git commit -m "refactor(naga-cli): audit and remove user-reachable panics"
```

---

## Self-Review

**Spec coverage (Phase 1 slice):**
- argh→clap migration → Tasks 1-2, 6. ✓
- Split monolith into focused modules → Tasks 2-6 (`cli`/`params`/`core`/`error`/`main`). ✓
- Reduce panics → Tasks 5, 7 (all `unwrap_pretty`/`process::exit`/`expect` on user paths removed). ✓
- Up-to-date help guard → Task 6 (snapshot + flag-presence tests). ✓
- Golden compile tests → Task 6. ✓
- stdin preserved → Task 6 (`reads_wgsl_from_stdin`). ✓
- NOT in this phase (deferred, by design): naga-core derives (Phase 2), `--config`/`--config-json`/`--print-config-schema` (Phase 3), `--format json`/structured diagnostics (Phase 4), tool hooks (Phase 5), examples/README (Phase 6). Interim: validation diagnostics still emit to stderr in `core.rs` (flagged with `TODO(phase 4)`).

**Placeholder scan:** no "TBD"/"implement later"; two intentional forward-references are `TODO(phase 4)` comments on interim diagnostic emission, which is a deliberate scope boundary, not a gap in Phase 1.

**Type consistency:**
- `Args` fields used in `params.rs`/`main.rs` match the definitions in Task 2 (`overrides: Vec<Overrides>`, `defines: Vec<Defines>`, `capabilities: naga::valid::Capabilities`, `task_limits: Option<TaskDispatchLimits>`, `shader_stage: Option<ShaderStageArg>`, `spirv_version: Option<(u8,u8)>`, `metal_version: Option<(u8,u8)>`, `profile: Option<GlslProfile>`, `shader_model: Option<ShaderModel>`).
- `Parameters` field names match naga.rs:404-429 exactly, so `core.rs` (copied from `write_output`/`run`) references resolve unchanged except the `input_kind`/`shader_stage` types, now `InputKind`/`ShaderStageArg` (both used only in `parse_input`, updated in Task 5 Step 1).
- `BoundsCheckPolicyArg::to_policy`, `ShaderStageArg::to_stage` are defined in Task 2 and consumed in Task 3/Task 5.

**Known verification points flagged for the implementer** (not placeholders — confirm against source during the task):
- GLSL parse-error rendering method name (`emit_to_string_with_path`) — Task 5 Step 1 says to match the WGSL path if it differs.
- `--validate 255` out-of-range assumption — Task 3 Step 1 note gives the fallback.
