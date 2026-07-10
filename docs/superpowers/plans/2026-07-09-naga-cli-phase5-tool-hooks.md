# naga-cli Rewrite — Phase 5 (External Tool Hooks) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--spirv-val`, `--spirv-opt`, and `--dxc` flags to the naga CLI that, after writing a shader output, run the corresponding tool (found on `PATH`) as a subprocess — validating SPIR-V, optimizing SPIR-V, and compiling generated HLSL to DXIL respectively — so a single `naga` invocation covers the whole chain.

**Architecture:** CLI-only. A new `naga-cli/src/hooks.rs` locates tools via the `which` crate and runs them via `std::process::Command` (mirroring the patterns in `naga/xtask/src/validate.rs`). Each hook is a function returning `anyhow::Result<()>`; write paths in `core.rs` call them after `fs::write`. HLSL→DXIL uses the `ReflectionInfo` returned by `hlsl::Writer::write` (currently discarded) for entry-point names, plus a stage+shader-model → DXC profile mapping.

**Tech Stack:** Rust, `which` (workspace dep), `std::process::Command`, naga.

## Global Constraints

- No changes to `naga/` or any crate other than `naga-cli`. `ReflectionInfo` is already returned by `hlsl::Writer::write` — just stop discarding it.
- Tools are located on `PATH` via `which::which(name)`. If a requested tool is absent, fail with a clear, actionable error (the user explicitly asked for it — never silently skip).
- Hooks are opt-in via flags; default behavior is byte-identical to today. Existing tests must pass unchanged.
- A hook flag with no matching output is a hard error (e.g. `--spirv-val` with no `.spv`/`.spirv` output path): don't silently no-op.
- Integration tests MUST skip gracefully when the tool is not installed (guard with `which`), so the suite passes on machines without dxc/spirv-tools; CI has them.
- `which = "8"` is a workspace dependency; add `which = { workspace = true }` to naga-cli.
- Hook ordering when multiple apply to the same SPIR-V file: `spirv-opt` first (rewrites the file), then `spirv-val` (validates the final bytes).

## File Structure

- `naga-cli/Cargo.toml` — add `which`.
- `naga-cli/src/hooks.rs` — NEW: `Hooks` struct, tool runners, DXC profile mapping.
- `naga-cli/src/cli.rs` — `--spirv-val`, `--spirv-opt`, `--dxc` flags.
- `naga-cli/src/core.rs` — capture HLSL `ReflectionInfo`; call hooks after writes; pre-check flag-without-output.
- `naga-cli/src/main.rs` — `mod hooks;`.
- `naga-cli/tests/cli.rs` — skip-if-absent integration tests.

---

### Task 1: `hooks.rs` — tool runners + profile mapping

Locate and invoke the tools; map (stage, shader model) → DXC profile.

**Files:**
- Modify: `naga-cli/Cargo.toml`, `naga-cli/src/main.rs`
- Create: `naga-cli/src/hooks.rs`
- Test: inline `#[cfg(test)]` (profile mapping only — pure, no subprocess)

**Interfaces:**
- Produces:
  - `pub struct Hooks { pub spirv_val: bool, pub spirv_opt: bool, pub dxc: bool }`, `impl Hooks { pub fn any(&self) -> bool }`.
  - `pub fn dxc_profile(stage: naga::ShaderStage, model: naga::back::hlsl::ShaderModel) -> String` (e.g. `(Compute, V6_0)` → `"cs_6_0"`).
  - `pub fn run_spirv_val(spv_path: &std::path::Path) -> anyhow::Result<()>`
  - `pub fn run_spirv_opt(spv_path: &std::path::Path) -> anyhow::Result<()>`
  - `pub fn run_dxc(hlsl_path: &std::path::Path, entry_points: &[(String, naga::ShaderStage)], model: naga::back::hlsl::ShaderModel) -> anyhow::Result<()>`

- [ ] **Step 1: Add deps + module**

`naga-cli/Cargo.toml` `[dependencies]`: `which = { workspace = true }`.
`naga-cli/src/main.rs`: add `mod hooks;`.

- [ ] **Step 2: Write the failing test (profile mapping)**

Create `naga-cli/src/hooks.rs`:
```rust
//! External tool hooks: spirv-val, spirv-opt, dxc (subprocesses on PATH).
```
Append:
```rust
#[cfg(test)]
mod tests {
    use super::dxc_profile;
    use naga::back::hlsl::ShaderModel;
    use naga::ShaderStage;

    #[test]
    fn profile_mapping() {
        assert_eq!(dxc_profile(ShaderStage::Vertex, ShaderModel::V6_0), "vs_6_0");
        assert_eq!(dxc_profile(ShaderStage::Fragment, ShaderModel::V6_2), "ps_6_2");
        assert_eq!(dxc_profile(ShaderStage::Compute, ShaderModel::V6_0), "cs_6_0");
        assert_eq!(dxc_profile(ShaderStage::Mesh, ShaderModel::V6_5), "ms_6_5");
        assert_eq!(dxc_profile(ShaderStage::Task, ShaderModel::V6_5), "as_6_5");
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p naga-cli --bin naga hooks::tests`
Expected: FAIL to compile — `dxc_profile` undefined.

- [ ] **Step 4: Implement `hooks.rs`**

```rust
//! External tool hooks: spirv-val, spirv-opt, dxc (subprocesses on PATH).

use anyhow::{anyhow, bail, Context as _};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, Default)]
pub struct Hooks {
    pub spirv_val: bool,
    pub spirv_opt: bool,
    pub dxc: bool,
}

impl Hooks {
    pub fn any(&self) -> bool {
        self.spirv_val || self.spirv_opt || self.dxc
    }
}

/// Map a shader stage + HLSL shader model to a DXC target profile, e.g. `cs_6_0`.
pub fn dxc_profile(stage: naga::ShaderStage, model: naga::back::hlsl::ShaderModel) -> String {
    let prefix = match stage {
        naga::ShaderStage::Vertex => "vs",
        naga::ShaderStage::Fragment => "ps",
        naga::ShaderStage::Compute => "cs",
        naga::ShaderStage::Mesh => "ms",
        naga::ShaderStage::Task => "as",
        // Ray-tracing stages compile under lib_*; fall back to lib for anything else.
        _ => "lib",
    };
    format!("{prefix}_{}", model.to_str())
}

/// Locate a tool on PATH or produce an actionable error.
fn find_tool(name: &str) -> anyhow::Result<std::path::PathBuf> {
    which::which(name).map_err(|_| {
        anyhow!("`{name}` was not found on PATH; install it or remove the corresponding flag")
    })
}

pub fn run_spirv_val(spv_path: &Path) -> anyhow::Result<()> {
    let tool = find_tool("spirv-val")?;
    let output = Command::new(&tool)
        .arg(spv_path)
        .output()
        .with_context(|| format!("failed to run spirv-val ({})", tool.display()))?;
    if !output.status.success() {
        bail!(
            "spirv-val failed for {}:\n{}",
            spv_path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

pub fn run_spirv_opt(spv_path: &Path) -> anyhow::Result<()> {
    let tool = find_tool("spirv-opt")?;
    // Optimize in place: read from spv_path, write back to spv_path.
    let output = Command::new(&tool)
        .arg(spv_path)
        .arg("-O")
        .arg("-o")
        .arg(spv_path)
        .output()
        .with_context(|| format!("failed to run spirv-opt ({})", tool.display()))?;
    if !output.status.success() {
        bail!(
            "spirv-opt failed for {}:\n{}",
            spv_path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

pub fn run_dxc(
    hlsl_path: &Path,
    entry_points: &[(String, naga::ShaderStage)],
    model: naga::back::hlsl::ShaderModel,
) -> anyhow::Result<()> {
    let tool = find_tool("dxc")?;
    if entry_points.is_empty() {
        bail!("--dxc: no entry points to compile in {}", hlsl_path.display());
    }
    for (name, stage) in entry_points {
        let profile = dxc_profile(*stage, model);
        // Output: <hlsl_path stem>.<entry>.dxil next to the HLSL file.
        let out = hlsl_path.with_extension(format!("{name}.dxil"));
        let output = Command::new(&tool)
            .arg(hlsl_path)
            .arg("-T")
            .arg(&profile)
            .arg("-E")
            .arg(name)
            .arg("-Fo")
            .arg(&out)
            .output()
            .with_context(|| format!("failed to run dxc ({})", tool.display()))?;
        if !output.status.success() {
            bail!(
                "dxc failed for entry `{name}` ({profile}) in {}:\n{}",
                hlsl_path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    Ok(())
}
```
(If `naga::ShaderStage` has more variants that DXC supports distinctly, the `_ => "lib"` arm is a safe fallback; adjust if a specific stage needs a distinct prefix. `model.to_str()` yields `"6_0"` etc. per `naga/src/back/hlsl/mod.rs:271`.)

- [ ] **Step 5: Run to verify pass**

Run: `cargo test -p naga-cli --bin naga hooks::tests`
Expected: PASS. (`run_*` fns are unused-warning until Task 2; acceptable, or `#[allow(dead_code)]` temporarily — prefer wiring in Task 2 promptly.)

- [ ] **Step 6: Commit**

```bash
git add naga-cli/Cargo.toml naga-cli/src/hooks.rs naga-cli/src/main.rs
git commit -m "feat(naga-cli): tool-hook runners for spirv-val/spirv-opt/dxc"
```

---

### Task 2: flags + wiring in `core.rs`

Add the flags, capture HLSL reflection, run hooks after writes, and pre-check flag-without-output.

**Files:**
- Modify: `naga-cli/src/cli.rs`, `naga-cli/src/core.rs`
- Test: covered by Task 3 integration tests (+ build).

**Interfaces:**
- Consumes: `crate::hooks::{Hooks, run_spirv_val, run_spirv_opt, run_dxc}`.
- Produces: `Args` gains `pub spirv_val: bool`, `pub spirv_opt: bool`, `pub dxc: bool`.

- [ ] **Step 1: Add flags (`cli.rs`)**

```rust
/// After writing SPIR-V output, validate it with `spirv-val` (must be on PATH).
#[arg(long)]
pub spirv_val: bool,

/// After writing SPIR-V output, optimize it in place with `spirv-opt -O` (must be on PATH).
#[arg(long)]
pub spirv_opt: bool,

/// After writing HLSL output, compile each entry point to DXIL with `dxc` (must be on PATH).
#[arg(long)]
pub dxc: bool,
```
(These are processing/post-output actions — do NOT add them to the `--config` exclusion group.)

- [ ] **Step 2: Pre-check in `run` (`core.rs`)**

Build a `Hooks { spirv_val: args.spirv_val, spirv_opt: args.spirv_opt, dxc: args.dxc }`. After the output paths are known and before/around the write loop, if `hooks.any()`, verify each requested hook has a matching output extension among `output_paths`:
- `spirv_val`/`spirv_opt` require at least one `.spv`/`.spirv` output.
- `dxc` requires at least one `.hlsl` output.
If a requested hook has no matching output, return an error, e.g. `Err(anyhow!("--spirv-val requires a SPIR-V (.spv) output file"))`. (This is a hard error in both text and json modes for v1 — document it.)

- [ ] **Step 3: Run SPIR-V hooks after the `.spv` write (`core.rs`)**

In `write_output`'s `"spv" | "spirv"` arm, after `fs::write(output_path, bytes...)`, run (opt then val):
```rust
if hooks.spirv_opt {
    crate::hooks::run_spirv_opt(std::path::Path::new(output_path))?;
}
if hooks.spirv_val {
    crate::hooks::run_spirv_val(std::path::Path::new(output_path))?;
}
```
Thread `hooks: &Hooks` into `write_output` (add the parameter; update its call sites in `run`).

- [ ] **Step 4: Capture reflection + run DXC after the `.hlsl` write (`core.rs`)**

In the `"hlsl"` arm, `hlsl::Writer::write` currently returns a value that is discarded (`writer.write(&module, &info, None)?;`). Capture it:
```rust
let reflection = writer.write(&module, &info, None)?; // naga::back::hlsl::ReflectionInfo
fs::write(output_path, buffer)?;
if hooks.dxc {
    // Map each entry point to its (possibly remapped) HLSL name + stage.
    let entries: Vec<(String, naga::ShaderStage)> = reflection
        .entry_point_names
        .iter()
        .zip(module.entry_points.iter())
        .filter_map(|(name_res, ep)| name_res.as_ref().ok().map(|n| (n.clone(), ep.stage)))
        .collect();
    crate::hooks::run_dxc(
        std::path::Path::new(output_path),
        &entries,
        params.hlsl.shader_model,
    )?;
}
```
(`ReflectionInfo.entry_point_names: Vec<Result<String, EntryPointError>>` per `naga/src/back/hlsl/mod.rs:621-631`; it aligns index-wise with `module.entry_points`. Skip entries whose name is an `Err`. `params.hlsl.shader_model` is the `--shader-model` value, default present.)

- [ ] **Step 5: Build**

Run: `cargo build -p naga-cli`
Expected: builds. Resolve any unused warnings from Task 1 (now consumed).

- [ ] **Step 6: Commit**

```bash
git add naga-cli/src/cli.rs naga-cli/src/core.rs
git commit -m "feat(naga-cli): wire --spirv-val/--spirv-opt/--dxc hooks into output"
```

---

### Task 3: integration tests (skip-if-absent) + polish

**Files:**
- Modify: `naga-cli/tests/cli.rs`, `naga-cli/tests/snapshots/help.txt`

- [ ] **Step 1: Write the tests**

Add to `naga-cli/tests/cli.rs`:
```rust
/// True if a tool is on PATH (tests skip when absent).
fn tool_on_path(name: &str) -> bool {
    which::which(name).is_ok()
}

#[test]
fn spirv_val_hook_when_available() {
    if !tool_on_path("spirv-val") {
        eprintln!("skipping spirv_val_hook_when_available: spirv-val not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("naga_cli_p5_val");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("s.spv");
    let r = naga().arg(&src).arg(&out).arg("--spirv-val").output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    assert!(out.exists());
}

#[test]
fn spirv_opt_hook_when_available() {
    if !tool_on_path("spirv-opt") {
        eprintln!("skipping spirv_opt_hook_when_available: spirv-opt not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("naga_cli_p5_opt");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("s.spv");
    let r = naga().arg(&src).arg(&out).arg("--spirv-opt").output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    // Still valid SPIR-V after optimization.
    assert_eq!(&std::fs::read(&out).unwrap()[0..4], &[0x03, 0x02, 0x23, 0x07]);
}

#[test]
fn dxc_hook_when_available() {
    if !tool_on_path("dxc") {
        eprintln!("skipping dxc_hook_when_available: dxc not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("naga_cli_p5_dxc");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("s.hlsl");
    let r = naga().arg(&src).arg(&out).args(["--dxc", "--shader-model", "60"]).output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    assert!(dir.join("s.main.dxil").exists());
}

#[test]
fn spirv_val_without_spv_output_errors() {
    let dir = std::env::temp_dir().join("naga_cli_p5_noout");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    // No .spv output path → --spirv-val should error (not silently no-op).
    let r = naga().arg(&src).arg("--spirv-val").output().unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).to_lowercase().contains("spir"));
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p naga-cli --test cli`
Expected: all pass; the three tool tests skip (print "skipping…") if tools absent; `spirv_val_without_spv_output_errors` passes regardless (no tool needed).

- [ ] **Step 3: Regenerate help snapshot + clippy + full suite**

Run:
```
cargo run -q -p naga-cli -- --help > naga-cli/tests/snapshots/help.txt
cargo clippy -p naga-cli --all-targets -- -D warnings
cargo test -p naga-cli
```
Expected: all pass, clippy clean.

- [ ] **Step 4: Commit**

```bash
git add naga-cli/tests
git commit -m "test(naga-cli): tool-hook integration tests (skip if tool absent)"
```

---

## Self-Review

**Spec coverage (Phase 5 slice):**
- DXC hook (HLSL→DXIL, single command) → Tasks 1-2 (`--dxc`, per-entry-point compile via reflection names + profile). ✓
- spirv-opt hook → Tasks 1-2 (`--spirv-opt`, in-place `-O`). ✓
- spirv-val hook → Tasks 1-2 (`--spirv-val`). ✓
- Tools on PATH, graceful actionable error if absent → `find_tool` (Task 1). ✓
- No silent no-op (flag without matching output errors) → Task 2 Step 2. ✓
- NOT in this phase (deferred to Phase 6): examples/README + carried minors (M2/M3/M4 stderr-purity, doc notes). Hook errors in json mode are hard `anyhow` errors (consistent with Phase 4's file-error choice) — documented, not routed to JSON diagnostics in v1.

**Placeholder scan:** No TBD/TODO. The `_ => "lib"` DXC profile fallback for non-graphics/compute stages is a real, documented fallback (ray-tracing uses `lib_*`), not a gap. Test snippets note skip-if-absent.

**Type consistency:**
- `Hooks` built in `core.rs` from `Args.{spirv_val,spirv_opt,dxc}` (Task 2 Step 1) matches the struct (Task 1).
- `run_dxc` consumes `&[(String, naga::ShaderStage)]` built from `ReflectionInfo.entry_point_names` zipped with `module.entry_points` (Task 2 Step 4) — index-aligned per naga's API.
- `params.hlsl.shader_model: naga::back::hlsl::ShaderModel` passed to `run_dxc` / `dxc_profile`; `model.to_str()` used for the version.
- `write_output` gains a `hooks: &Hooks` parameter; all its call sites in `run` updated (Task 2 Step 3).

**Known verification points (not placeholders):** exact `hlsl::ReflectionInfo` field/shape (`entry_point_names: Vec<Result<String, EntryPointError>>`, mapped in exploration); whether `module.entry_points` and `entry_point_names` are strictly index-aligned (they are — reflection is built per entry point in order); DXC `-Fo` output-path semantics.
