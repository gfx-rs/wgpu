# naga-cli Rewrite — Phase 6 (Docs, Examples & Polish) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the naga-cli rewrite: write a README + runnable examples, and clear the accumulated polish items — route advisory warnings into JSON output (finally using `Severity::Warning`), make `--bulk-validate --format json` a clear error instead of silent-empty, and document the `ZeroInitializeWorkgroupMemoryMode::Native` behavior.

**Architecture:** Mostly docs + small CLI polish. One naga-core doc-comment (no behavior change). A `warnings` sink threaded to the two advisory-warning sites so json mode surfaces them as `Severity::Warning` diagnostics. A new `naga-cli/README.md` + `naga-cli/examples/` (sample shader + sample config). An integration test that runs the documented example invocations so the README stays honest.

**Tech Stack:** Rust, serde_json, naga.

## Global Constraints

- naga-cli builds against naga 1.87. Only doc-comment changes touch `naga/`.
- Text-mode behavior stays byte-identical (the two warnings keep going to stderr in text mode). JSON mode gains the warnings as structured `Severity::Warning` diagnostics.
- README examples must be real, working invocations — the Task-3 test runs them.
- CHANGELOG: add an `## Unreleased` entry; the PR link is a placeholder `[#XXXX]` to be filled when the branch becomes a PR (there is no PR yet).
- Carry-forward items being closed here (from prior-phase reviews): P4-M2 (`--generate-debug-symbols` warning), P4-M4 (GLSL stage-mismatch warning), P4-M3 (bulk+json), P3 `Native`-doc, plus the P2/P3 doc notes captured in the README.

## File Structure

- `naga/src/back/mod.rs` — doc-comment on `ZeroInitializeWorkgroupMemoryMode`.
- `naga-cli/src/core.rs` — thread a `warnings` sink; bulk+json error.
- `naga-cli/README.md` — NEW.
- `naga-cli/examples/` — NEW: `triangle.wgsl`, `options.json`.
- `naga-cli/tests/cli.rs` — example-verification tests.
- `CHANGELOG.md` — Unreleased entry.

---

### Task 1: JSON-mode warning routing + bulk+json error + core doc-comment

**Files:**
- Modify: `naga-cli/src/core.rs`
- Modify: `naga/src/back/mod.rs`
- Test: `naga-cli/tests/cli.rs`

**Interfaces:**
- `write_output` gains a `warnings: &mut Vec<crate::output::Diagnostic>` parameter (used only for the glsl stage-mismatch advisory; text mode still `eprintln!`s and pushes nothing meaningful there — see below).

- [ ] **Step 1: Write the failing tests**

Add to `naga-cli/tests/cli.rs`:
```rust
#[test]
fn bulk_validate_json_is_rejected() {
    let dir = std::env::temp_dir().join("naga_cli_p6_bulkjson");
    std::fs::create_dir_all(&dir).unwrap();
    let a = dir.join("a.wgsl");
    std::fs::write(&a, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = naga().args(["--bulk-validate", "--format", "json"]).arg(&a).output().unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).to_lowercase().contains("bulk"),
        "expected a clear bulk+json error, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn json_debug_symbols_warning_is_a_diagnostic() {
    // `-g` with non-human-readable (bincode) input triggers the advisory warning.
    // First produce a .bin module, then feed it back with -g in json mode.
    let dir = std::env::temp_dir().join("naga_cli_p6_gwarn");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let bin = dir.join("s.bin");
    let mk = naga().arg(&src).arg(&bin).output().unwrap();
    assert!(mk.status.success(), "stderr: {}", String::from_utf8_lossy(&mk.stderr));

    let out = naga().arg(&bin).args(["--format", "json", "-g"]).output().unwrap();
    // Producing a .bin re-import with -g: validation still succeeds → success true,
    // but a warning diagnostic about -g on non-human-readable input is present.
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let diags = v["diagnostics"].as_array().unwrap();
    assert!(
        diags.iter().any(|d| d["severity"] == "warning"),
        "expected a warning diagnostic in json mode, got: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}
```
(If a `.bin` re-import path makes the `-g` warning test awkward on this naga version — e.g. the warning fires elsewhere — adjust to whatever reliably triggers the `--generate-debug-symbols`-on-non-text-input advisory, and assert a `"warning"` severity diagnostic appears in json stdout. The point is: that advisory becomes a json Warning, not stderr noise.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p naga-cli --test cli bulk_validate_json_is_rejected json_debug_symbols_warning_is_a_diagnostic`
Expected: FAIL (no bulk+json error; warning currently to stderr, not in json).

- [ ] **Step 3: bulk+json error (core.rs)**

At the bulk dispatch (core.rs:29-31), before calling `bulk_validate`, reject json:
```rust
if args.bulk_validate {
    if args.format == OutputFormat::Json {
        return Err(anyhow!(
            "`--format json` is not supported with `--bulk-validate`"
        ));
    }
    bulk_validate(&args.files, params)?;
    return Ok(true); // match the current run() return type (bool)
}
```
(Use whatever the current `run` signature returns — Phase 4 made it `anyhow::Result<bool>`. Keep it consistent.)

- [ ] **Step 4: Route the `--generate-debug-symbols` advisory (core.rs ~99-103)**

The debug-symbols warning lives in `run` (or `parse_input_json`/setup) where the diagnostics vec is in scope. In json mode, instead of `eprintln!`, push:
```rust
diagnostics.push(crate::output::Diagnostic {
    severity: crate::output::Severity::Warning,
    message: format!(
        "`--generate-debug-symbols` was passed, but input is not human-readable: {}",
        input_path.display()
    ),
    location: None,
    labels: Vec::new(),
    notes: Vec::new(),
});
```
In text mode keep the existing `eprintln!`. (Name the diagnostics vec to match Phase 4's variable.)

- [ ] **Step 5: Route the GLSL stage-mismatch advisory (core.rs ~647, in write_output glsl arm)**

Add `warnings: &mut Vec<crate::output::Diagnostic>` as a `write_output` parameter. Update all `write_output` call sites in `run` to pass a `&mut Vec<Diagnostic>` (in json mode, pass the run-level diagnostics vec so the warning lands in JSON; in text mode, pass a throwaway `&mut Vec::new()` since the existing `eprintln!` still fires). In the glsl arm's stage-mismatch branch, keep the `eprintln!` for text mode AND, when in json mode, push a `Severity::Warning` diagnostic (message = the existing warning text) to `warnings`. Pass the format/is_json signal into `write_output` too (add a `format: OutputFormat` param or a `is_json: bool`), or gate on whether `warnings` points at the real sink — simplest is an explicit `is_json: bool` parameter.

(Yes, `write_output`'s signature is growing. A follow-up refactor to bundle `params`/`hooks`/`format`/`warnings` into a `WriteCtx<'_>` struct is worth doing, but is OUT OF SCOPE here — note it in the report. For this task, add the parameters directly.)

- [ ] **Step 6: Remove the `#[allow(dead_code)]` on `Severity::Warning`**

Now that `Warning` is constructed, delete its `#[allow(dead_code)]` in `output.rs`. `cargo clippy -- -D warnings` must stay clean.

- [ ] **Step 7: naga core doc-comment (naga/src/back/mod.rs)**

On `ZeroInitializeWorkgroupMemoryMode`, add a doc note (behavior clarification only, no code change):
```rust
/// How to zero-initialize workgroup memory.
///
/// Only the SPIR-V backend honors [`Native`]; the MSL, HLSL, and GLSL backends
/// have no native zero-init path and treat [`Native`] identically to [`Polyfill`]
/// (they emit zeroing code whenever the mode is not [`None`]).
///
/// [`Native`]: ZeroInitializeWorkgroupMemoryMode::Native
/// [`Polyfill`]: ZeroInitializeWorkgroupMemoryMode::Polyfill
/// [`None`]: ZeroInitializeWorkgroupMemoryMode::None
```
(Merge with any existing doc-comment; don't duplicate.)

- [ ] **Step 8: Run tests + clippy**

Run: `cargo test -p naga-cli && cargo clippy -p naga-cli --all-targets -- -D warnings && cargo build -p naga`
Expected: all pass; the 2 new tests green; text-mode tests unchanged; naga builds (doc-comment only).

- [ ] **Step 9: Commit**

```bash
git add naga-cli/src/core.rs naga-cli/src/output.rs naga/src/back/mod.rs naga-cli/tests/cli.rs
git commit -m "feat(naga-cli): surface advisory warnings as JSON diagnostics; reject bulk+json"
```

---

### Task 2: README + examples + CHANGELOG

**Files:**
- Create: `naga-cli/README.md`, `naga-cli/examples/triangle.wgsl`, `naga-cli/examples/options.json`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Create the example shader**

`naga-cli/examples/triangle.wgsl`:
```wgsl
// A minimal shader used by the naga-cli README examples.
@group(0) @binding(0) var<uniform> tint: vec4<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i) - 1);
    let y = f32(i32(i & 1u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return tint;
}
```

- [ ] **Step 2: Create the example config**

`naga-cli/examples/options.json` — a config exercising the flattened + backend keys (must deserialize against the current `Config`):
```json
{
  "spv_out": {
    "lang_version": [1, 3],
    "force_loop_bounding": false
  },
  "msl": {
    "lang_version": [2, 0]
  }
}
```
(Verify this deserializes: `naga naga-cli/examples/triangle.wgsl out.spv --config naga-cli/examples/options.json` must succeed. Adjust keys to the real `Config`/Options field names if any differ.)

- [ ] **Step 3: Write `naga-cli/README.md`**

Cover, in this order (keep it accurate to the actual flags — cross-check `--help`):
1. **What it is** — one paragraph: the CLI for the naga shader translator/validator.
2. **Install / build** — `cargo build -p naga-cli`; binary name `naga`.
3. **Basic usage** — input inferred by extension; output inferred by output-file extension; validation-only when no output. Examples:
   - `naga shader.wgsl` (validate)
   - `naga shader.wgsl out.spv` (WGSL → SPIR-V)
   - `naga in.spv out.wgsl` (SPIR-V → WGSL)
   - `cat shader.wgsl | naga --stdin-file-path shader.wgsl` (stdin)
4. **Options: flags vs config** — the two ways to set translation options; they are mutually exclusive. Show a flag example and the equivalent `--config`/`--config-json` example. Point at `--print-config-schema` for the full config shape.
5. **Structured output** — `--format json`: describe the `{success, diagnostics, reflection}` shape with a short real example (run it and paste). Note diagnostics carry `severity`/`message`/`location`/`labels`/`notes`; reflection carries entry points, resources, overrides.
6. **External tool hooks** — `--spirv-val`, `--spirv-opt`, `--dxc`: what each does, that the tool must be on `PATH`, and the DXIL output naming (`<hlsl-stem>.<entry>.dxil`).
7. **Notes / gotchas** (the carried doc items):
   - Coordinate space is controlled by the top-level `--keep-coordinate-space` / config `keep_coordinate_space`, not by `spv_out.flags`.
   - `--task-limits` and `--validate-mesh-output` fan out to all applicable backends; via config, set the per-backend `common` fields.
   - `--print-config-schema` omits a few config-only fields that can't be schema'd (SPIR-V capabilities set, GLSL `defines`); they are still settable via config JSON.
   - In `--format json`, every diagnostic is currently `error` or `warning`; `notes` may be empty for validation errors; SPIR-V parse errors have no source location.
   - `ZeroInitializeWorkgroupMemoryMode::Native` is honored only by the SPIR-V backend.

Keep code fences runnable. Use the example files from Steps 1-2.

- [ ] **Step 4: Point the crate at the README**

In `naga-cli/Cargo.toml` `[package]`, add `readme = "README.md"` if not present.

- [ ] **Step 5: CHANGELOG entry**

Under `## Unreleased` in `CHANGELOG.md`, add (in the appropriate subsection, e.g. a naga-cli / Added area consistent with the file's structure):
```markdown
- Rewrote `naga-cli`: migrated to `clap`, added `--config`/`--config-json` for full option coverage, `--print-config-schema`, `--format json` (structured diagnostics + reflection), and external tool hooks (`--spirv-val`, `--spirv-opt`, `--dxc`). The `.spirv` output extension now writes SPIR-V. By @Inner-Daemons in [#XXXX](https://github.com/gfx-rs/wgpu/pull/XXXX).
```
(Leave `#XXXX` as a placeholder — fill at PR time. Match the surrounding entry format/subsection.)

- [ ] **Step 6: Verify examples build/run**

Run:
```
cargo build -p naga-cli
./target/debug/naga naga-cli/examples/triangle.wgsl --config naga-cli/examples/options.json /tmp/tri.spv
./target/debug/naga naga-cli/examples/triangle.wgsl --format json
```
Expected: both succeed; json output parses and shows both entry points (`vs_main`, `fs_main`) + the `tint` resource.

- [ ] **Step 7: Commit**

```bash
git add naga-cli/README.md naga-cli/examples naga-cli/Cargo.toml CHANGELOG.md
git commit -m "docs(naga-cli): README, runnable examples, and changelog entry"
```

---

### Task 3: README example-verification tests

Keep the README's headline invocations honest with tests that run them against the committed example files.

**Files:**
- Modify: `naga-cli/tests/cli.rs`

- [ ] **Step 1: Write the tests**

Add to `naga-cli/tests/cli.rs` (path the example files relative to `CARGO_MANIFEST_DIR`):
```rust
fn example_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join(name)
}

#[test]
fn readme_config_file_example() {
    let dir = std::env::temp_dir().join("naga_cli_p6_readme_cfg");
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("tri.spv");
    let r = naga()
        .arg(example_path("triangle.wgsl"))
        .arg(&out)
        .arg("--config").arg(example_path("options.json"))
        .output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(&std::fs::read(&out).unwrap()[0..4], &[0x03, 0x02, 0x23, 0x07]);
}

#[test]
fn readme_json_reflection_example() {
    let r = naga().arg(example_path("triangle.wgsl")).args(["--format", "json"]).output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    let v: serde_json::Value = serde_json::from_slice(&r.stdout).unwrap();
    assert_eq!(v["success"], true);
    let names: Vec<&str> = v["reflection"]["entry_points"]
        .as_array().unwrap().iter()
        .map(|e| e["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"vs_main") && names.contains(&"fs_main"), "names: {names:?}");
    assert!(v["reflection"]["resources"].as_array().unwrap().iter()
        .any(|res| res["name"] == "tint"));
}
```

- [ ] **Step 2: Run tests + full suite + clippy**

Run: `cargo test -p naga-cli && cargo clippy -p naga-cli --all-targets -- -D warnings`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add naga-cli/tests/cli.rs
git commit -m "test(naga-cli): verify README example invocations"
```

---

## Self-Review

**Spec coverage (Phase 6 slice + carry-forward):**
- Examples & better docs → Task 2 (README + examples + CHANGELOG). ✓
- P4-M3 bulk+json silent → Task 1 (clear error). ✓
- P4-M2/M4 advisory warnings in json → Task 1 (routed to `Severity::Warning` diagnostics; dead-code allow removed). ✓
- P3 `Native` doc → Task 1 Step 7. ✓
- P2 config doc notes (coordinate-space, schema gaps, task-limits fan-out) → Task 2 README §7. ✓
- `.spirv`-writes-SPIR-V changelog note → Task 2 Step 5. ✓
- README stays honest → Task 3 (example-verification tests). ✓
- Explicitly OUT OF SCOPE (noted for a future cleanup, not this rewrite): splitting the now-large `core.rs` / bundling `write_output`'s parameters into a context struct; supporting `--format json` *inside* bulk mode (v1 rejects it); optional bitflags schema descriptions.

**Placeholder scan:** The only literal placeholder is the CHANGELOG `#XXXX` PR link — intentional and called out (no PR exists yet). Test snippets note "adjust if this naga version differs" for the `-g` warning trigger — a robustness caveat, not a gap.

**Type consistency:** `write_output` gains `warnings: &mut Vec<Diagnostic>` + an `is_json`/`format` signal; all call sites in `run` updated (Task 1 Step 5). The debug-symbols warning pushes to the same run-level diagnostics vec Phase 4 established. README examples use the committed `examples/` files that Task 3 tests exercise; config keys match the current `Config` (verified in Task 2 Step 2 / Step 6).

**Known verification points (not placeholders):** the exact trigger for the `-g` advisory (Task 1 Step 1 note); the current `run` return type (`anyhow::Result<bool>`) for the bulk early-return; the real `Config` field names for `examples/options.json`.
