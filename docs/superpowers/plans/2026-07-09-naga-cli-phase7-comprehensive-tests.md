# naga-cli — Phase 7 (Comprehensive Test Coverage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every CLI flag and every input/output language is covered by at least one test. Close the gaps left after Phases 1–6.

**Architecture:** Integration tests in `naga-cli/tests/cli.rs` (exec the built binary), plus a few unit tests in `naga-cli/src/output.rs` for the diagnostic converters. Prefer table-driven tests for the I/O-language matrix so completeness is visible at a glance. No production-code changes (test-only), except where a test reveals a real bug (then fix + note).

**Tech Stack:** Rust, `std::process::Command`, serde_json, `which` (already dev-usable).

## Global Constraints

- Test-only. If a test surfaces a real behavior bug, STOP, report it, and fix separately — don't weaken the test.
- Tests must pass on machines WITHOUT external tools: any test needing `dxc`/`spirv-opt`/`spirv-val`/`spirv-cross` guards with `tool_on_path(..)` and skips (prints a skip line) when absent.
- Assertions must check real behavior (output signature, exit code + message, byte content), not just "didn't panic". Where a flag's effect isn't externally observable with a simple shader, the minimum bar is: the flag parses, is accepted, and a compile/validate that exercises its code path succeeds — and say so in a comment.
- Keep test shaders minimal. Reuse small helpers (`naga()`, `tool_on_path`, `example_path`, a `write_shader(dir,name,src)` helper).
- Every item in the **Coverage Checklist** below must map to ≥1 test by the end. The final review verifies the checklist is fully ticked.

## Coverage Checklist (the contract — final review verifies every line has a test)

**Input languages (parser):** wgsl · glsl · spv · bin
**Output forms (writer, by extension):** txt (IR) · bin (IR) · metal · spv · spirv · vert · frag · comp · dot · hlsl · wgsl
**Flags:**
`--validate` · `--index-bounds-check-policy` · `--buffer-bounds-check-policy` · `--image-load-bounds-check-policy` · `--block-ctx-dir` · `--entry-point` · `--profile` · `--shader-model` · `--spirv-version` · `--shader-stage` · `--input-kind` · `--metal-version` · `--keep-coordinate-space` · `--dot-cfg-only` · `--stdin-file-path` · `-g/--generate-debug-symbols` · `--compact` · `--before-compaction` · `--bulk-validate` · `--override` · `-D/--defines` · `--capabilities` · `--task-limits` · `--validate-mesh-output` · `--config` · `--config-json` · `--print-config-schema` · `--format` · `--spirv-val` · `--spirv-opt` · `--dxc` · `--fake-missing-bindings` · `--force-loop-bounding` · `--ray-query-initialization-tracking` · `--zero-initialize-workgroup-memory` · `--version`

**Already covered by Phases 1–6 (verify still present; do not duplicate):** `--stdin-file-path`, `-g`, `--compact` (compose), `--config`, `--config-json`, `--print-config-schema`, `--format`, `--spirv-val`, `--spirv-opt`, `--dxc`, `--force-loop-bounding`, `--zero-initialize-workgroup-memory`, `--validate-mesh-output`, `--shader-stage` (glsl input), `--spirv-version` (config-parity), spv output, wgsl input, glsl input, bin (produced), `--version` (via `help_matches_snapshot`? — add an explicit `--version` test).

---

### Task 1: I/O language matrix (table-driven)

Cover every input parser and every output writer.

**Files:** Modify `naga-cli/tests/cli.rs`.

- [ ] **Step 1: Add a shared helper + the output-matrix test**

Add near the top of `tests/cli.rs`:
```rust
fn write_tmp(dir: &str, name: &str, contents: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(dir);
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, contents).unwrap();
    p
}

/// A small WGSL module with a vertex + fragment entry and a bound resource,
/// broad enough to emit to every backend.
const TRIANGLE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> tint: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(i), 0.0, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return tint; }
"#;
```

Then the output matrix test — WGSL → every output form, asserting a per-form signature:
```rust
#[test]
fn output_language_matrix_from_wgsl() {
    let src = write_tmp("naga_cli_p7_out", "in.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap().to_path_buf();
    // (output filename, needs --shader-stage/entry?, signature predicate on the bytes/string)
    struct Case { file: &'static str, args: &'static [&'static str], check: fn(&[u8]) -> bool }
    let cases: &[Case] = &[
        Case { file: "out.txt",   args: &[], check: |b| String::from_utf8_lossy(b).contains("(") }, // IR debug dump, non-empty
        Case { file: "out.bin",   args: &[], check: |b| !b.is_empty() },
        Case { file: "out.metal", args: &[], check: |b| { let s=String::from_utf8_lossy(b); s.contains("metal") || s.contains("[[") } },
        Case { file: "out.spv",   args: &[], check: |b| b[0..4]==[0x03,0x02,0x23,0x07] },
        Case { file: "out.spirv", args: &[], check: |b| b[0..4]==[0x03,0x02,0x23,0x07] },
        Case { file: "out.dot",   args: &[], check: |b| String::from_utf8_lossy(b).contains("digraph") },
        Case { file: "out.hlsl",  args: &[], check: |b| !b.is_empty() },
        Case { file: "out.wgsl",  args: &[], check: |b| String::from_utf8_lossy(b).contains("fn ") },
        // GLSL outputs need a single stage + matching entry point:
        Case { file: "out.vert",  args: &["--entry-point","vs_main"], check: |b| String::from_utf8_lossy(b).contains("#version") },
        Case { file: "out.frag",  args: &["--entry-point","fs_main"], check: |b| String::from_utf8_lossy(b).contains("#version") },
    ];
    for c in cases {
        let out = dir.join(c.file);
        let r = naga().arg(&src).arg(&out).args(c.args).output().unwrap();
        assert!(r.status.success(), "{}: {}", c.file, String::from_utf8_lossy(&r.stderr));
        let bytes = std::fs::read(&out).unwrap();
        assert!((c.check)(&bytes), "{} signature mismatch", c.file);
    }
}
```
(Add a `comp` case with a compute-only shader if `fs_main`/`vs_main` don't map to `.comp`; a separate tiny test `compute_wgsl_to_comp_glsl` using `@compute @workgroup_size(1) fn main(){}` → `out.comp` asserting `#version`. Confirm the `.txt` IR dump's actual content and pick a robust substring — inspect one real dump.)

- [ ] **Step 2: Input-parser matrix test**

```rust
#[test]
fn input_language_matrix() {
    let dir = std::env::temp_dir().join("naga_cli_p7_in");
    std::fs::create_dir_all(&dir).unwrap();

    // 1. WGSL input (validate).
    let wgsl = write_tmp("naga_cli_p7_in", "a.wgsl", "@compute @workgroup_size(1) fn main() {}");
    assert!(naga().arg(&wgsl).output().unwrap().status.success());

    // 2. GLSL input (fragment) — needs a stage; use .frag extension convention.
    let glsl = write_tmp("naga_cli_p7_in", "a.frag", "#version 450\nlayout(location=0) out vec4 c;\nvoid main(){ c=vec4(1.0); }");
    let r = naga().args(["--input-kind","glsl","--shader-stage","frag"]).arg(&glsl).output().unwrap();
    assert!(r.status.success(), "glsl: {}", String::from_utf8_lossy(&r.stderr));

    // 3. SPIR-V input — produce via wgsl->spv, then read back.
    let spv = dir.join("a.spv");
    assert!(naga().arg(&wgsl).arg(&spv).output().unwrap().status.success());
    let r = naga().arg(&spv).output().unwrap(); // spv input, validate
    assert!(r.status.success(), "spv-in: {}", String::from_utf8_lossy(&r.stderr));

    // 4. Bincode IR input — produce via wgsl->bin, then read back.
    let bin = dir.join("a.bin");
    assert!(naga().arg(&wgsl).arg(&bin).output().unwrap().status.success());
    let r = naga().arg(&bin).output().unwrap(); // bin input, validate
    assert!(r.status.success(), "bin-in: {}", String::from_utf8_lossy(&r.stderr));
}
```

- [ ] **Step 3: Cross input→output pairs (exercise non-wgsl parsers into writers)**

```rust
#[test]
fn cross_language_conversions() {
    let dir = std::env::temp_dir().join("naga_cli_p7_cross");
    std::fs::create_dir_all(&dir).unwrap();
    let wgsl = write_tmp("naga_cli_p7_cross", "a.wgsl", "@compute @workgroup_size(1) fn main() {}");

    // spv -> wgsl
    let spv = dir.join("a.spv");
    assert!(naga().arg(&wgsl).arg(&spv).output().unwrap().status.success());
    let out_wgsl = dir.join("from_spv.wgsl");
    let r = naga().arg(&spv).arg(&out_wgsl).output().unwrap();
    assert!(r.status.success(), "spv->wgsl: {}", String::from_utf8_lossy(&r.stderr));
    assert!(std::fs::read_to_string(&out_wgsl).unwrap().contains("fn "));

    // bin -> txt
    let bin = dir.join("a.bin");
    assert!(naga().arg(&wgsl).arg(&bin).output().unwrap().status.success());
    let out_txt = dir.join("from_bin.txt");
    assert!(naga().arg(&bin).arg(&out_txt).output().unwrap().status.success());

    // glsl -> spv
    let glsl = write_tmp("naga_cli_p7_cross", "a.frag", "#version 450\nlayout(location=0) out vec4 c;\nvoid main(){ c=vec4(1.0); }");
    let out_spv = dir.join("from_glsl.spv");
    let r = naga().args(["--input-kind","glsl","--shader-stage","frag"]).arg(&glsl).arg(&out_spv).output().unwrap();
    assert!(r.status.success(), "glsl->spv: {}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(&std::fs::read(&out_spv).unwrap()[0..4], &[0x03,0x02,0x23,0x07]);
}
```

- [ ] **Step 4: Run + commit**

Run: `cargo test -p naga-cli --test cli`. Fix any real failures (report if a bug). Commit:
```bash
git add naga-cli/tests/cli.rs
git commit -m "test(naga-cli): input/output language coverage matrix"
```

---

### Task 2: Translation-option flags with observable effects

One test per flag; assert the flag's effect where observable, else assert accepted+success (with a comment).

**Files:** Modify `naga-cli/tests/cli.rs`.

- [ ] **Step 1: Effect-observable flags**

Add tests (group as you like; each must assert something real):
- `override_value_applied`: `--override tint_scale=2.0`-style — pick an `override x: f32 = 1.0;` in WGSL used in output; compile to wgsl/spv with `--override x=5.0`; assert the emitted WGSL/text reflects the overridden value (or that output differs from the un-overridden compile). If override substitution isn't visible in the chosen output, assert the two outputs (default vs overridden) DIFFER.
- `capabilities_restrict_rejects`: a shader needing a capability (e.g. `enable f16;` + f16 use) compiled with `--capabilities none` → non-zero/validation error; with default (all) → success. (Confirm f16 is capability-gated on this naga; else pick another gated feature.)
- `keep_coordinate_space_changes_output`: WGSL with a `@builtin(position)` vertex → `.spv` with and without `--keep-coordinate-space`; assert the two outputs DIFFER (coordinate flip toggles).
- `dot_cfg_only_changes_output`: `.dot` output with and without `--dot-cfg-only`; assert they DIFFER (cfg-only is smaller/different).
- `before_compaction_writes_file`: `--compact --before-compaction pre.txt in.wgsl out.spv`; assert `pre.txt` exists and is non-empty.
- `entry_point_selects`: multi-entry WGSL → `.spv` with `--entry-point vs_main`; assert success (pipeline entry selected). (Effect on spv is hard to assert cheaply; at least assert success + that an unknown `--entry-point bogus` errors.)
- `profile_sets_glsl_version`: WGSL fragment → `.frag` with `--profile es300` (or a supported value from `--help`); assert output contains the matching `#version 300 es` (or the right string for the chosen profile).
- `metal_version_in_output`: WGSL → `.metal` with `--metal-version 2.0`; assert success (and, if the version appears in a comment/header, assert it; else accepted+success).
- `shader_model_affects_hlsl`: WGSL → `.hlsl` with `--shader-model 60`; assert success; combined with `--dxc` (skip if dxc absent) profile uses 6_0.
- `defines_affect_glsl`: GLSL fragment using `#ifdef FOO` → with `-D FOO=1` compiles one way, without differs/fails. Assert the define changes the result.
- `bounds_check_policies_accepted`: `--index-bounds-check-policy restrict --buffer-bounds-check-policy read-zero-skip-write --image-load-bounds-check-policy unchecked` on a shader with indexing → success (and optionally output differs from `unchecked`). At minimum: all three flags parse + compile succeeds.
- `input_kind_overrides_extension`: name a WGSL file `weird.txt` and pass `--input-kind wgsl` → validates (extension override works).
- `block_ctx_dir_accepted`: spv INPUT with `--block-ctx-dir <dir>` → success and (if it dumps) a file appears in dir; else accepted+success. (block_ctx_dir affects the spv FRONTEND.)

- [ ] **Step 2: Run + commit**

Run tests; fix real failures (report bugs). Commit:
```bash
git add naga-cli/tests/cli.rs
git commit -m "test(naga-cli): per-flag coverage for translation options"
```

---

### Task 3: Remaining flags, modes, and `--version`

**Files:** Modify `naga-cli/tests/cli.rs`.

- [ ] **Step 1: Modes + acceptance flags**

- `version_flag_prints_version`: `--version` → success, stdout contains the crate version (e.g. matches `env!("CARGO_PKG_VERSION")` — read it in the test).
- `validate_flag_bitmask`: `--validate 0` disables validation (a shader that would fail validation still... careful — pick a case where `--validate 0` changes outcome, e.g. a validation-only run of a shader that fails validation: with default it exits non-zero, with `--validate 0` it may succeed. Confirm behavior; at minimum assert `--validate 0` and `--validate <all-bits>` both parse and run).
- `bulk_validate_success`: two valid shaders `--bulk-validate a.wgsl b.wgsl` → success.
- `bulk_validate_reports_invalid`: one valid + one invalid → non-zero exit, stderr names the invalid file.
- `task_limits_accepted`: `--task-limits 8,8` on a normal shader → success (flag parses + reaches the option).
- `fake_missing_bindings_accepted`: `--fake-missing-bindings false` → success (flag plumbs to CommonBackendOptions).
- `ray_query_initialization_tracking_accepted`: `--ray-query-initialization-tracking false` → success.
- (These three "accepted" tests document that the flag reaches the backend option; observable codegen effect needs feature-specific shaders — out of scope, note it.)

- [ ] **Step 2: Run + commit**

```bash
git add naga-cli/tests/cli.rs
git commit -m "test(naga-cli): modes, bulk-validate, version, remaining option flags"
```

---

### Task 4: Diagnostic converter units + remaining gaps

**Files:** Modify `naga-cli/src/output.rs`, `naga-cli/tests/cli.rs`.

- [ ] **Step 1: GLSL + SPIR-V converter unit tests (`output.rs`)**

Add to `output.rs` `#[cfg(test)]`:
- `glsl_errors_become_diagnostics`: parse an INVALID GLSL fragment via `naga::front::glsl::Frontend`, take the `ParseErrors`, call `glsl_parse_errors_to_diagnostics(&errs, src)`; assert ≥1 diagnostic, severity Error, non-empty message. (Use a definitely-invalid GLSL snippet; verify it errors on this naga.)
- `spv_error_becomes_diagnostic`: construct or provoke a `naga::front::spv::Error` (e.g. feed `parse_u8_slice` obviously-invalid bytes like `[0,0,0,0]`), call `spv_error_to_diagnostic(&err)`; assert severity Error, non-empty message, `location` is None.

- [ ] **Step 2: GLSL stage-mismatch warning in json (integration)**

`glsl_stage_mismatch_is_json_warning`: WGSL with a fragment entry, output to `.vert` (vertex extension) with `--entry-point fs_main --format json` → the emitted JSON `diagnostics` contains a `"warning"` whose message mentions "stage". (This exercises the write_output `warnings` plumbing added in Phase 6. If the exact scenario doesn't trigger the mismatch on this naga, adjust to whatever reliably produces the "does not match the shader stage" advisory; assert a warning diagnostic appears.)

- [ ] **Step 3: json reflection for a resource-heavy shader (optional depth)**

`json_reflection_reports_overrides_and_resources`: WGSL with an `override`, a uniform, and a storage buffer → `--format json`; assert reflection lists the override (by name), and both resources (by group/binding). Strengthens reflection coverage beyond entry points.

- [ ] **Step 4: Run everything + clippy + commit**

Run:
```
cargo test -p naga-cli
cargo test -p naga --features serialize,deserialize,schemars,spv-in,spv-out,msl-out,hlsl-out,glsl-out,dot-out,wgsl-in,glsl-in
cargo clippy -p naga-cli --all-targets -- -D warnings
```
Expected: all pass. Commit:
```bash
git add naga-cli/src/output.rs naga-cli/tests/cli.rs
git commit -m "test(naga-cli): diagnostic converter units, glsl-mismatch json warning, reflection depth"
```

---

## Self-Review

**Coverage checklist → tests (fill during implementation; final review confirms each has ≥1 test):**
- Input parsers: wgsl/glsl/spv/bin → Task 1 Step 2 + Step 3.
- Output writers: txt/bin/metal/spv/spirv/vert/frag/comp/dot/hlsl/wgsl → Task 1 Step 1 (+ compute→comp note).
- Flags: mapped across Tasks 2 (translation options), 3 (modes/version/acceptance), and pre-existing Phase 1–6 tests (config/format/hooks/stdin/-g/compact/zero-init/force-loop-bounding/validate-mesh-output/shader-stage/spirv-version).
- Converters: wgsl (existing) + validation (existing) + glsl + spv (Task 4).

**Placeholder scan:** The "accepted+success" bar for `--fake-missing-bindings`/`--ray-query-initialization-tracking`/`--task-limits`/`--metal-version` is an explicit, documented minimum (observable codegen effect needs feature-specific shaders — out of scope), not a gap. Test snippets carry "adjust if this naga version differs" caveats for version-sensitive assertions (profile string, capability gating, glsl invalidity) — real robustness notes.

**Anti-tautology:** every test asserts a signature/effect/exit+message, never bare success where an effect is observable (override diff, capabilities rejection, keep-coordinate-space diff, dot-cfg-only diff, profile string, bulk-invalid naming).

**Out of scope (noted):** exhaustive per-backend output of every naga language *feature* (that's naga's snapshot suite's job — `naga/tests`); observable codegen effects of flags requiring exotic shaders (ray queries, mesh/task) — covered at the accepted+plumbed level here.
