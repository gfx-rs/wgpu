# naga-cli Rewrite — Phase 3 (CommonBackendOptions + Auto-Flags) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate the backend `Options` structs by extracting a shared `CommonBackendOptions`, unify `zero_initialize_workgroup_memory` to a single enum type across all backends, and auto-expose the shared fields as clap flags in the CLI (flattened once, collision-free).

**Architecture:** naga core: a new `CommonBackendOptions` struct (5 fields) in `naga::back`, embedded as `pub common: CommonBackendOptions` in the spv/msl/hlsl `Options`. `ZeroInitializeWorkgroupMemoryMode` moves to `naga::back` and becomes the type of `zero_initialize_workgroup_memory` on all four backends (bool→enum). A new off-by-default `clap` feature derives `clap::Args` on `CommonBackendOptions` and `clap::ValueEnum` on the mode enum. naga-cli: flattens `CommonBackendOptions` into its `Args` once, adds `--zero-initialize-workgroup-memory`, and applies them to the selected backends; the JSON `Config` gains the nested `common` (kept flat via `serde(flatten)`).

**Tech Stack:** Rust, clap v4, serde, schemars, naga.

## Global Constraints

- naga core MSRV `1.87`; naga-cli builds against it. clap already a workspace dep (v4).
- **Behavior-preserving for translation output.** The naga snapshot test suite (`cargo test -p naga`, comparing against `naga/tests/out`) is the guardrail: it MUST pass unchanged after Tasks 1–2. Any snapshot diff means a behavior regression — stop and investigate, do not regenerate snapshots to paper over it.
- Additive/mechanical field relocation only. No change to WHICH options exist or their effective defaults. `CommonBackendOptions::default()` = `{ fake_missing_bindings: true, force_loop_bounding: true, ray_query_initialization_tracking: true, task_dispatch_limits: None, mesh_shader_primitive_indices_clamp: true }` — matching the identical current defaults in spv/msl/hlsl.
- `zero_initialize_workgroup_memory` unification is semantics-preserving: current `bool` `true` → `ZeroInitializeWorkgroupMemoryMode::Polyfill`, `false` → `None`. In msl/hlsl/glsl codegen, "emit zeroing" happens iff the mode is `!= None` (both `Native` and `Polyfill` zero; those backends have no native path so they treat both as polyfill-style zeroing — matching the old `true` behavior).
- clap derives strictly behind the new naga `clap` feature (`#[cfg_attr(feature = "clap", derive(clap::Args))]`), OFF by default; naga-cli enables it.
- Downstream (`wgpu-hal`, `wgpu-core`, naga tests/benches) construction sites must be updated for the moved fields; the compiler enumerates them. `wgpu-hal` and `wgpu-core` must build after each task.

## File Structure

- `naga/src/back/mod.rs` — NEW home of `ZeroInitializeWorkgroupMemoryMode` and `CommonBackendOptions`.
- `naga/src/back/spv/mod.rs` — re-export the moved enum for compat; `Options` embeds `common`, keeps `emit_int_div_checks` + `zero_initialize_workgroup_memory` (now enum, already was).
- `naga/src/back/{msl,hlsl,glsl}/mod.rs` — `zero_initialize_workgroup_memory` bool→enum; msl/hlsl embed `common`.
- `naga/src/back/{spv,msl,hlsl,glsl}/writer.rs` — update field accesses + zero-init codegen.
- `wgpu-hal/src/{vulkan/adapter.rs, metal/device.rs, dx12/device.rs, gles/device.rs}` — update construction.
- `wgpu-core`, `naga/tests`, benches — update any construction/access the compiler flags.
- `naga-cli/src/{cli.rs, params.rs, config.rs, main.rs}` — flatten + apply + config nesting.

---

### Task 1: Unify + relocate `ZeroInitializeWorkgroupMemoryMode`

Move the enum to `naga::back`, re-export from spv, and change msl/hlsl/glsl's `zero_initialize_workgroup_memory` from `bool` to the enum, preserving behavior.

**Files:**
- `naga/src/back/mod.rs`, `naga/src/back/spv/mod.rs`
- `naga/src/back/{msl,hlsl,glsl}/mod.rs` + their `writer.rs`
- `wgpu-hal/src/{metal/device.rs, dx12/device.rs, gles/device.rs}`

**Interfaces:**
- Produces: `naga::back::ZeroInitializeWorkgroupMemoryMode` (moved; spv re-exports it as `naga::back::spv::ZeroInitializeWorkgroupMemoryMode` for compat). msl/hlsl/glsl `Options.zero_initialize_workgroup_memory: ZeroInitializeWorkgroupMemoryMode`.

- [ ] **Step 1: Move the enum to `naga::back`**

Cut the `ZeroInitializeWorkgroupMemoryMode` enum definition (naga/src/back/spv/mod.rs:1061, with its serde/schemars/derive attrs from Phase 2) and paste into `naga/src/back/mod.rs` (near the top-level backend types). In `naga/src/back/spv/mod.rs`, add a re-export so existing paths keep working:
```rust
pub use crate::back::ZeroInitializeWorkgroupMemoryMode;
```
Update spv's own usages if they referenced `super::ZeroInitializeWorkgroupMemoryMode` (writer.rs:1726, 3573) — they resolve via the re-export or `crate::back::`.

- [ ] **Step 2: Confirm naga still builds (spv unchanged behavior)**

Run: `cargo build -p naga --features spv-out,serialize,deserialize,schemars`
Expected: builds. `cargo test -p naga spv` snapshot tests unchanged.

- [ ] **Step 3: Change msl/hlsl/glsl field type bool→enum**

In each of `naga/src/back/{msl,hlsl,glsl}/mod.rs`:
- Change `pub zero_initialize_workgroup_memory: bool,` → `pub zero_initialize_workgroup_memory: crate::back::ZeroInitializeWorkgroupMemoryMode,`.
- In the manual `Default` impl, change `zero_initialize_workgroup_memory: true,` → `zero_initialize_workgroup_memory: crate::back::ZeroInitializeWorkgroupMemoryMode::Polyfill,`.

- [ ] **Step 4: Update the codegen read sites**

- `naga/src/back/msl/writer.rs:8442` — the expression `options.zero_initialize_workgroup_memory` (a bool) becomes a mode comparison. Find its use (likely `if options.zero_initialize_workgroup_memory { ... }`) and change to `if options.zero_initialize_workgroup_memory != crate::back::ZeroInitializeWorkgroupMemoryMode::None { ... }`.
- `naga/src/back/hlsl/writer.rs:2002` — same transformation.
- `naga/src/back/glsl/writer.rs:1342` — same transformation.
(If any site binds the bool to a variable or passes it along, adjust to the `!= None` boolean. The behavior: zero iff mode is not `None`.)

- [ ] **Step 5: Update wgpu-hal construction sites**

- `wgpu-hal/src/metal/device.rs` (~191): wherever `zero_initialize_workgroup_memory: <bool-expr>` is set, map the bool to the enum: `if <bool-expr> { ZeroInitializeWorkgroupMemoryMode::Polyfill } else { ZeroInitializeWorkgroupMemoryMode::None }`. Import `naga::back::ZeroInitializeWorkgroupMemoryMode`.
- `wgpu-hal/src/dx12/device.rs` (~1481): same.
- `wgpu-hal/src/gles/device.rs` (~1371): same.
(These backends previously passed a bool; preserve the exact truth value via the map above.)

- [ ] **Step 6: Verify behavior-preserving**

Run:
```
cargo test -p naga
cargo build -p wgpu-hal
```
Expected: naga snapshot tests PASS UNCHANGED (no diffs in `naga/tests/out`); wgpu-hal builds. If any snapshot changed, a codegen mapping is wrong — fix so output is identical.

- [ ] **Step 7: Commit**

```bash
git add naga/src/back wgpu-hal/src
git commit -m "refactor(naga): unify zero_initialize_workgroup_memory to a shared enum"
```

---

### Task 2: Extract `CommonBackendOptions`

Move the 5 shared fields into a `CommonBackendOptions` struct embedded in spv/msl/hlsl as `pub common`.

**Files:**
- `naga/src/back/mod.rs` (define struct), `naga/src/back/{spv,msl,hlsl}/mod.rs` (embed), their `writer.rs` (accesses)
- `wgpu-hal/src/{vulkan/adapter.rs, metal/device.rs, dx12/device.rs}`, `wgpu-core`, `naga/tests`, benches

**Interfaces:**
- Produces: `naga::back::CommonBackendOptions { fake_missing_bindings: bool, force_loop_bounding: bool, ray_query_initialization_tracking: bool, task_dispatch_limits: Option<TaskDispatchLimits>, mesh_shader_primitive_indices_clamp: bool }` with the specified `Default`. spv/msl/hlsl `Options` gain `pub common: CommonBackendOptions` (via `#[serde(flatten)]`) and lose the 5 standalone fields.

- [ ] **Step 1: Define `CommonBackendOptions` in `naga::back`**

In `naga/src/back/mod.rs`:
```rust
/// Backend options shared by the SPIR-V, MSL, and HLSL backends.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct CommonBackendOptions {
    /// Bind fake resources for shaders that reference missing bindings.
    pub fake_missing_bindings: bool,
    /// Add a bound to all loops that might not terminate.
    pub force_loop_bounding: bool,
    /// Track ray query initialization.
    pub ray_query_initialization_tracking: bool,
    /// Limits on mesh-shader task dispatch size.
    pub task_dispatch_limits: Option<TaskDispatchLimits>,
    /// Clamp mesh shader primitive indices.
    pub mesh_shader_primitive_indices_clamp: bool,
}

impl Default for CommonBackendOptions {
    fn default() -> Self {
        Self {
            fake_missing_bindings: true,
            force_loop_bounding: true,
            ray_query_initialization_tracking: true,
            task_dispatch_limits: None,
            mesh_shader_primitive_indices_clamp: true,
        }
    }
}
```
(Copy the exact doc comments from the original fields where they existed.)

- [ ] **Step 2: Embed in spv/msl/hlsl `Options`, remove the 5 standalone fields**

In each of `naga/src/back/{spv,msl,hlsl}/mod.rs`:
- Remove the 5 fields (`fake_missing_bindings`, `force_loop_bounding`, `ray_query_initialization_tracking`, `task_dispatch_limits`, `mesh_shader_primitive_indices_clamp`).
- Add:
```rust
    #[cfg_attr(feature = "serialize", serde(flatten))]
    #[cfg_attr(feature = "deserialize", serde(flatten))]
    pub common: crate::back::CommonBackendOptions,
```
`serde(flatten)` keeps the JSON keys flat (e.g. `{"spv_out":{"force_loop_bounding":false}}`), preserving the Phase-2 config shape.
- In each manual `Default` impl, replace the 5 field initializers with `common: crate::back::CommonBackendOptions::default(),`.
- Leave `emit_int_div_checks` as-is on spv and msl (NOT extracted — hlsl lacks it).

Note on serde: `serde(flatten)` composes with the struct-level `serde(default)`. If `serde(flatten)` + `deny_unknown_fields` ever conflict, note that these naga Options do NOT use `deny_unknown_fields` (only the CLI `Config` does), so flatten is safe here. Verify the schemars derive still generates (flatten is supported by schemars; if a specific interaction fails, report it).

- [ ] **Step 3: Update codegen accesses (compiler-guided)**

Build and let the compiler point at every `options.<field>` that moved. Change each `options.fake_missing_bindings` → `options.common.fake_missing_bindings`, etc., across `naga/src/back/{spv,msl,hlsl}/writer.rs` and any `mod.rs` helpers. Run:
`cargo build -p naga --features spv-out,msl-out,hlsl-out,serialize,deserialize,schemars`
Fix every access error until it builds.

- [ ] **Step 4: Update downstream construction sites**

Build the workspace pieces and fix construction:
- `wgpu-hal/src/vulkan/adapter.rs` (~2826): the spv `Options { ... }` literal now sets `common: CommonBackendOptions { fake_missing_bindings, force_loop_bounding, ray_query_initialization_tracking, task_dispatch_limits, mesh_shader_primitive_indices_clamp }, ...` — group the 5 moved fields under `common`.
- `wgpu-hal/src/metal/device.rs` (~191) and `wgpu-hal/src/dx12/device.rs` (~1481): same grouping for msl/hlsl.
- Any `wgpu-core`, `naga/tests`, `naga/benches` construction the compiler flags.
Run: `cargo build -p wgpu-hal && cargo build -p wgpu-core`

- [ ] **Step 5: Verify behavior-preserving**

Run: `cargo test -p naga`
Expected: snapshot tests PASS UNCHANGED. If any `naga/tests/out` file differs, the extraction changed a default or access — fix until identical.

- [ ] **Step 6: Commit**

```bash
git add naga/src wgpu-hal/src wgpu-core naga/tests
git commit -m "refactor(naga): extract CommonBackendOptions shared by spv/msl/hlsl"
```

---

### Task 3: naga `clap` feature + derives

Add an optional `clap` dependency and feature, derive `clap::Args` on `CommonBackendOptions` and `clap::ValueEnum` on `ZeroInitializeWorkgroupMemoryMode`.

**Files:**
- `naga/Cargo.toml`, `naga/src/back/mod.rs`

**Interfaces:**
- Produces: with `--features clap`, `CommonBackendOptions: clap::Args` and `ZeroInitializeWorkgroupMemoryMode: clap::ValueEnum`.

- [ ] **Step 1: Cargo**

`naga/Cargo.toml`: `clap = { workspace = true, optional = true }` in deps; feature `clap = ["dep:clap"]`.

- [ ] **Step 2: Derive clap on the shared types**

On `ZeroInitializeWorkgroupMemoryMode` (naga/src/back/mod.rs) add:
```rust
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
```
(Variants `Native | Polyfill | None` become clap values `native | polyfill | none`.)

On `CommonBackendOptions` add `#[cfg_attr(feature = "clap", derive(clap::Args))]`, and per-field clap attributes gated with `cfg_attr(feature = "clap", ...)`:
- The 3 bools (`fake_missing_bindings`, `force_loop_bounding`, `ray_query_initialization_tracking`) — valued bools that stay disable-able (Phase-1 lesson): `#[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set, default_value_t = true))]`.
- `task_dispatch_limits` — skip (the CLI keeps its bespoke `--task-limits` X,Y parser): `#[cfg_attr(feature = "clap", arg(skip = None))]`.
- `mesh_shader_primitive_indices_clamp` — skip (CLI keeps bespoke `--validate-mesh-output`): `#[cfg_attr(feature = "clap", arg(skip = true))]`.

The `skip = <default>` values MUST equal `CommonBackendOptions::default()`'s values (None, true) so a flattened-with-no-flags value equals the struct default (behavior-preserving when applied).

- [ ] **Step 3: Verify**

Run: `cargo build -p naga --features clap` and `cargo build -p naga` (no features).
Expected: both build. (A quick inline test under `#[cfg(all(test, feature="clap"))]` using `clap::CommandFactory`/`Args` is optional; the real exercise is Task 4's CLI tests.)

- [ ] **Step 4: Commit**

```bash
git add naga/Cargo.toml naga/src/back/mod.rs
git commit -m "feat(naga): clap Args/ValueEnum derives for CommonBackendOptions behind clap feature"
```

---

### Task 4: CLI — flatten common flags + zero-init flag + config nesting

Flatten `CommonBackendOptions` into the CLI once, add `--zero-initialize-workgroup-memory`, apply to the backends, and update the `Config`.

**Files:**
- `naga-cli/Cargo.toml`, `naga-cli/src/{cli.rs, params.rs, config.rs, main.rs}`, `naga-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `naga::back::{CommonBackendOptions, ZeroInitializeWorkgroupMemoryMode}` with clap derives.

- [ ] **Step 1: Cargo**

`naga-cli/Cargo.toml`: add `"clap"` to the naga features list.

- [ ] **Step 2: Write the failing integration tests**

Add to `naga-cli/tests/cli.rs`:
```rust
#[test]
fn force_loop_bounding_flag_applies_to_spv() {
    // A shader whose SPIR-V differs when loop bounding is off vs on would be ideal,
    // but at minimum assert the flag parses, is accepted, and compiles successfully.
    let dir = std::env::temp_dir().join("naga_cli_p3_flb");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();
    let out = dir.join("s.spv");
    let r = naga().arg(&src).arg(&out).arg("--force-loop-bounding").arg("false").output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(&std::fs::read(&out).unwrap()[0..4], &[0x03, 0x02, 0x23, 0x07]);
}

#[test]
fn zero_init_flag_accepts_modes() {
    for mode in ["native", "polyfill", "none"] {
        let out = naga().arg("--help").output().unwrap(); // ensure binary exists
        assert!(out.status.success());
        // parse-only check via a compile:
        let dir = std::env::temp_dir().join("naga_cli_p3_zi");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("s.wgsl");
        std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
        let dst = dir.join(format!("s_{mode}.spv"));
        let r = naga().arg(&src).arg(&dst)
            .arg("--zero-initialize-workgroup-memory").arg(mode).output().unwrap();
        assert!(r.status.success(), "mode {mode} stderr: {}", String::from_utf8_lossy(&r.stderr));
    }
}

#[test]
fn config_nested_common_flat_json() {
    let dir = std::env::temp_dir().join("naga_cli_p3_cfg");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();
    let out = dir.join("s.spv");
    // serde(flatten) keeps common keys flat inside spv_out:
    let r = naga().arg(&src).arg(&out)
        .arg("--config-json").arg(r#"{"spv_out":{"force_loop_bounding":false}}"#)
        .output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p naga-cli --test cli force_loop_bounding_flag_applies_to_spv zero_init_flag_accepts_modes config_nested_common_flat_json`
Expected: FAIL — flags don't exist.

- [ ] **Step 4: Flatten + add the zero-init flag in `cli.rs`**

In `Args`:
```rust
/// Options shared across the SPIR-V, MSL, and HLSL backends.
#[command(flatten)]
pub common: naga::back::CommonBackendOptions,

/// How to zero-initialize workgroup memory (native | polyfill | none).
#[arg(long, value_enum, group = "options")]
pub zero_initialize_workgroup_memory: Option<naga::back::ZeroInitializeWorkgroupMemoryMode>,
```
The flattened `common` fields must be in the exclusion group. Since `#[command(flatten)]` cannot take `group = "options"` directly per-field, add the flattened args to the group via the struct-level ArgGroup: extend the existing `ArgGroup::new("options")` — clap includes flattened args in a parent group if you add `#[group(id = "options", ...)]`? If that is awkward, instead give the CommonBackendOptions clap derive fields an explicit `group = "options"` attribute in Task 3 (add `group = "options"` to each `arg(...)` there). Choose whichever clap supports cleanly; the requirement is: `--force-loop-bounding` etc. and `--zero-initialize-workgroup-memory` all conflict with `--config`/`--config-json`. Add an integration test asserting `--config-json {} --force-loop-bounding false` errors with "cannot be used with".

(Implementer note: the cleanest route is adding `#[cfg_attr(feature = "clap", arg(..., group = "options"))]` to the CommonBackendOptions fields in Task 3. If you discover that during Task 4, amend Task 3's derive — record it.)

- [ ] **Step 5: Apply to backends in `params.rs`**

In `build_parameters`, after constructing the per-backend Options, copy the flattened common into each backend that has it, and apply zero-init to all four:
```rust
params.spv_out.common = args.common.clone();
params.msl.common = args.common.clone();
params.hlsl.common = args.common.clone();
if let Some(mode) = args.zero_initialize_workgroup_memory {
    params.spv_out.zero_initialize_workgroup_memory = mode;
    params.msl.zero_initialize_workgroup_memory = mode;
    params.hlsl.zero_initialize_workgroup_memory = mode;
    params.glsl.zero_initialize_workgroup_memory = mode;
}
```
Then KEEP the existing bespoke `--task-limits` / `--validate-mesh-output` lines, but retarget them to the nested field: `params.spv_out.common.task_dispatch_limits = args.task_limits;`, `params.spv_out.common.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;` (and msl/hlsl). This must run AFTER the `params.<backend>.common = args.common.clone()` assignment so the bespoke flags win. Because `args.common`'s skipped `task_dispatch_limits`/`mesh_shader_primitive_indices_clamp` carry the struct defaults (None/true), copying then overriding is correct.

- [ ] **Step 6: Update `config.rs`**

The `Config`'s `spv_out`/`msl`/`hlsl` fields are `naga::back::*::Options`, which now contain `common` via `serde(flatten)` — so no structural change to `Config` is needed; `apply_config` already moves whole Options structs. Verify `apply_config` still compiles and the flattened JSON (`{"spv_out":{"force_loop_bounding":false}}`) deserializes. Confirm the `params.spv_out.zero_initialize_workgroup_memory` and `.common` are populated from config when present (they are, since the whole Options is replaced).

- [ ] **Step 7: Run tests + regenerate help snapshot**

Run: `cargo test -p naga-cli` (all pass incl. the 3 new). Regenerate the help snapshot (new flags): `cargo run -q -p naga-cli -- --help > naga-cli/tests/snapshots/help.txt`. Run `cargo clippy -p naga-cli --all-targets -- -D warnings`.

- [ ] **Step 8: Commit**

```bash
git add naga-cli
git commit -m "feat(naga-cli): flatten CommonBackendOptions flags and --zero-initialize-workgroup-memory"
```

---

## Self-Review

**Spec coverage (Phase 3 slice):**
- CommonBackendOptions extraction (dedup) → Task 2. ✓
- zero_init unified to shared enum → Task 1. ✓
- Auto-expose common fields as flags (flatten once, collision-free) → Tasks 3-4. ✓
- Common-only flag scope (per-backend unique fields stay config-only) → honored (only CommonBackendOptions flattened + zero-init). ✓
- NOT in this phase (deferred): `--format json` diagnostics/reflection (Phase 4); tool hooks (Phase 5); examples/docs + the 3 carried doc minors (Phase 6). `emit_int_div_checks` stays config-only (spv+msl; not common — hlsl lacks it).

**Placeholder scan:** No TBD/TODO. The clap-group-on-flattened-args mechanism (Task 4 Step 4) has an explicit "choose whichever clap supports; cleanest is group attrs in Task 3" resolution with a stop-and-record instruction — a real clap-API adaptation point, not a gap.

**Type/behavior consistency:**
- `CommonBackendOptions::default()` values (Task 2 Step 1) match the verified identical spv/msl/hlsl defaults (all true, task_dispatch_limits None) → extraction preserves defaults.
- clap `skip` defaults (Task 3 Step 2: `skip = None`, `skip = true`) match those struct defaults → flattening with no flags equals the default (no behavior change).
- zero_init map (Task 1): `true→Polyfill`, `false→None`, codegen zeros iff `!= None` → preserves the old bool behavior. Guarded by naga snapshot tests (Tasks 1, 2 Step 5).
- The bespoke `--task-limits`/`--validate-mesh-output` retarget to `.common.*` and run after the `common` copy (Task 4 Step 5) → Phase-1 flag behavior preserved.

**Guardrail emphasis:** Tasks 1 and 2 are only correct if `cargo test -p naga` snapshot output is byte-identical. Any diff = regression.
