# naga-cli Rewrite — Phase 2 (Config Vertical) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every naga frontend/backend option reachable from the CLI via a JSON config — `naga --config opts.json` / `--config-json '{...}'` — and expose the config's JSON Schema via `--print-config-schema`, by filling the remaining serde gaps in naga core and adding `schemars` derives.

**Architecture:** naga core: fill `Serialize`/`Deserialize` gaps on the Options structs that lack them (behind the existing `serialize`/`deserialize` features), and add `schemars::JsonSchema` derives behind a new `schemars` feature. naga-cli: a serde-deserializable `Config` struct mirrors the translation options; `--config`/`--config-json` (mutually exclusive with each other and with the per-option flags via a clap `ArgGroup`) deserialize into it and populate `Parameters`; `--print-config-schema` prints the `Config` schema.

**Tech Stack:** Rust, serde, serde_json, schemars, clap v4, naga.

## Global Constraints

- naga core MSRV: `rust-version = "1.87"` (naga/Cargo.toml:17). naga-cli builds against this naga, so all added deps must build on 1.87.
- serde gating idiom used throughout naga: `#[cfg_attr(feature = "serialize", derive(serde::Serialize))]` and `#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]`, with `#[cfg_attr(feature = "deserialize", serde(default))]` on option structs so partial JSON works. Match this exactly.
- schemars derives must be feature-gated the same way: `#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]`. The `schemars` feature is NEW and OFF by default.
- Derives are additive — do NOT change any field, type, name, or the ~12 downstream construction sites in wgpu-hal/wgpu-core. No behavior change to translation.
- `back::spv::Options<'a>` has a lifetime via `debug_info: Option<DebugInfo<'a>>`. `debug_info` is runtime-populated by the CLI (from `-g`), never a config field: it MUST be `#[serde(skip)]` and excluded from the schema. The CLI `Config` never carries debug info.
- JSON is the config dialect; `--config` (file) and `--config-json` (inline) are mutually exclusive with each other and with every per-option flag (enforced by a clap `ArgGroup`). If a config is given, passing any option flag is a hard parse error. I/O flags (input/output files, `--stdin-file-path`, `--format` when it exists) remain allowed.
- Confirmed already-serde-ready (do NOT re-derive): `back::msl::Options`, `back::hlsl::Options`, `back::glsl::Options`, `naga::proc::BoundsCheckPolicy` (index.rs:39-41), `BoundsCheckPolicies`, `Capabilities`, `ShaderStage`, `glsl::Version`, `hlsl::ShaderModel`, glsl `WriterFlags`.

---

## File Structure

- `Cargo.toml` (workspace) — add `schemars` and (naga dev-dep) rely on existing `serde_json`.
- `naga/Cargo.toml` — add optional `schemars` dep + `schemars` feature; add `serde_json` dev-dep.
- `naga/src/back/spv/mod.rs` — serde+schemars on `WriterFlags`, `ZeroInitializeWorkgroupMemoryMode`, `Options<'a>`.
- `naga/src/front/spv/mod.rs`, `naga/src/front/glsl/mod.rs`, `naga/src/front/wgsl/parse/mod.rs` — serde+schemars on each `Options`.
- `naga/src/back/dot/mod.rs` — serde+schemars on `Options`.
- Various: add `#[cfg_attr(feature="schemars", derive(schemars::JsonSchema))]` to the already-serde structs that the Config will reference.
- `naga-cli/Cargo.toml` — enable naga `schemars` feature; add `serde_json`.
- `naga-cli/src/config.rs` — NEW: the `Config` struct (serde + schemars) + `apply_config`.
- `naga-cli/src/cli.rs` — add `--config`, `--config-json`, `--print-config-schema`; ArgGroup exclusivity.
- `naga-cli/src/params.rs` / `main.rs` — route config into `Parameters`.
- `naga-cli/tests/cli.rs` — integration tests.

---

### Task 1: Core serde gaps — spv supporting types

`back::spv::Options` cannot derive serde until its two non-serde field types do. Add serde (and schemars, gated) to `WriterFlags` and `ZeroInitializeWorkgroupMemoryMode`.

**Files:**
- Modify: `naga/Cargo.toml` (add `serde_json` dev-dep)
- Modify: `naga/src/back/spv/mod.rs` (WriterFlags ~1000, ZeroInitializeWorkgroupMemoryMode ~1046)
- Test: `naga/src/back/spv/mod.rs` inline `#[cfg(test)]`

**Interfaces:**
- Produces: `WriterFlags` and `ZeroInitializeWorkgroupMemoryMode` implement `Serialize`/`Deserialize` under features `serialize`/`deserialize`.

- [ ] **Step 1: Add serde_json dev-dep to naga**

In `naga/Cargo.toml` `[dev-dependencies]`, add:
```toml
serde_json = { workspace = true }
```

- [ ] **Step 2: Write the failing test**

Add to the bottom of `naga/src/back/spv/mod.rs`:
```rust
#[cfg(all(test, feature = "serialize", feature = "deserialize"))]
mod serde_tests {
    use super::*;

    #[test]
    fn writer_flags_round_trip() {
        let flags = WriterFlags::DEBUG | WriterFlags::ADJUST_COORDINATE_SPACE;
        let json = serde_json::to_string(&flags).unwrap();
        let back: WriterFlags = serde_json::from_str(&json).unwrap();
        assert_eq!(flags, back);
    }

    #[test]
    fn zero_init_mode_round_trip() {
        for mode in [
            ZeroInitializeWorkgroupMemoryMode::Native,
            ZeroInitializeWorkgroupMemoryMode::Polyfill,
            ZeroInitializeWorkgroupMemoryMode::None,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let back: ZeroInitializeWorkgroupMemoryMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, back);
        }
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p naga --features serialize,deserialize back::spv::serde_tests`
Expected: FAIL to compile — `WriterFlags`/`ZeroInitializeWorkgroupMemoryMode` don't implement `Serialize`.

- [ ] **Step 4: Add the derives**

On the `WriterFlags` bitflags block (naga/src/back/spv/mod.rs:1000), match the glsl `WriterFlags` idiom (back/glsl/mod.rs:86-90):
```rust
bitflags::bitflags! {
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct WriterFlags: u32 {
```

On `ZeroInitializeWorkgroupMemoryMode` (naga/src/back/spv/mod.rs:1046):
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum ZeroInitializeWorkgroupMemoryMode {
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p naga --features serialize,deserialize back::spv::serde_tests`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add naga/Cargo.toml naga/src/back/spv/mod.rs
git commit -m "feat(naga): serde derives for spv WriterFlags and ZeroInitializeWorkgroupMemoryMode"
```

---

### Task 2: Core serde gaps — frontend Options + dot Options

Add serde derives to the four trivially-ready structs.

**Files:**
- Modify: `naga/src/front/wgsl/parse/mod.rs` (Options ~283), `naga/src/front/glsl/mod.rs` (Options ~51), `naga/src/front/spv/mod.rs` (Options ~409), `naga/src/back/dot/mod.rs` (Options ~24)
- Test: inline `#[cfg(test)]` in `naga/src/front/spv/mod.rs` (representative round-trip)

**Interfaces:**
- Produces: `front::wgsl::Options`, `front::glsl::Options`, `front::spv::Options`, `back::dot::Options` implement `Serialize`/`Deserialize` under the features, with `serde(default)`.

- [ ] **Step 1: Write the failing test**

Add to `naga/src/front/spv/mod.rs`:
```rust
#[cfg(all(test, feature = "serialize", feature = "deserialize"))]
mod serde_tests {
    use super::Options;

    #[test]
    fn options_round_trip_and_partial() {
        let opts = Options {
            adjust_coordinate_space: false,
            strict_capabilities: true,
            block_ctx_dump_prefix: Some("dump".into()),
        };
        let json = serde_json::to_string(&opts).unwrap();
        let back: Options = serde_json::from_str(&json).unwrap();
        assert_eq!(back.strict_capabilities, true);
        assert_eq!(back.block_ctx_dump_prefix.as_deref(), Some("dump"));

        // serde(default): empty object deserializes to defaults.
        let def: Options = serde_json::from_str("{}").unwrap();
        assert_eq!(def.strict_capabilities, false);
    }
}
```
(This requires `front::spv::Options` to derive `Default`; it currently derives only `Clone, Debug`. Add `Default` in Step 3 — the default of each field is the natural zero: `adjust_coordinate_space:false, strict_capabilities:false, block_ctx_dump_prefix:None`. If any consumer relies on a non-Default construction, that's fine — Default is additive.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p naga --features serialize,deserialize,spv-in front::spv::serde_tests`
Expected: FAIL to compile — `Options` lacks `Serialize`/`Default`.

- [ ] **Step 3: Add derives to all four structs**

`front::spv::Options` (naga/src/front/spv/mod.rs:409):
```rust
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default))]
pub struct Options {
```

`front::wgsl::Options` (naga/src/front/wgsl/parse/mod.rs:283) — note it already has a `Default`/`new` pattern elsewhere; add serde. If it does NOT derive `Default`, add it (fields: `parse_doc_comments: bool`, `capabilities: Capabilities` — both `Default`):
```rust
#[derive(Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default))]
pub struct Options {
```
(If `front::wgsl::Options` already derives `Default` or has a manual impl, do NOT add a duplicate — just add the three serde cfg_attr lines.)

`front::glsl::Options` (naga/src/front/glsl/mod.rs:51) — fields `stage: ShaderStage`, `defines: FastHashMap<String,String>`. `ShaderStage` has no obvious Default; do NOT add `serde(default)` to the struct if `stage` has no Default. Instead derive serde WITHOUT `serde(default)`:
```rust
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct Options {
```
(The CLI Config will always supply `stage` for glsl, or derive it from filename. Verify `ShaderStage: Default` — if it IS Default, add `serde(default)` too and default `defines` to empty. Check naga-types.)

`back::dot::Options` (naga/src/back/dot/mod.rs:24) — already derives `Default`:
```rust
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default))]
pub struct Options {
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p naga --features serialize,deserialize,spv-in front::spv::serde_tests`
Expected: PASS.
Also verify the crate compiles with each relevant feature: `cargo build -p naga --features serialize,deserialize,wgsl-in,glsl-in,spv-in,dot-out`
Expected: builds.

- [ ] **Step 5: Commit**

```bash
git add naga/src/front naga/src/back/dot/mod.rs
git commit -m "feat(naga): serde derives for frontend Options and dot Options"
```

---

### Task 3: Core serde — `back::spv::Options<'a>` with skipped debug_info

Derive serde on the lifetime-bearing spv backend Options, skipping `debug_info`.

**Files:**
- Modify: `naga/src/back/spv/mod.rs` (Options ~1055)
- Test: inline `#[cfg(test)]` in `naga/src/back/spv/mod.rs`

**Interfaces:**
- Produces: `back::spv::Options<'a>` implements `Serialize`/`Deserialize` under the features; `debug_info` is always `None` after deserialize.

- [ ] **Step 1: Write the failing test**

Add to the `serde_tests` module created in Task 1 (naga/src/back/spv/mod.rs):
```rust
#[test]
fn spv_options_round_trip_skips_debug_info() {
    let opts = Options {
        lang_version: (1, 5),
        force_loop_bounding: false,
        ..Options::default()
    };
    let json = serde_json::to_string(&opts).unwrap();
    let back: Options = serde_json::from_str(&json).unwrap();
    assert_eq!(back.lang_version, (1, 5));
    assert_eq!(back.force_loop_bounding, false);
    assert!(back.debug_info.is_none());

    // partial JSON works via serde(default)
    let def: Options = serde_json::from_str(r#"{"lang_version":[1,3]}"#).unwrap();
    assert_eq!(def.lang_version, (1, 3));
    assert!(def.debug_info.is_none());
}
```
(`Options` must derive `Default` for `..Options::default()`. Confirm it does; if not, this is a blocker — report it, because the CLI relies on `Options::default()` already in Phase 1's params.rs. It is constructed via `Default` in wgpu-hal, so it almost certainly derives or impls Default. If it has a manual `Default` impl, keep it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p naga --features serialize,deserialize,spv-out back::spv::serde_tests::spv_options`
Expected: FAIL to compile — `Options` lacks `Serialize`.

- [ ] **Step 3: Add the derives with skip + bound**

On `back::spv::Options<'a>` (naga/src/back/spv/mod.rs:1055):
```rust
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default, bound(deserialize = "")))]
pub struct Options<'a> {
```
And on the `debug_info` field:
```rust
    #[cfg_attr(feature = "serialize", serde(skip))]
    #[cfg_attr(feature = "deserialize", serde(skip))]
    pub debug_info: Option<DebugInfo<'a>>,
```

Notes for the implementer:
- `serde(skip)` makes `debug_info` use `Default::default()` (`None`) on deserialize and be omitted on serialize, so `DebugInfo<'a>` never needs serde impls.
- `bound(deserialize = "")` prevents serde's derive from adding a `DebugInfo<'a>: Deserialize<'de>` bound (which would fail) and stops it from requiring `'a: 'de`. If the empty bound alone is insufficient and the compiler complains about the unused/unconstrained lifetime, also add a serialize bound: `#[cfg_attr(feature = "serialize", serde(bound(serialize = "")))]`. If serde still can't derive `Serialize` because of the lifetime, apply `serde(skip)` as above (already applied) — serialize skips the only lifetime-bearing field, so `Serialize` should derive cleanly.
- If, after this, `Serialize`/`Deserialize` still will not derive due to the lifetime, STOP and report BLOCKED with the exact error — do NOT remove the lifetime or change `DebugInfo` (that is a larger core change out of scope here).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p naga --features serialize,deserialize,spv-out back::spv::serde_tests`
Expected: PASS (all serde_tests, including Task 1's two).
Verify no downstream break: `cargo build -p naga --features serialize,deserialize,spv-out`
Expected: builds.

- [ ] **Step 5: Commit**

```bash
git add naga/src/back/spv/mod.rs
git commit -m "feat(naga): serde derives for spv backend Options (debug_info skipped)"
```

---

### Task 4: Core `schemars` feature + JsonSchema derives

Add an optional `schemars` dependency and feature, and derive `JsonSchema` on every Options struct and its non-std field types that the CLI Config will reference.

**Files:**
- Modify: `Cargo.toml` (workspace `[workspace.dependencies]`)
- Modify: `naga/Cargo.toml` (optional dep + feature)
- Modify: the Options structs + supporting types across `naga/src/{front,back,proc,valid}` and `naga-types`.
- Test: inline `#[cfg(test)]` schema-generation check.

**Interfaces:**
- Produces: with `--features schemars`, `JsonSchema` is implemented for all Options structs and their transitive field types.

- [ ] **Step 1: Add schemars to the workspace**

In `Cargo.toml` `[workspace.dependencies]` (alphabetical, near `serde`):
```toml
schemars = "1"
```
(If `schemars = "1"` fails to build on MSRV 1.87 or its derive API differs from what Step 4 uses, fall back to `schemars = "0.8"` and adjust the derive path accordingly — `schemars::JsonSchema` is the derive in both, but attribute syntax differs. Note which version you used in your report.)

- [ ] **Step 2: Add optional dep + feature to naga**

In `naga/Cargo.toml` `[dependencies]`:
```toml
schemars = { workspace = true, optional = true }
```
In `[features]`:
```toml
schemars = ["dep:schemars", "naga-types/schemars"]
```
(Add a matching `schemars` feature to `naga-types/Cargo.toml` that enables `dep:schemars`, since `Capabilities`, `ShaderStage`, `glsl::Version` live there. Mirror the `serialize` feature wiring in naga-types.)

- [ ] **Step 3: Write the failing test**

Add to `naga/src/back/spv/mod.rs` a schemars test:
```rust
#[cfg(all(test, feature = "schemars"))]
mod schema_tests {
    use super::Options;

    #[test]
    fn spv_options_schema_generates() {
        let schema = schemars::schema_for!(Options);
        let json = serde_json::to_string(&schema).unwrap();
        // debug_info is skipped; lang_version must appear.
        assert!(json.contains("lang_version"), "schema: {json}");
        assert!(!json.contains("debug_info"), "debug_info must be skipped: {json}");
    }
}
```
(schemars 1.x uses `schema_for!`; if 0.8, the same macro exists. This test also needs `serde_json` — already a dev-dep from Task 1.)

- [ ] **Step 4: Run test to verify it fails, then add derives**

Run: `cargo test -p naga --features schemars,serialize,deserialize,spv-out back::spv::schema_tests`
Expected: FAIL to compile — `Options` doesn't implement `JsonSchema`.

Add `#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]` alongside the existing serde cfg_attr lines on EACH of:
- `back::spv::{Options, WriterFlags, ZeroInitializeWorkgroupMemoryMode}` — for `Options`, also skip debug_info in the schema: on the `debug_info` field add `#[cfg_attr(feature = "schemars", schemars(skip))]`.
- `back::{msl,hlsl,glsl,dot}::Options` and their non-std field types that lack JsonSchema (e.g. `hlsl::ShaderModel`, `glsl::Version`, `WriterFlags`, `BindTarget`, the binding-map value types). Derive `JsonSchema` on each transitively until `schema_for!` compiles. Work outward from compiler errors: each "does not implement JsonSchema" points at the next type to annotate.
- `front::{wgsl,glsl,spv}::Options`.
- `proc::{BoundsCheckPolicy, BoundsCheckPolicies}`.
- naga-types: `Capabilities`, `ShaderStage`, `glsl::Version`, `TaskDispatchLimits` (add the `schemars` cfg_attr + the naga-types `schemars` feature).

For bitflags types (`WriterFlags`, `Capabilities`), schemars may need a manual `JsonSchema` impl or the `schemars(with = "...")` attribute if the derive doesn't understand bitflags. If a bitflags type won't derive `JsonSchema`, implement it minimally as a string or integer schema:
```rust
#[cfg(feature = "schemars")]
impl schemars::JsonSchema for WriterFlags {
    fn schema_name() -> String { "WriterFlags".into() }
    fn json_schema(g: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        u32::json_schema(g)
    }
}
```
(Adjust to the actual schemars 1.x API — the method signatures differ between 0.8 and 1.x. Use whichever the chosen version requires; the intent is "serialize as its underlying integer.")

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p naga --features schemars,serialize,deserialize,spv-out,msl-out,hlsl-out,glsl-out,dot-out,wgsl-in,glsl-in,spv-in back::spv::schema_tests`
Expected: PASS.
Also: `cargo build -p naga --features schemars,serialize,deserialize` — builds; and `cargo build -p naga` (no features) — still builds (schemars off by default).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml naga/Cargo.toml naga-types/Cargo.toml naga/src naga-types/src
git commit -m "feat(naga): schemars feature and JsonSchema derives for Options"
```

---

### Task 5: CLI `Config` struct + `--config` / `--config-json`

Define a serde `Config` mirroring the translation options, add the two mutually-exclusive flags, and apply the config to `Parameters`.

**Files:**
- Modify: `naga-cli/Cargo.toml` (enable naga `schemars`; add `serde_json`)
- Create: `naga-cli/src/config.rs`
- Modify: `naga-cli/src/cli.rs` (flags + ArgGroup), `naga-cli/src/params.rs` (apply), `naga-cli/src/main.rs` (module)
- Test: `naga-cli/tests/cli.rs`

**Interfaces:**
- Consumes: the now-serde-deserializable naga Options.
- Produces:
  - `pub struct Config` (serde `Deserialize` + `Serialize` + schemars `JsonSchema`) with optional fields: `validate: Option<u8>`, `capabilities: Option<naga::valid::Capabilities>`, `bounds_check_policies: Option<naga::proc::BoundsCheckPolicies>`, `entry_point: Option<String>`, `keep_coordinate_space: Option<bool>`, `defines: Option<std::collections::BTreeMap<String,String>>`, `overrides: Option<std::collections::BTreeMap<String,f64>>`, `spv_in: Option<naga::front::spv::Options>`, `spv_out: Option<naga::back::spv::Options<'static>>`, `msl: Option<naga::back::msl::Options>`, `glsl_out: Option<naga::back::glsl::Options>`, `hlsl: Option<naga::back::hlsl::Options>`, `dot: Option<naga::back::dot::Options>`. All `#[serde(default)]`, `#[serde(deny_unknown_fields)]`.
  - `pub fn apply_config(config: Config, params: &mut crate::params::Parameters<'static>)`.

- [ ] **Step 1: Cargo wiring**

In `naga-cli/Cargo.toml`, add to the naga features list: `"schemars"`. In `[dependencies]` add:
```toml
serde_json = { workspace = true }
serde = { workspace = true, features = ["derive"] }
```

- [ ] **Step 2: Write the failing integration test**

Add to `naga-cli/tests/cli.rs`:
```rust
#[test]
fn config_json_matches_equivalent_flag() {
    let dir = std::env::temp_dir().join("naga_cli_p2_config");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();
    let out_flag = dir.join("flag.spv");
    let out_cfg = dir.join("cfg.spv");

    // Via flag:
    let a = naga().arg(&src).arg(&out_flag).arg("--spirv-version").arg("1.5").output().unwrap();
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));

    // Via config-json (same lang_version):
    let b = naga()
        .arg(&src).arg(&out_cfg)
        .arg("--config-json").arg(r#"{"spv_out":{"lang_version":[1,5]}}"#)
        .output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));

    assert_eq!(std::fs::read(&out_flag).unwrap(), std::fs::read(&out_cfg).unwrap());
}

#[test]
fn config_conflicts_with_option_flag() {
    let out = naga()
        .arg("in.wgsl")
        .arg("--config-json").arg("{}")
        .arg("--spirv-version").arg("1.5")
        .output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot be used with"),
        "expected clap conflict error, got: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn config_and_config_json_mutually_exclusive() {
    let out = naga()
        .arg("in.wgsl")
        .arg("--config").arg("x.json")
        .arg("--config-json").arg("{}")
        .output().unwrap();
    assert!(!out.status.success());
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p naga-cli --test cli config`
Expected: FAIL — flags don't exist / no exclusivity.

- [ ] **Step 4: Implement `config.rs`**

Create `naga-cli/src/config.rs` with the `Config` struct (fields per Interfaces above), deriving:
```rust
#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct Config { /* fields */ }
```
(Use a local naga-cli feature `schema` OR just always derive JsonSchema — since naga-cli always enables naga `schemars`, deriving unconditionally is simpler; do that: `#[derive(Debug, Default, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]` and add `schemars = { workspace = true }` to naga-cli deps.)

Implement `apply_config` to move each present field into `params` (mirroring build_parameters' targets): validation flags, capabilities, bounds policies, entry_point, keep_coordinate_space, defines, overrides, and the per-frontend/backend Options (`params.spv_in`, `params.spv_out`, `params.msl`, `params.glsl`, `params.hlsl`, `params.dot`). For `keep_coordinate_space`, re-apply the ADJUST_COORDINATE_SPACE writer-flag logic (as build_parameters does) so config and flags produce identical output.

- [ ] **Step 5: Add flags + ArgGroup to `cli.rs`**

Add to `Args`:
```rust
/// Read all translation options from a JSON config file (mutually exclusive with option flags).
#[arg(long, group = "config_input")]
pub config: Option<String>,

/// Read all translation options from an inline JSON string (mutually exclusive with option flags).
#[arg(long, group = "config_input", conflicts_with = "config")]
pub config_json: Option<String>,

/// Print the JSON Schema for the config document and exit.
#[arg(long)]
pub print_config_schema: bool,
```
Mark every per-option flag (validate, *_bounds_check_policy, profile, shader_model, spirv_version, metal_version, keep_coordinate_space, capabilities, task_limits, validate_mesh_output, overrides, defines, block_ctx_dir, dot_cfg_only, generate_debug_symbols, entry_point) with `conflicts_with = "config_input"`. Do this by adding a shared clap `ArgGroup` named `options` (via `#[command(group(ArgGroup::new("options").multiple(true).conflicts_with("config_input")))]` on the struct, and `group = "options"` on each option arg) so a single declaration enforces "config XOR any option flag". Keep I/O args (`files`, `stdin_file_path`) OUT of the group.

- [ ] **Step 6: Route config in `main.rs` / `params.rs`**

In `real_main` (main.rs): after `build_parameters`, if `args.config` or `args.config_json` is set, load the JSON (from file or the inline string), `serde_json::from_str` into `Config`, and `config::apply_config(config, &mut params)`. Handle parse errors with `.context(...)`. (Because config conflicts with option flags, `build_parameters` will have only defaults + I/O, so applying config on top is clean.)

- [ ] **Step 7: Run tests to verify pass**

Run: `cargo test -p naga-cli --test cli`
Expected: PASS including the 3 new config tests. Update the help snapshot (new flags): `cargo run -q -p naga-cli -- --help > naga-cli/tests/snapshots/help.txt`.

- [ ] **Step 8: Commit**

```bash
git add naga-cli/Cargo.toml naga-cli/src/config.rs naga-cli/src/cli.rs naga-cli/src/params.rs naga-cli/src/main.rs naga-cli/tests
git commit -m "feat(naga-cli): --config and --config-json for full option coverage"
```

---

### Task 6: CLI `--print-config-schema`

Print the `Config` JSON Schema and exit.

**Files:**
- Modify: `naga-cli/src/main.rs`
- Test: `naga-cli/tests/cli.rs`

- [ ] **Step 1: Write the failing test**

Add to `naga-cli/tests/cli.rs`:
```rust
#[test]
fn prints_config_schema() {
    let out = naga().arg("--print-config-schema").output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let schema = String::from_utf8(out.stdout).unwrap();
    // Valid JSON, and mentions a known option field.
    let v: serde_json::Value = serde_json::from_str(&schema).unwrap();
    assert!(v.is_object());
    assert!(schema.contains("spv_out") || schema.contains("lang_version"), "schema: {schema}");
}
```
(naga-cli tests can use serde_json — it is now a dep; if tests/ can't see it as a dev-dep, add `serde_json` under `[dev-dependencies]` too, or reference the crate dep.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p naga-cli --test cli prints_config_schema`
Expected: FAIL — flag prints nothing / not handled.

- [ ] **Step 3: Implement**

In `real_main` (main.rs), before building parameters / reading input:
```rust
if args.print_config_schema {
    let schema = schemars::schema_for!(crate::config::Config);
    println!("{}", serde_json::to_string_pretty(&schema)?);
    return Ok(());
}
```
(Place this alongside the existing early-exit flags. Ensure `--print-config-schema` does not require input files — it should work with no positional args.)

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p naga-cli --test cli prints_config_schema`
Expected: PASS.

- [ ] **Step 5: Full suite + clippy**

Run: `cargo test -p naga-cli && cargo clippy -p naga-cli --all-targets -- -D warnings`
Expected: all PASS, clippy clean. Regenerate help snapshot if the new flag changed `--help`.

- [ ] **Step 6: Commit**

```bash
git add naga-cli/src/main.rs naga-cli/tests
git commit -m "feat(naga-cli): --print-config-schema"
```

---

## Self-Review

**Spec coverage (Phase 2 slice — config vertical):**
- Fill serde gaps so all Options are Deserialize → Tasks 1-3. ✓
- schemars for discoverability → Task 4. ✓
- `--config` / `--config-json` (exclusive) → Task 5. ✓
- `--print-config-schema` → Task 6. ✓
- Full option coverage via config (design's primary "expose everything" lever) → Tasks 5-6. ✓
- NOT in this phase (deferred): clap per-field auto-flag flatten (Phase 3, pending CommonOptions/collision decision); `--format json` diagnostics/reflection (Phase 4); tool hooks (Phase 5); examples/docs (Phase 6).

**Placeholder scan:** No TBD/TODO. Two genuine external-dependency unknowns are flagged with explicit fallbacks (schemars 1 vs 0.8 API in Tasks 4-6; the serde `bound`/`skip` combination for the spv lifetime in Task 3) — these are real-world adaptation points with stated stop-and-report conditions, not vague hand-waving.

**Type consistency:**
- `Config` field types (Task 5) reference exactly the naga Options made serde-ready in Tasks 1-3 (`front::spv::Options`, `back::spv::Options<'static>`, `back::{msl,glsl,hlsl,dot}::Options`) plus already-serde `BoundsCheckPolicies`/`Capabilities`.
- `apply_config` writes the same `Parameters` fields that `build_parameters` (Phase 1) populates — names verified against params.rs: `spv_in, spv_out, msl, glsl, hlsl, dot, validation_flags, bounds_check_policies, capabilities, entry_point, keep_coordinate_space, defines, overrides`.
- schemars derive path `schemars::JsonSchema` + `schema_for!` used consistently in Tasks 4 and 6.

**Known verification points for the implementer** (not placeholders):
- `front::wgsl::Options` / `front::glsl::Options` current `Default` status (Task 2 Step 3) — check before adding a duplicate derive.
- `ShaderStage: Default` (Task 2) — determines whether glsl Options gets `serde(default)`.
- `back::spv::Options: Default` (Task 3) — needed for the test; report if absent.
- schemars bitflags handling (Task 4) — manual impl fallback provided.
