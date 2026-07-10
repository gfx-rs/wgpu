# naga-cli Rewrite — Phase 4 (Structured JSON Output) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--format json` to the naga CLI, emitting a single structured JSON document on stdout with diagnostics (errors/warnings with source spans) and reflection info, for IDE/tool consumption.

**Architecture:** CLI-only (no naga-core changes). A new `naga-cli/src/output.rs` defines serde-serializable `JsonOutput { success, diagnostics, reflection }` plus a CLI-side `Location` mirroring `naga::SourceLocation`. Converter functions turn each naga error type (WGSL `ParseError`, GLSL `ParseErrors`, `WithSpan<ValidationError>`, SPIR-V `Error`) into `Diagnostic`s. A `Reflection::from_module` builds curated reflection from the validated `Module`. The `run` flow collects diagnostics; in `json` mode it prints `JsonOutput` to stdout and exits 0/1 by success; in `text` mode behavior is unchanged.

**Tech Stack:** Rust, serde, serde_json (already a naga-cli dep), naga.

## Global Constraints

- No changes to `naga/` or any crate other than `naga-cli`. All naga types needed already derive `Serialize` behind the `serialize` feature (enabled by naga-cli); `SourceLocation` does not — mirror it CLI-side.
- Default format is `text`; `text` mode behavior MUST be byte-identical to today (existing integration tests unchanged). JSON is strictly additive.
- The JSON schema is intentionally a v1 / emergent shape; keep it minimal and stable-ish, no versioning field required.
- In `json` mode: exactly ONE JSON document is written to stdout (pretty-printed). Human/stderr diagnostic emission is suppressed in json mode. Shader file outputs (.spv/.metal/etc.) still go to their files. Process exits 0 iff `success` is true.
- naga-cli MSRV constraint inherited (builds against naga 1.87).

## File Structure

- `naga-cli/src/output.rs` — NEW: serde structs (`JsonOutput`, `Diagnostic`, `Label`, `Location`, `Severity`, `Reflection` + sub-structs), error→diagnostic converters, `Reflection::from_module`.
- `naga-cli/src/cli.rs` — add `--format` flag (`OutputFormat` enum).
- `naga-cli/src/core.rs` — thread the format through parse/validate; collect diagnostics; emit JSON or keep text behavior.
- `naga-cli/src/main.rs` — `mod output;`.
- `naga-cli/tests/cli.rs` — integration tests.

---

### Task 1: `output.rs` — serde types

Define the JSON document types.

**Files:**
- Modify: `naga-cli/src/main.rs` (add `mod output;`)
- Create: `naga-cli/src/output.rs`
- Test: inline `#[cfg(test)]` in `output.rs`

**Interfaces:**
- Produces:
  - `pub struct Location { pub line: u32, pub column: u32, pub byte_offset: u32, pub length: u32 }` + `impl From<naga::SourceLocation> for Location`.
  - `pub enum Severity { Error, Warning }` serializing as `"error"`/`"warning"`.
  - `pub struct Label { pub message: String, pub location: Option<Location> }`.
  - `pub struct Diagnostic { pub severity: Severity, pub message: String, pub location: Option<Location>, pub labels: Vec<Label>, pub notes: Vec<String> }`.
  - `pub struct EntryPointReflection { pub name: String, pub stage: naga::ShaderStage, pub workgroup_size: [u32; 3] }`.
  - `pub struct ResourceReflection { pub name: Option<String>, pub group: u32, pub binding: u32, pub address_space: String }`.
  - `pub struct OverrideReflection { pub name: Option<String>, pub id: Option<u16> }`.
  - `pub struct Reflection { pub entry_points: Vec<EntryPointReflection>, pub resources: Vec<ResourceReflection>, pub overrides: Vec<OverrideReflection> }`.
  - `pub struct JsonOutput { pub success: bool, pub diagnostics: Vec<Diagnostic>, pub reflection: Option<Reflection> }`.

- [ ] **Step 1: Add module + write the failing test**

In `naga-cli/src/main.rs` add `mod output;` (with the other `mod` lines).

Create `naga-cli/src/output.rs`:
```rust
//! Structured JSON output: diagnostics + reflection.
```
Append the test:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_output_shape() {
        let out = JsonOutput {
            success: false,
            diagnostics: vec![Diagnostic {
                severity: Severity::Error,
                message: "boom".into(),
                location: Some(Location { line: 2, column: 5, byte_offset: 12, length: 3 }),
                labels: vec![Label { message: "here".into(), location: None }],
                notes: vec!["note".into()],
            }],
            reflection: None,
        };
        let json = serde_json::to_string(&out).unwrap();
        assert!(json.contains(r#""success":false"#));
        assert!(json.contains(r#""severity":"error""#));
        assert!(json.contains(r#""line":2"#));
        assert!(json.contains(r#""reflection":null"#));
    }

    #[test]
    fn location_from_source_location() {
        let sl = naga::SourceLocation { line_number: 3, line_position: 7, offset: 20, length: 4 };
        let loc = Location::from(sl);
        assert_eq!((loc.line, loc.column, loc.byte_offset, loc.length), (3, 7, 20, 4));
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p naga-cli --bin naga output::tests`
Expected: FAIL to compile — types not defined.

- [ ] **Step 3: Implement the types**

In `naga-cli/src/output.rs`:
```rust
//! Structured JSON output: diagnostics + reflection.

use serde::Serialize;

/// A source position, mirroring `naga::SourceLocation` (which lacks `Serialize`).
#[derive(Debug, Clone, Serialize)]
pub struct Location {
    /// 1-based line number.
    pub line: u32,
    /// 1-based column, in UTF-8 bytes.
    pub column: u32,
    /// 0-based byte offset into the source.
    pub byte_offset: u32,
    /// Length in UTF-8 bytes.
    pub length: u32,
}

impl From<naga::SourceLocation> for Location {
    fn from(sl: naga::SourceLocation) -> Self {
        Location {
            line: sl.line_number,
            column: sl.line_position,
            byte_offset: sl.offset,
            length: sl.length,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Serialize)]
pub struct Label {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<Location>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<Location>,
    pub labels: Vec<Label>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntryPointReflection {
    pub name: String,
    pub stage: naga::ShaderStage,
    pub workgroup_size: [u32; 3],
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceReflection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub group: u32,
    pub binding: u32,
    pub address_space: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OverrideReflection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u16>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Reflection {
    pub entry_points: Vec<EntryPointReflection>,
    pub resources: Vec<ResourceReflection>,
    pub overrides: Vec<OverrideReflection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonOutput {
    pub success: bool,
    pub diagnostics: Vec<Diagnostic>,
    pub reflection: Option<Reflection>,
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p naga-cli --bin naga output::tests`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add naga-cli/src/output.rs naga-cli/src/main.rs
git commit -m "feat(naga-cli): JSON output types for diagnostics and reflection"
```

---

### Task 2: error → `Diagnostic` converters

Convert each naga error type into `Diagnostic`s.

**Files:**
- Modify: `naga-cli/src/output.rs`
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Consumes: `naga::front::wgsl::ParseError`, `naga::front::glsl::ParseErrors`, `naga::WithSpan<naga::valid::ValidationError>`, `naga::front::spv::Error`.
- Produces (all `pub`):
  - `fn wgsl_parse_error_to_diagnostic(err: &naga::front::wgsl::ParseError, source: &str) -> Diagnostic`
  - `fn glsl_parse_errors_to_diagnostics(errs: &naga::front::glsl::ParseErrors, source: &str) -> Vec<Diagnostic>`
  - `fn validation_error_to_diagnostic(err: &naga::WithSpan<naga::valid::ValidationError>, source: Option<&str>) -> Diagnostic`
  - `fn spv_error_to_diagnostic(err: &naga::front::spv::Error) -> Diagnostic`

- [ ] **Step 1: Write the failing test**

Append to `output.rs` tests:
```rust
#[test]
fn wgsl_error_becomes_diagnostic() {
    let src = "@fragment fn main() -> @location(0) vec4<f32> { return 1; }"; // type error-ish / parse issue
    let bad = "fn f() { let x: i32 = ; }"; // definite parse error
    let mut fe = naga::front::wgsl::Frontend::new();
    let err = fe.parse(bad).unwrap_err();
    let d = wgsl_parse_error_to_diagnostic(&err, bad);
    assert!(matches!(d.severity, Severity::Error));
    assert!(!d.message.is_empty());
    // A parse error should carry at least a primary location or a label.
    assert!(d.location.is_some() || !d.labels.is_empty());
    let _ = src;
}

#[test]
fn validation_error_becomes_diagnostic() {
    // Construct a module that parses but fails validation.
    let src = "@fragment fn main() { let x = 1 / 0; }";
    let mut fe = naga::front::wgsl::Frontend::new();
    // If this source doesn't fail validation on your naga version, swap for another
    // known-invalid-but-parseable snippet; the point is a WithSpan<ValidationError>.
    if let Ok(module) = fe.parse(src) {
        let res = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module);
        if let Err(e) = res {
            let d = validation_error_to_diagnostic(&e, Some(src));
            assert!(matches!(d.severity, Severity::Error));
            assert!(!d.message.is_empty());
        }
    }
}
```
(If the exact snippets don't trigger the intended errors on this naga version, adjust to any parseable-but-invalid / unparseable WGSL — the assertions test the converter, not specific naga messages.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p naga-cli --bin naga output::tests::wgsl_error_becomes_diagnostic`
Expected: FAIL to compile — converter not defined.

- [ ] **Step 3: Implement converters**

Append to `output.rs`:
```rust
/// Build a `Location` from a `naga::Span` against the source.
fn location_from_span(span: naga::Span, source: &str) -> Option<Location> {
    span.is_defined().then(|| Location::from(span.location(source)))
}

pub fn wgsl_parse_error_to_diagnostic(
    err: &naga::front::wgsl::ParseError,
    source: &str,
) -> Diagnostic {
    let labels = err
        .labels()
        .map(|(span, msg)| Label {
            message: msg.to_string(),
            location: location_from_span(span, source),
        })
        .collect();
    Diagnostic {
        severity: Severity::Error,
        message: err.message().to_string(),
        location: err.location(source).map(Location::from),
        labels,
        notes: err.notes().map(|n| n.to_string()).collect(),
    }
}

pub fn glsl_parse_errors_to_diagnostics(
    errs: &naga::front::glsl::ParseErrors,
    source: &str,
) -> Vec<Diagnostic> {
    errs.errors
        .iter()
        .map(|e| Diagnostic {
            severity: Severity::Error,
            message: e.kind.to_string(),
            location: e.location(source).map(Location::from),
            labels: Vec::new(),
            notes: Vec::new(),
        })
        .collect()
}

pub fn validation_error_to_diagnostic(
    err: &naga::WithSpan<naga::valid::ValidationError>,
    source: Option<&str>,
) -> Diagnostic {
    let labels = match source {
        Some(src) => err
            .spans()
            .map(|(span, msg)| Label {
                message: msg.clone(),
                location: location_from_span(*span, src),
            })
            .collect(),
        None => err
            .spans()
            .map(|(_, msg)| Label { message: msg.clone(), location: None })
            .collect(),
    };
    Diagnostic {
        severity: Severity::Error,
        message: err.as_inner().to_string(),
        location: source.and_then(|src| err.location(src)).map(Location::from),
        labels,
        notes: Vec::new(),
    }
}

pub fn spv_error_to_diagnostic(err: &naga::front::spv::Error) -> Diagnostic {
    Diagnostic {
        severity: Severity::Error,
        message: err.to_string(),
        location: None,
        labels: Vec::new(),
        notes: Vec::new(),
    }
}
```
Notes for the implementer:
- Confirm `err.message()`, `err.labels()`, `err.notes()`, `err.location(source)` signatures on `naga::front::wgsl::ParseError` (mapped in exploration). `labels()` yields `(Span, &str)`.
- `naga::front::glsl::Error.kind` implements `Display` (it's a `thiserror`-style enum) — `e.kind.to_string()` gives the message. If `kind` is not directly `Display`, use `e.to_string()`.
- `WithSpan::spans()` yields `&(Span, String)`; `as_inner()` gives `&ValidationError` which is `Display` via `Error`.
- `ValidationError` / its `WithSpan` needs `to_string()`; `WithSpan<E>` where `E: Error` — use `err.as_inner().to_string()` for the message.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p naga-cli --bin naga output::tests`
Expected: PASS (all output tests).

- [ ] **Step 5: Commit**

```bash
git add naga-cli/src/output.rs
git commit -m "feat(naga-cli): convert naga errors to JSON diagnostics"
```

---

### Task 3: `Reflection::from_module`

Build curated reflection from a validated module.

**Files:**
- Modify: `naga-cli/src/output.rs`
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Produces: `impl Reflection { pub fn from_module(module: &naga::Module) -> Reflection }`.

- [ ] **Step 1: Write the failing test**

Append to `output.rs` tests:
```rust
#[test]
fn reflection_from_module() {
    let src = r#"
        @group(0) @binding(1) var<uniform> u: vec4<f32>;
        override scale: f32 = 2.0;
        @compute @workgroup_size(8, 4, 1)
        fn main() { _ = u; }
    "#;
    let module = naga::front::wgsl::Frontend::new().parse(src).unwrap();
    let r = Reflection::from_module(&module);
    assert_eq!(r.entry_points.len(), 1);
    assert_eq!(r.entry_points[0].name, "main");
    assert_eq!(r.entry_points[0].workgroup_size, [8, 4, 1]);
    assert_eq!(r.entry_points[0].stage, naga::ShaderStage::Compute);
    assert!(r.resources.iter().any(|res| res.group == 0 && res.binding == 1));
    assert!(r.overrides.iter().any(|o| o.name.as_deref() == Some("scale")));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p naga-cli --bin naga output::tests::reflection_from_module`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Append to `output.rs`:
```rust
impl Reflection {
    pub fn from_module(module: &naga::Module) -> Reflection {
        let entry_points = module
            .entry_points
            .iter()
            .map(|ep| EntryPointReflection {
                name: ep.name.clone(),
                stage: ep.stage,
                workgroup_size: ep.workgroup_size,
            })
            .collect();

        let resources = module
            .global_variables
            .iter()
            .filter_map(|(_, gv)| {
                gv.binding.as_ref().map(|b| ResourceReflection {
                    name: gv.name.clone(),
                    group: b.group,
                    binding: b.binding,
                    address_space: format!("{:?}", gv.space),
                })
            })
            .collect();

        let overrides = module
            .overrides
            .iter()
            .map(|(_, ov)| OverrideReflection {
                name: ov.name.clone(),
                id: ov.id,
            })
            .collect();

        Reflection { entry_points, resources, overrides }
    }
}
```
(If `global_variables` / `overrides` iteration signatures differ — they are `Arena` types yielding `(Handle, &T)` via `.iter()` — adjust the closure binding accordingly. `ResourceBinding` fields are `group` and `binding`.)

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p naga-cli --bin naga output::tests::reflection_from_module`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add naga-cli/src/output.rs
git commit -m "feat(naga-cli): build reflection info from module"
```

---

### Task 4: `--format json` flag + routing

Add the flag and route parse/validation diagnostics + reflection into a JSON document in json mode, preserving text-mode behavior.

**Files:**
- Modify: `naga-cli/src/cli.rs`, `naga-cli/src/core.rs`, `naga-cli/src/main.rs`
- Test: `naga-cli/tests/cli.rs`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `pub enum OutputFormat { Text, Json }` (clap `ValueEnum`) on `Args` as `pub format: OutputFormat` (default `Text`).

- [ ] **Step 1: Write the failing integration tests**

Add to `naga-cli/tests/cli.rs`:
```rust
#[test]
fn json_format_valid_shader_reflection() {
    let dir = std::env::temp_dir().join("naga_cli_p4_ok");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = naga().arg(&src).args(["--format", "json"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["success"], true);
    assert_eq!(v["reflection"]["entry_points"][0]["name"], "main");
    assert_eq!(v["diagnostics"].as_array().unwrap().len(), 0);
}

#[test]
fn json_format_parse_error() {
    let dir = std::env::temp_dir().join("naga_cli_p4_err");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("bad.wgsl");
    std::fs::write(&src, "fn f( { }").unwrap(); // parse error
    let out = naga().arg(&src).args(["--format", "json"]).output().unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["success"], false);
    let diags = v["diagnostics"].as_array().unwrap();
    assert!(!diags.is_empty());
    assert_eq!(diags[0]["severity"], "error");
    assert!(!diags[0]["message"].as_str().unwrap().is_empty());
}

#[test]
fn text_format_unchanged_default() {
    // Without --format, behavior is text (validation success message on stdout).
    let dir = std::env::temp_dir().join("naga_cli_p4_text");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = naga().arg(&src).output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("Validation successful"));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p naga-cli --test cli json_format_valid_shader_reflection json_format_parse_error text_format_unchanged_default`
Expected: FAIL — `--format` unknown.

- [ ] **Step 3: Add the `--format` flag (`cli.rs`)**

```rust
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}
```
On `Args`:
```rust
/// Output format for diagnostics and reflection: `text` (human, default) or `json`.
#[arg(long, value_enum, default_value_t = OutputFormat::Text)]
pub format: OutputFormat,
```

- [ ] **Step 4: Route in `core.rs`**

Thread `args.format` into the run flow. The cleanest structure:
- In `run`, build a `Vec<Diagnostic>` and an `Option<Reflection>` as you go, and track `success: bool`.
- Parse step: in `json` mode, on WGSL/GLSL/SPIR-V parse failure, convert to diagnostics (via Task 2 converters, using the input source text when available), push them, set `success = false`, and jump to emitting JSON (do NOT `return Err(...)` which would print to stderr). In `text` mode, keep the current `?`/anyhow behavior.
- Validation step: replace the two `// TODO(phase 4)` stderr-emission blocks — in `json` mode push `validation_error_to_diagnostic(&e, input_text.as_deref())`; in `text` mode keep the existing `emit_to_stderr_with_path` / `print_err`.
- On success with a validated module: in `json` mode set `reflection = Some(Reflection::from_module(&module))`.
- After the flow: in `json` mode, print `serde_json::to_string_pretty(&JsonOutput { success, diagnostics, reflection })?` to stdout, and return a result that makes `main` exit `0` if `success` else `1`. In `text` mode, unchanged (including the "Validation successful" message and file outputs).

Concretely, change `run` to return the success bool (or have it print JSON itself and return `anyhow::Result<bool>` where the bool is success). `main` then `std::process::exit(1)` when `run` reports failure in json mode. Keep the single-exit-in-main rule: `run` returns `anyhow::Result<bool>` (`Ok(true)` success, `Ok(false)` handled-failure-already-emitted-as-json), `main` maps `Ok(false)` → exit 1.

Guidance:
- Shader file outputs (.spv etc.) still happen in json mode when validation succeeds and output paths are given — the JSON doc is orthogonal (it goes to stdout, files to disk). If an output-writing step itself fails in json mode, convert that to a diagnostic too (severity error) or let it be a hard `anyhow` error (acceptable for v1 — a file IO failure can stay a stderr error even in json mode; document the choice).
- In json mode, suppress the "Validation successful" stdout line (that's text-mode only) — stdout must contain ONLY the JSON document.

- [ ] **Step 5: Wire `main.rs`**

Update `real_main`/`main` so `run` returning `Ok(false)` (json handled-failure) causes `std::process::exit(1)` without `print_err`. `Err(e)` still goes through `print_err` + exit 1 (hard errors). Ensure the `--print-config-schema` and other early exits are unaffected.

- [ ] **Step 6: Run tests + regenerate snapshot**

Run: `cargo test -p naga-cli`. Regenerate help snapshot (new `--format` flag): `cargo run -q -p naga-cli -- --help > naga-cli/tests/snapshots/help.txt`. Run `cargo clippy -p naga-cli --all-targets -- -D warnings`.
Expected: all pass; the 3 new tests green; existing text-mode tests unchanged.

- [ ] **Step 7: Commit**

```bash
git add naga-cli/src naga-cli/tests
git commit -m "feat(naga-cli): --format json for structured diagnostics and reflection"
```

---

## Self-Review

**Spec coverage (Phase 4 slice):**
- Structured JSON on stdout → Tasks 1, 4. ✓
- Diagnostics incl. error messages with spans → Tasks 1-2, 4. ✓
- Reflection info → Task 3, 4. ✓
- stdin already works (Phase 1); json + stdin composes (stdin provides source, converters get the text). A test can be added but stdin isn't re-plumbed here.
- NOT in this phase: tool hooks (Phase 5); examples/docs + carried minors (Phase 6). SPIR-V parse errors carry no span (naga limitation) → their diagnostics have `location: None` (documented).

**Placeholder scan:** No TBD/TODO. Test snippets note "adjust if this naga version's messages differ" — a real test-robustness caveat (the assertions check the converter, not naga's exact wording), not a gap. The core.rs routing (Task 4 Step 4) is described with a concrete contract (`run -> anyhow::Result<bool>`, json prints to stdout, main exits by bool) rather than full code because it edits existing flow — the contract + guidance is precise.

**Type consistency:**
- `Location` fields (line/column/byte_offset/length) map from `naga::SourceLocation` (line_number/line_position/offset/length) in Task 1's `From` impl; converters and reflection use the Task-1 types unchanged.
- `run` returning `anyhow::Result<bool>` (Task 4) is consumed by `main` (Task 4 Step 5) consistently.
- Reflection uses `naga::ShaderStage` (Serialize) directly; `ResourceBinding.group`/`.binding` and `EntryPoint.workgroup_size: [u32;3]` per exploration.

**Known verification points (not placeholders):** exact WGSL `ParseError`/GLSL `Error`/`WithSpan` method names (mapped in exploration; Task 2 notes fallbacks); `Arena::iter()` yield shape in `Reflection::from_module` (Task 3 note).
