//! Structured JSON output: diagnostics + reflection.

use serde::Serialize;

/// Version of the `--format json` document. Bump on any incompatible change.
pub const JSON_OUTPUT_VERSION: u32 = 1;

/// A source position, mirroring `naga::SourceLocation` (which lacks `Serialize`).
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
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

#[derive(Debug, Clone, Copy, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct Label {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<Location>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<Location>,
    pub labels: Vec<Label>,
    pub notes: Vec<String>,
}

/// The `--format json` document.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct JsonOutput {
    /// See [`JSON_OUTPUT_VERSION`]. The format is not yet stable.
    pub version: u32,
    pub success: bool,
    pub diagnostics: Vec<Diagnostic>,
    /// Reflection of the validated module, before pipeline overrides are applied.
    pub reflection: Option<naga::reflect::ModuleReflection>,
}

impl JsonOutput {
    pub fn new(
        success: bool,
        diagnostics: Vec<Diagnostic>,
        reflection: Option<naga::reflect::ModuleReflection>,
    ) -> Self {
        JsonOutput {
            version: JSON_OUTPUT_VERSION,
            success,
            diagnostics,
            reflection,
        }
    }
}

/// Reflect a validated module. Layout failures cannot happen after successful
/// validation, so they are reported as a missing reflection.
pub fn reflect(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Option<naga::reflect::ModuleReflection> {
    naga::reflect::reflect(module, info).ok()
}

/// Build a `Location` from a `naga::Span` against the source.
fn location_from_span(span: naga::Span, source: &str) -> Option<Location> {
    span.is_defined()
        .then(|| Location::from(span.location(source)))
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
            .map(|(_, msg)| Label {
                message: msg.clone(),
                location: None,
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_error_becomes_diagnostic() {
        let bad = "fn f() { let x: i32 = ; }"; // definite parse error
        let mut fe = naga::front::wgsl::Frontend::new();
        let err = fe.parse(bad).unwrap_err();
        let d = wgsl_parse_error_to_diagnostic(&err, bad);
        assert!(matches!(d.severity, Severity::Error));
        assert!(!d.message.is_empty());
        // A parse error should carry at least a primary location or a label.
        assert!(d.location.is_some() || !d.labels.is_empty());
    }

    #[test]
    fn validation_error_becomes_diagnostic() {
        // A fragment entry point returning f32 without a @location annotation
        // parses successfully but fails naga validation with
        // "Entry point main at Fragment is invalid".
        let src = "@fragment fn main() -> f32 { return 0.0; }";
        let module = naga::front::wgsl::Frontend::new()
            .parse(src)
            .expect("snippet must PARSE (validation-only failure)");
        let err = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect_err("snippet must FAIL validation");
        let d = validation_error_to_diagnostic(&err, Some(src));
        assert!(matches!(d.severity, Severity::Error));
        assert!(!d.message.is_empty());
        // Location may or may not be present depending on the error's span,
        // but a validation error must carry at least a message.
    }

    #[test]
    fn json_output_shape() {
        let out = JsonOutput::new(
            false,
            vec![Diagnostic {
                severity: Severity::Error,
                message: "boom".into(),
                location: Some(Location {
                    line: 2,
                    column: 5,
                    byte_offset: 12,
                    length: 3,
                }),
                labels: vec![Label {
                    message: "here".into(),
                    location: None,
                }],
                notes: vec!["note".into()],
            }],
            None,
        );
        let json = serde_json::to_string(&out).unwrap();
        assert!(json.contains(r#""version":1"#));
        assert!(json.contains(r#""success":false"#));
        assert!(json.contains(r#""severity":"error""#));
        assert!(json.contains(r#""byteOffset":12"#));
        assert!(json.contains(r#""reflection":null"#));
    }

    #[test]
    fn glsl_errors_become_diagnostics() {
        // A definitely-invalid GLSL fragment — "this is not valid glsl" is not a
        // statement, so the parser must emit at least one parse error.
        let src = "#version 450\nvoid main() { this is not valid glsl }";
        let mut fe = naga::front::glsl::Frontend::default();
        let opts = naga::front::glsl::Options {
            stage: naga::ShaderStage::Fragment,
            defines: Default::default(),
        };
        let errs = fe.parse(&opts, src).unwrap_err();
        let diags = glsl_parse_errors_to_diagnostics(&errs, src);
        assert!(
            !diags.is_empty(),
            "expected ≥1 diagnostic from invalid GLSL"
        );
        assert!(
            diags.iter().all(|d| matches!(d.severity, Severity::Error)),
            "all GLSL parse diagnostics must be Error"
        );
        assert!(
            diags.iter().all(|d| !d.message.is_empty()),
            "all GLSL parse diagnostics must have a non-empty message"
        );
    }

    #[test]
    fn spv_error_becomes_diagnostic() {
        // [0, 0, 0, 0] is four bytes that do not form a valid SPIR-V magic word,
        // so parse_u8_slice must return an Err.
        let err = naga::front::spv::parse_u8_slice(&[0, 0, 0, 0], &Default::default()).unwrap_err();
        let d = spv_error_to_diagnostic(&err);
        assert!(matches!(d.severity, Severity::Error));
        assert!(
            !d.message.is_empty(),
            "spv error diagnostic must have a non-empty message"
        );
        assert!(
            d.location.is_none(),
            "spv error diagnostic must have no source location"
        );
    }

    #[test]
    fn location_from_source_location() {
        let sl = naga::SourceLocation {
            line_number: 3,
            line_position: 7,
            offset: 20,
            length: 4,
        };
        let loc = Location::from(sl);
        assert_eq!(
            (loc.line, loc.column, loc.byte_offset, loc.length),
            (3, 7, 20, 4)
        );
    }

    #[test]
    fn reflection_from_module() {
        let src = r#"
            @group(0) @binding(1) var<uniform> u: vec4<f32>;
            override scale: f32 = 2.0;
            @compute @workgroup_size(8, 4, 1)
            fn main() { _ = u; }
        "#;
        let module = naga::front::wgsl::Frontend::new().parse(src).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let r = reflect(&module, &info).unwrap();
        assert_eq!(r.entry_points.len(), 1);
        assert_eq!(r.entry_points[0].name, "main");
        assert_eq!(r.entry_points[0].workgroup_size, [8, 4, 1]);
        assert_eq!(r.entry_points[0].stage, naga::ShaderStage::Compute);
        assert_eq!(r.entry_points[0].resources, [0]);
        assert!(r
            .resources
            .iter()
            .any(|res| res.binding.group == 0 && res.binding.binding == 1));
        assert!(r
            .overrides
            .iter()
            .any(|o| o.name.as_deref() == Some("scale")));
    }
}
