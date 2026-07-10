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
    #[allow(dead_code)]
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
}
