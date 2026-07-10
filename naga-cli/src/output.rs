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
