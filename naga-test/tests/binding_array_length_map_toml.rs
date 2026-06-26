use naga::{back::msl::PipelineOptions, ResourceBinding};

#[test]
fn binding_array_length_map_toml_roundtrip() {
    let toml_str = r#"
vertex_pulling_transform = true
vertex_buffer_mappings = []

[[binding_array_length_map]]
resource_binding = { group = 0, binding = 0 }
count = 10
"#;

    let parsed: PipelineOptions = toml::from_str(toml_str).unwrap();
    assert_eq!(parsed.binding_array_length_map.len(), 1);
    assert_eq!(
        parsed.binding_array_length_map.get(&ResourceBinding {
            group: 0,
            binding: 0
        }),
        Some(&10)
    );
}
