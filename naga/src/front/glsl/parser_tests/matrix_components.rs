use super::{parse_and_validate, Frontend, Options, ShaderStage};

#[test]
fn matrix_component_selection() {
    parse_and_validate(
        include_str!("../../../../tests/in/glsl/matrix-components.vert"),
        ShaderStage::Vertex,
    );
    for field in ["w", "xx"] {
        let source = alloc::format!(
            r#"
            #version 450
            void main() {{
                mat3 m;
                m[0].{field} = vec2(1.0);
            }}
            "#
        );
        assert!(Frontend::default()
            .parse(&Options::from(ShaderStage::Vertex), &source)
            .is_err());
    }
}
