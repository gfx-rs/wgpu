use super::{parse_and_validate, ErrorKind, Frontend, Options, ShaderStage};
use alloc::vec;

#[test]
fn matrix_interface_locations() {
    let module = parse_and_validate(
        include_str!("../../../../tests/in/glsl/matrix-interface.frag"),
        ShaderStage::Fragment,
    );
    let args = &module.entry_points[0].function.arguments;
    assert_eq!(args.len(), 9);
    for (i, arg) in args.iter().enumerate() {
        let Some(crate::Binding::Location {
            location,
            interpolation,
            ..
        }) = arg.binding
        else {
            panic!("missing location")
        };
        assert_eq!(location, i as u32);
        assert_eq!(
            interpolation,
            Some(if i < 6 {
                crate::Interpolation::Flat
            } else {
                crate::Interpolation::Perspective
            })
        );
    }
    let source = r#"
        #version 450
        layout(location=4294967295u) in mat2 m;
        void main() { gl_Position = vec4(m[0], m[1]); }
    "#;
    let error = Frontend::default()
        .parse(&Options::from(ShaderStage::Vertex), source)
        .unwrap_err();
    assert!(error.errors.iter().any(|error| {
        matches!(&error.kind, ErrorKind::SemanticError(message) if message == "interface location overflow")
    }));
    let module = parse_and_validate(
        r#"
            #version 450
            layout(location=0) noperspective centroid in mat2 m[2];
            void main() {}
        "#,
        ShaderStage::Fragment,
    );
    for arg in &module.entry_points[0].function.arguments {
        assert!(matches!(
            arg.binding,
            Some(crate::Binding::Location {
                interpolation: Some(crate::Interpolation::Linear),
                sampling: Some(crate::Sampling::Centroid),
                ..
            })
        ));
    }
}

#[test]
fn interface_sampling_qualifiers() {
    for (qualifier, expected_sampling) in [
        ("", crate::Sampling::Center),
        ("centroid", crate::Sampling::Centroid),
        ("sample", crate::Sampling::Sample),
    ] {
        let source = alloc::format!(
            r#"
            #version 450
            layout(location=0) {qualifier} in float scalar;
            layout(location=1) {qualifier} in vec2 vector;
            layout(location=2) {qualifier} in mat2 matrix;
            layout(location=4) {qualifier} in mat2 matrices[2];
            void main() {{}}
        "#
        );
        let module = parse_and_validate(&source, ShaderStage::Fragment);
        let arguments = &module.entry_points[0].function.arguments;
        assert_eq!(arguments.len(), 8);
        for (index, argument) in arguments.iter().enumerate() {
            let Some(crate::Binding::Location {
                location,
                interpolation,
                sampling,
                ..
            }) = argument.binding
            else {
                panic!("missing location")
            };
            assert_eq!(location, index as u32);
            assert_eq!(interpolation, Some(crate::Interpolation::Perspective));
            assert_eq!(
                sampling,
                Some(expected_sampling),
                "{qualifier} at {location}"
            );
        }
    }
}

#[test]
fn matrix_output_locations() {
    let module = parse_and_validate(
        r#"
            #version 450
            layout(location=2) flat out mat2x3 m[2];
            void main() {
                m[0] = mat2x3(1.0);
                m[1] = mat2x3(2.0);
                gl_Position = vec4(0.0);
            }
        "#,
        ShaderStage::Vertex,
    );
    let result = module.entry_points[0].function.result.as_ref().unwrap();
    let crate::TypeInner::Struct { ref members, .. } = module.types[result.ty].inner else {
        panic!("missing output struct")
    };
    let locations: vec::Vec<_> = members
        .iter()
        .filter_map(|m| match m.binding {
            Some(crate::Binding::Location { location, .. }) => Some(location),
            _ => None,
        })
        .collect();
    assert_eq!(locations, [2, 3, 4, 5]);
}
