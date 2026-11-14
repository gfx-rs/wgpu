fn validate(module: &naga::Module) {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .unwrap();
}

fn round_trip(module: naga::Module) {
    validate(&module);
    let bytes = bincode::serde::encode_to_vec(&module, bincode::config::standard()).unwrap();
    let (decoded, consumed) =
        bincode::serde::decode_from_slice::<naga::Module, _>(&bytes, bincode::config::standard())
            .unwrap();
    assert_eq!(consumed, bytes.len());
    validate(&decoded);

    // Arenas and blocks intentionally do not serialize source spans. Compare
    // the entire serialized graph instead of Debug output, which includes spans.
    assert_eq!(
        bytes,
        bincode::serde::encode_to_vec(&decoded, bincode::config::standard()).unwrap()
    );
    assert_eq!(module.functions.len(), decoded.functions.len());
    assert_eq!(module.entry_points.len(), decoded.entry_points.len());
    for (original, decoded) in module
        .functions
        .iter()
        .map(|(handle, function)| (function, &decoded.functions[handle]))
        .chain(
            module
                .entry_points
                .iter()
                .zip(&decoded.entry_points)
                .map(|(original, decoded)| (&original.function, &decoded.function)),
        )
    {
        assert_eq!(original.expression_controls, decoded.expression_controls);
        for (&handle, &control) in &decoded.expression_controls {
            assert_eq!(control, naga::ArithmeticControl::NoContraction);
            assert!(matches!(
                decoded.expressions[handle],
                naga::Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    ..
                }
            ));
        }
    }
}

const SOURCE: &str = "
    fn multiply(a: f32, b: f32) -> f32 { return a * b; }
    fn add(a: f32, b: f32) -> f32 { return a + b; }
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let value = multiply(f32(id.x), f32(id.y));
        let product = value * f32(id.z);
        let sum = add(product, value);
    }
    @compute @workgroup_size(1)
    fn ordinary(@builtin(global_invocation_id) id: vec3<u32>) {
        let value = add(f32(id.x), f32(id.y));
    }
";

fn mark_multiply(function: &mut naga::Function) {
    let producer = function
        .expressions
        .iter()
        .find_map(|(handle, expression)| {
            matches!(
                expression,
                naga::Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    ..
                }
            )
            .then_some(handle)
        })
        .unwrap();
    function
        .expression_controls
        .insert(producer, naga::ArithmeticControl::NoContraction);
}

#[test]
fn empty_controls_round_trip() {
    round_trip(naga::front::wgsl::parse_str(SOURCE).unwrap());
}

#[test]
fn mixed_controls_round_trip() {
    let mut module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    let function = module
        .functions
        .iter_mut()
        .find(|(_, function)| function.name.as_deref() == Some("multiply"))
        .unwrap()
        .1;
    mark_multiply(function);
    mark_multiply(&mut module.entry_points[0].function);
    assert_eq!(module.functions.len(), 2);
    assert_eq!(module.entry_points.len(), 2);
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|(_, function)| function.expression_controls.is_empty())
            .count(),
        1
    );
    assert!(module.entry_points[1]
        .function
        .expression_controls
        .is_empty());
    round_trip(module);
}

#[test]
fn missing_controls_default_in_text() {
    let module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    let text = ron::to_string(&module).unwrap();
    // Older textual IR has no expression_controls field on either owner kind.
    assert_eq!(text.matches("expression_controls:{},").count(), 4);
    let old_text = text.replace("expression_controls:{},", "");
    let decoded: naga::Module = ron::from_str(&old_text).unwrap();
    validate(&decoded);
    assert!(decoded
        .functions
        .iter()
        .all(|(_, function)| function.expression_controls.is_empty()));
    assert!(decoded
        .entry_points
        .iter()
        .all(|entry_point| entry_point.function.expression_controls.is_empty()));
    assert_eq!(text, ron::to_string(&decoded).unwrap());
    round_trip(decoded);
}
