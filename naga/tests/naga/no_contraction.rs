#![cfg(all(
    feature = "wgsl-in",
    feature = "wgsl-out",
    feature = "spv-in",
    feature = "spv-out",
    feature = "glsl-out",
    feature = "hlsl-out",
    feature = "msl-out"
))]

use naga::{back, front, valid, ArithmeticControl, Expression, Module};

const OP_DECORATE: u32 = 71;
const OP_TYPE_VOID: u32 = 19;
const OP_F_ADD: u32 = 129;
const OP_F_MUL: u32 = 133;
const OP_F_DIV: u32 = 136;
const NO_CONTRACTION: u32 = 42;

const SOURCE: &str = r#"
    @fragment fn main(@location(0) x: vec4<f32>) -> @location(0) vec4<f32> {
        let product = x.x * x.y;
        let sum = product + x.z;
        let quotient = sum / x.w;
        return vec4<f32>(product, sum, quotient, 1.0);
    }
"#;

fn validate(module: &Module) -> valid::ModuleInfo {
    valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(module)
        .unwrap()
}

fn write(module: &Module) -> Vec<u32> {
    let options = back::spv::Options::default();
    back::spv::write_vec(module, &validate(module), &options, None).unwrap()
}

fn parse(words: &[u32], preserve: bool) -> Result<Module, front::spv::Error> {
    let options = front::spv::Options {
        preserve_arithmetic_controls: preserve,
        ..Default::default()
    };
    front::spv::Frontend::new(words.iter().copied(), &options).parse()
}

fn instructions(words: &[u32]) -> Vec<&[u32]> {
    let mut rest = &words[5..];
    let mut instructions = Vec::new();
    while !rest.is_empty() {
        let (instruction, tail) = rest.split_at((rest[0] >> 16) as usize);
        instructions.push(instruction);
        rest = tail;
    }
    instructions
}

fn result_of(words: &[u32], opcode: u32) -> u32 {
    instructions(words)
        .into_iter()
        .find(|instruction| instruction[0] & 0xffff == opcode)
        .unwrap()[2]
}

fn no_contraction_targets(words: &[u32]) -> Vec<u32> {
    instructions(words)
        .into_iter()
        .filter(|i| i[0] & 0xffff == OP_DECORATE && i[2] == NO_CONTRACTION)
        .map(|i| i[1])
        .collect()
}

/// `words` with a `NoContraction` decoration on each of `targets`.
fn decorated(words: &[u32], targets: &[u32]) -> Vec<u32> {
    let mut output = words[..5].to_vec();
    let mut inserted = false;
    for instruction in instructions(words) {
        if !inserted && instruction[0] & 0xffff == OP_TYPE_VOID {
            for &target in targets {
                output.extend([(3 << 16) | OP_DECORATE, target, NO_CONTRACTION]);
            }
            inserted = true;
        }
        output.extend(instruction);
    }
    assert!(inserted);
    output
}

/// The controlled expressions of every function. The SPIR-V frontend wraps
/// an entry point around the function that holds its body.
fn controls(module: &Module) -> Vec<(&Expression, ArithmeticControl)> {
    let functions = module.functions.iter().map(|(_, function)| function);
    let entry_points = module.entry_points.iter().map(|entry| &entry.function);
    functions
        .chain(entry_points)
        .flat_map(|function| {
            function
                .expression_controls
                .iter()
                .map(|(&handle, &control)| (&function.expressions[handle], control))
        })
        .collect()
}

#[test]
fn spv_in_ignores_no_contraction_by_default() {
    let plain = write(&front::wgsl::parse_str(SOURCE).unwrap());
    let targets = [result_of(&plain, OP_F_MUL), result_of(&plain, OP_F_ADD)];
    let words = decorated(&plain, &targets);
    assert!(!front::spv::Options::default().preserve_arithmetic_controls);

    let module = parse(&words, false).unwrap();
    assert!(!module.has_arithmetic_controls());
    // The module is what the undecorated SPIR-V gives.
    assert_eq!(write(&module), write(&parse(&plain, false).unwrap()));

    // A decoration that could not be preserved is ignored as well.
    let unsupported = decorated(&plain, &[result_of(&plain, OP_F_DIV)]);
    let module = parse(&unsupported, false).unwrap();
    assert!(!module.has_arithmetic_controls());
}

#[test]
fn spv_in_preserves_no_contraction_on_request() {
    let plain = write(&front::wgsl::parse_str(SOURCE).unwrap());
    let targets = [result_of(&plain, OP_F_MUL), result_of(&plain, OP_F_ADD)];
    let words = decorated(&plain, &targets);

    let module = parse(&words, true).unwrap();
    let found = controls(&module);
    assert_eq!(found.len(), 2);
    for (expression, control) in found {
        assert_eq!(control, ArithmeticControl::NoContraction);
        assert!(matches!(
            *expression,
            Expression::Binary {
                op: naga::BinaryOperator::Multiply | naga::BinaryOperator::Add,
                ..
            }
        ));
    }

    // The controls are written back as decorations and survive another import.
    let rewritten = write(&module);
    assert_eq!(no_contraction_targets(&rewritten).len(), 2);
    assert_eq!(controls(&parse(&rewritten, true).unwrap()).len(), 2);

    // Without decorations the option changes nothing.
    assert!(!parse(&plain, true).unwrap().has_arithmetic_controls());

    // A decoration that cannot be preserved is an error.
    let quotient = result_of(&plain, OP_F_DIV);
    let unsupported = decorated(&plain, &[quotient]);
    assert!(matches!(
        parse(&unsupported, true),
        Err(front::spv::Error::UnsupportedNoContraction(id)) if id == quotient
    ));
}

fn msl(module: &Module) -> Result<String, back::msl::Error> {
    let options = back::msl::Options::default();
    let pipeline = back::msl::PipelineOptions::default();
    back::msl::write_string(module, &validate(module), &options, &pipeline).map(|(text, _)| text)
}

fn wgsl(module: &Module) -> Result<String, back::wgsl::Error> {
    back::wgsl::write_string(module, &validate(module), back::wgsl::WriterFlags::empty())
}

fn hlsl(
    module: &Module,
    shader_model: back::hlsl::ShaderModel,
) -> Result<String, back::hlsl::Error> {
    let options = back::hlsl::Options {
        shader_model,
        ..Default::default()
    };
    let pipeline = back::hlsl::PipelineOptions::default();
    let mut text = String::new();
    back::hlsl::Writer::new(&mut text, &options, &pipeline)
        .write(module, &validate(module), None)
        .map(|_| text)
}

fn glsl(module: &Module) -> Result<String, back::glsl::Error> {
    let options = back::glsl::Options {
        version: back::glsl::Version::Desktop(450),
        ..Default::default()
    };
    let pipeline = back::glsl::PipelineOptions {
        shader_stage: naga::ShaderStage::Fragment,
        entry_point: "main".into(),
        multiview: None,
    };
    let info = validate(module);
    let policies = naga::proc::BoundsCheckPolicies::default();
    let mut text = String::new();
    back::glsl::Writer::new(&mut text, module, &info, &options, &pipeline, policies)
        .and_then(|mut writer| writer.write())
        .map(|_| text)
}

/// With default frontend options, decorated SPIR-V is translated exactly like
/// the same SPIR-V without the decorations, on every backend. This includes
/// the backends that cannot honor a control.
#[test]
fn default_translation_does_not_depend_on_no_contraction() {
    let plain = write(&front::wgsl::parse_str(SOURCE).unwrap());
    let targets = [result_of(&plain, OP_F_MUL), result_of(&plain, OP_F_ADD)];
    let decorated = parse(&decorated(&plain, &targets), false).unwrap();
    let plain = parse(&plain, false).unwrap();

    assert_eq!(msl(&decorated).unwrap(), msl(&plain).unwrap());
    assert_eq!(wgsl(&decorated).unwrap(), wgsl(&plain).unwrap());
    assert_eq!(glsl(&decorated).unwrap(), glsl(&plain).unwrap());
    for shader_model in [back::hlsl::ShaderModel::V5_1, back::hlsl::ShaderModel::V6_0] {
        assert_eq!(
            hlsl(&decorated, shader_model).unwrap(),
            hlsl(&plain, shader_model).unwrap()
        );
    }
    assert!(!msl(&plain).unwrap().is_empty());
}

/// With preserved controls, a backend either honors them or returns an error.
#[test]
fn preserved_controls_are_honored_or_refused() {
    let plain = write(&front::wgsl::parse_str(SOURCE).unwrap());
    let targets = [result_of(&plain, OP_F_MUL), result_of(&plain, OP_F_ADD)];
    let module = parse(&decorated(&plain, &targets), true).unwrap();

    assert!(matches!(
        msl(&module),
        Err(back::msl::Error::UnsupportedArithmeticControls)
    ));
    assert!(matches!(
        wgsl(&module),
        Err(back::wgsl::Error::UnsupportedArithmeticControls)
    ));
    // Shader model 5 is compiled by FXC, which has no function boundary
    // that confines `precise`.
    assert!(matches!(
        hlsl(&module, back::hlsl::ShaderModel::V5_1),
        Err(back::hlsl::Error::UnsupportedArithmeticControls)
    ));

    assert_eq!(no_contraction_targets(&write(&module)).len(), 2);
    let text = hlsl(&module, back::hlsl::ShaderModel::V6_0).unwrap();
    assert_eq!(text.matches("precise float result = ").count(), 2, "{text}");
    let text = glsl(&module).unwrap();
    assert_eq!(text.matches("precise ").count(), 2, "{text}");
}
