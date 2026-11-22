//! Tests for [`naga::inline`].

#![cfg(feature = "wgsl-in")]

use naga::{
    inline::{InlineStrategy, InlineTarget},
    valid::{Capabilities, ModuleInfo, ValidationFlags, Validator},
    Module, Statement,
};

fn parse(source: &str) -> (Module, ModuleInfo) {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    (module, info)
}

/// Run `process` on a copy of `module`. Return `None` if it changed nothing.
fn inline(module: &Module, strategy: InlineStrategy) -> Option<Module> {
    let mut module = module.clone();
    naga::inline::process(&mut module, strategy)
        .unwrap()
        .map(|_| module)
}

fn has_call(block: &naga::Block) -> bool {
    block.iter().any(|statement| match *statement {
        Statement::Call { .. } => true,
        Statement::Block(ref block) => has_call(block),
        Statement::If {
            ref accept,
            ref reject,
            ..
        } => has_call(accept) || has_call(reject),
        Statement::Switch { ref cases, .. } => cases.iter().any(|case| has_call(&case.body)),
        Statement::Loop {
            ref body,
            ref continuing,
            ..
        } => has_call(body) || has_call(continuing),
        _ => false,
    })
}

const STORAGE_POINTER: &str = "
    @group(0) @binding(0) var<storage, read_write> buf: array<u32, 4>;

    fn store(p: ptr<storage, u32, read_write>) {
        *p = 1u;
    }

    @compute @workgroup_size(1)
    fn main() {
        store(&buf[1]);
    }
";

const IMMEDIATE_POINTER: &str = "
    struct Pc {
        a: u32,
        b: u32,
    }

    var<immediate> pc: Pc;
    @group(0) @binding(0) var<storage, read_write> buf: array<u32, 2>;

    fn read(p: ptr<immediate, u32>) -> u32 {
        return *p;
    }

    @compute @workgroup_size(1)
    fn main() {
        buf[0] = read(&pc.b);
    }
";

const TASK_PAYLOAD_POINTER: &str = "
    enable wgpu_mesh_shader;

    struct Payload {
        a: u32,
    }

    var<task_payload> payload: Payload;

    fn store(p: ptr<task_payload, u32>) {
        *p = 1u;
    }

    @task
    @payload(payload)
    @workgroup_size(1)
    fn main() -> @builtin(mesh_task_size) vec3<u32> {
        store(&payload.a);
        return vec3(1u, 1u, 1u);
    }
";

const BINDING_ARRAY: &str = "
    enable wgpu_binding_array;

    @group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
    @group(0) @binding(1) var samplers: binding_array<sampler, 2>;

    fn sample(t: binding_array<texture_2d<f32>, 2>, s: binding_array<sampler, 2>) -> vec4<f32> {
        return textureSample(t[1], s[0], vec2(0.5));
    }

    @fragment
    fn main() -> @location(0) vec4<f32> {
        return sample(textures, samplers);
    }
";

const SUB_OBJECT_ARGUMENT: &str = "
    fn store(p: ptr<function, u32>) {
        *p = 1u;
    }

    fn forward(p: ptr<function, u32>) {
        store(p);
    }

    @compute @workgroup_size(1)
    fn main() {
        var a = array<u32, 4>();
        forward(&a[1]);
    }
";

const RUNTIME_SIZED_POINTER: &str = "
    @group(0) @binding(0) var<storage, read_write> buf: array<u32>;

    fn len(p: ptr<storage, array<u32>, read_write>) -> u32 {
        return arrayLength(p);
    }

    @compute @workgroup_size(1)
    fn main() {
        buf[0] = len(&buf);
    }
";

#[test]
fn never_changes() {
    let (module, _) = parse(STORAGE_POINTER);
    let inlined = inline(&module, InlineStrategy::Never);
    assert!(inlined.is_none());
}

#[test]
fn storage_pointer() {
    let (module, _) = parse(STORAGE_POINTER);
    for target in [InlineTarget::Spirv, InlineTarget::Hlsl, InlineTarget::Glsl] {
        let inlined = inline(&module, InlineStrategy::OnlyNeeded(target)).unwrap();
        assert!(inlined.functions.is_empty(), "{target:?}");
        assert!(!has_call(&inlined.entry_points[0].function.body));
    }
    let inlined = inline(&module, InlineStrategy::OnlyNeeded(InlineTarget::Msl));
    assert!(inlined.is_none());
}

#[test]
fn immediate_pointer() {
    let (module, _) = parse(IMMEDIATE_POINTER);
    for target in [InlineTarget::Spirv, InlineTarget::Hlsl, InlineTarget::Glsl] {
        let inlined = inline(&module, InlineStrategy::OnlyNeeded(target)).unwrap();
        assert!(inlined.functions.is_empty(), "{target:?}");
        assert!(!has_call(&inlined.entry_points[0].function.body));
    }
    let inlined = inline(&module, InlineStrategy::OnlyNeeded(InlineTarget::Msl));
    assert!(inlined.is_none());
}

#[test]
fn task_payload_pointer() {
    let (module, _) = parse(TASK_PAYLOAD_POINTER);
    for target in [InlineTarget::Spirv, InlineTarget::Hlsl, InlineTarget::Glsl] {
        let inlined = inline(&module, InlineStrategy::OnlyNeeded(target)).unwrap();
        assert!(inlined.functions.is_empty(), "{target:?}");
        assert!(!has_call(&inlined.entry_points[0].function.body));
    }
    assert!(inline(&module, InlineStrategy::OnlyNeeded(InlineTarget::Msl)).is_none());
}

#[test]
fn binding_array_inlined_for_every_target() {
    {
        let (module, _) = parse(BINDING_ARRAY);
        for target in [
            InlineTarget::Spirv,
            InlineTarget::Hlsl,
            InlineTarget::Glsl,
            InlineTarget::Msl,
        ] {
            let inlined = inline(&module, InlineStrategy::OnlyNeeded(target)).unwrap();
            assert!(inlined.functions.is_empty(), "{target:?}");
            assert!(!has_call(&inlined.entry_points[0].function.body));
        }
    }
}

#[test]
fn sub_object_argument_passed_through() {
    let (module, _) = parse(SUB_OBJECT_ARGUMENT);
    let inlined = inline(&module, InlineStrategy::OnlyNeeded(InlineTarget::Spirv)).unwrap();
    assert!(inlined.functions.is_empty());
    assert!(!has_call(&inlined.entry_points[0].function.body));

    for target in [InlineTarget::Hlsl, InlineTarget::Glsl, InlineTarget::Msl] {
        let inlined = inline(&module, InlineStrategy::OnlyNeeded(target));
        assert!(inlined.is_none(), "{target:?}");
    }
}

#[test]
fn runtime_sized_pointer() {
    let (module, _) = parse(RUNTIME_SIZED_POINTER);
    let inlined = inline(&module, InlineStrategy::OnlyNeeded(InlineTarget::Msl)).unwrap();
    assert!(inlined.functions.is_empty());
}

#[test]
fn uncalled_function_is_removed() {
    let (module, _) = parse(
        "
        fn store(p: ptr<workgroup, u32>) {
            *p = 1u;
        }

        @compute @workgroup_size(1)
        fn main() {}
        ",
    );
    let inlined = inline(&module, InlineStrategy::OnlyNeeded(InlineTarget::Spirv)).unwrap();
    assert!(inlined.functions.is_empty());
}

#[test]
fn all() {
    let (module, _) = parse(
        "
        fn f(x: i32) -> i32 {
            if x > 0 {
                return 1;
            }
            for (var i = 0; i < x; i++) {
                if i == 3 {
                    return i;
                }
            }
            switch x {
                case 1: {
                    return 2;
                }
                default: {}
            }
            return 0;
        }

        fn g(x: i32) -> i32 {
            var y = vec2(1, 2);
            y.x += f(x);
            return y.x + f(y.y);
        }

        @compute @workgroup_size(1)
        fn main() {
            for (var i = 0; i < 2; i++) {
                _ = g(i);
            }
        }
        ",
    );
    let inlined = inline(&module, InlineStrategy::All).unwrap();
    assert!(inlined.functions.is_empty());
    assert!(!has_call(&inlined.entry_points[0].function.body));
}

/// Backends inline the calls they can't express themselves.
#[cfg(spv_out)]
#[test]
fn spirv_inlines_what_it_needs() {
    let options = naga::back::spv::Options::default();
    for source in [
        STORAGE_POINTER,
        IMMEDIATE_POINTER,
        TASK_PAYLOAD_POINTER,
        BINDING_ARRAY,
        SUB_OBJECT_ARGUMENT,
    ] {
        let (module, _) = parse(source);
        let result = naga::back::spv::write_vec(&module, &options, None);
        assert!(
            !matches!(result, Err(naga::back::spv::Error::NeedsInline(_))),
            "{result:?}"
        );
    }
}

#[cfg(hlsl_out)]
#[test]
fn hlsl_inlines_what_it_needs() {
    let (module, _) = parse(STORAGE_POINTER);
    let options = naga::back::hlsl::Options::default();
    let pipeline_options = naga::back::hlsl::PipelineOptions::default();
    let mut output = String::new();
    naga::back::hlsl::Writer::new(&mut output, &options, &pipeline_options)
        .write(&module, None)
        .unwrap();
}

#[cfg(msl_out)]
#[test]
fn msl_inlines_what_it_needs() {
    for source in [RUNTIME_SIZED_POINTER, BINDING_ARRAY] {
        let (module, _) = parse(source);
        let result = naga::back::msl::write_string(
            &module,
            &naga::back::msl::Options::default(),
            &naga::back::msl::PipelineOptions::default(),
        );
        assert!(
            !matches!(result, Err(naga::back::msl::Error::NeedsInline(_))),
            "{result:?}"
        );
    }
}

/// `requires unrestricted_pointer_parameters;` only documents the use of the
/// language extension, which is available without it. Both forms of a module
/// validate, and each backend writes the same output for them.
#[test]
fn requires_unrestricted_pointer_parameters_is_optional() {
    const BODY: &str = "
        struct R { m: mat2x2<f32> }
        struct Outer { inner: R }

        @group(0) @binding(0) var<storage, read_write> buffer: array<u32, 4>;

        fn f(r: ptr<function, R>) {
            (*r).m = mat2x2<f32>();
        }

        fn g(p: ptr<storage, u32, read_write>) {
            *p = 1u;
        }

        @compute @workgroup_size(1)
        fn main() {
            var n: Outer;
            f(&n.inner);
            g(&buffer[1]);
        }
    ";
    let with = parse(&format!(
        "requires unrestricted_pointer_parameters;\n{BODY}"
    ));
    let without = parse(BODY);
    let outputs = |(module, _): &(Module, ModuleInfo)| {
        let mut outputs: Vec<String> = Vec::new();
        #[cfg(spv_out)]
        outputs.push(format!(
            "{:?}",
            naga::back::spv::write_vec(module, &Default::default(), None).unwrap()
        ));
        #[cfg(hlsl_out)]
        {
            let mut output = String::new();
            naga::back::hlsl::Writer::new(&mut output, &Default::default(), &Default::default())
                .write(module, None)
                .unwrap();
            outputs.push(output);
        }
        #[cfg(msl_out)]
        outputs.push(
            naga::back::msl::write_string(module, &Default::default(), &Default::default())
                .unwrap()
                .0,
        );
        #[cfg(glsl_out)]
        {
            let mut output = String::new();
            let pipeline_options = naga::back::glsl::PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: "main".into(),
                multiview: None,
            };
            naga::back::glsl::Writer::new(
                &mut output,
                &Default::default(),
                &pipeline_options,
                Default::default(),
            )
            .write(module)
            .unwrap();
            outputs.push(output);
        }
        outputs
    };
    let with = outputs(&with);
    assert!(!with.is_empty(), "no backends enabled");
    assert_eq!(with, outputs(&without));
}
