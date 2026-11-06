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

/// Return `true` if `process` returned the module it was given.
fn unchanged(inlined: &Module, module: &Module) -> bool {
    core::ptr::eq(inlined, module)
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
fn never_borrows() {
    let (module, info) = parse(STORAGE_POINTER);
    let (inlined, _) = naga::inline::process(&module, &info, InlineStrategy::Never).unwrap();
    assert!(unchanged(&inlined, &module));
}

#[test]
fn storage_pointer() {
    let (module, info) = parse(STORAGE_POINTER);
    for target in [InlineTarget::Spirv, InlineTarget::Hlsl, InlineTarget::Glsl] {
        let (inlined, _) =
            naga::inline::process(&module, &info, InlineStrategy::OnlyNeeded(target)).unwrap();
        assert!(inlined.functions.is_empty(), "{target:?}");
        assert!(!has_call(&inlined.entry_points[0].function.body));
    }
    let (inlined, _) = naga::inline::process(
        &module,
        &info,
        InlineStrategy::OnlyNeeded(InlineTarget::Msl),
    )
    .unwrap();
    assert!(unchanged(&inlined, &module));
}

#[test]
fn sub_object_argument_passed_through() {
    let (module, info) = parse(SUB_OBJECT_ARGUMENT);
    let (inlined, _) = naga::inline::process(
        &module,
        &info,
        InlineStrategy::OnlyNeeded(InlineTarget::Spirv),
    )
    .unwrap();
    assert!(inlined.functions.is_empty());
    assert!(!has_call(&inlined.entry_points[0].function.body));

    for target in [InlineTarget::Hlsl, InlineTarget::Glsl, InlineTarget::Msl] {
        let (inlined, _) =
            naga::inline::process(&module, &info, InlineStrategy::OnlyNeeded(target)).unwrap();
        assert!(unchanged(&inlined, &module), "{target:?}");
    }
}

#[test]
fn runtime_sized_pointer() {
    let (module, info) = parse(RUNTIME_SIZED_POINTER);
    let (inlined, _) = naga::inline::process(
        &module,
        &info,
        InlineStrategy::OnlyNeeded(InlineTarget::Msl),
    )
    .unwrap();
    assert!(inlined.functions.is_empty());
}

#[test]
fn uncalled_function_is_removed() {
    let (module, info) = parse(
        "
        fn store(p: ptr<workgroup, u32>) {
            *p = 1u;
        }

        @compute @workgroup_size(1)
        fn main() {}
        ",
    );
    let (inlined, _) = naga::inline::process(
        &module,
        &info,
        InlineStrategy::OnlyNeeded(InlineTarget::Spirv),
    )
    .unwrap();
    assert!(inlined.functions.is_empty());
}

#[test]
fn all() {
    let (module, info) = parse(
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
    let (inlined, _) = naga::inline::process(&module, &info, InlineStrategy::All).unwrap();
    assert!(inlined.functions.is_empty());
    assert!(!has_call(&inlined.entry_points[0].function.body));
}

#[cfg(spv_out)]
#[test]
fn spirv_needs_inline() {
    let options = naga::back::spv::Options::default();
    for source in [STORAGE_POINTER, SUB_OBJECT_ARGUMENT] {
        let (module, info) = parse(source);
        let error = naga::back::spv::write_vec(&module, &info, &options, None).unwrap_err();
        assert!(
            matches!(error, naga::back::spv::Error::NeedsInline(_)),
            "{error}"
        );
    }
}

#[cfg(hlsl_out)]
#[test]
fn hlsl_needs_inline() {
    let (module, info) = parse(STORAGE_POINTER);
    let options = naga::back::hlsl::Options::default();
    let pipeline_options = naga::back::hlsl::PipelineOptions::default();
    let mut output = String::new();
    let error = naga::back::hlsl::Writer::new(&mut output, &options, &pipeline_options)
        .write(&module, &info, None)
        .unwrap_err();
    assert!(
        matches!(error, naga::back::hlsl::Error::NeedsInline(_)),
        "{error}"
    );
}

#[cfg(msl_out)]
#[test]
fn msl_needs_inline() {
    let (module, info) = parse(RUNTIME_SIZED_POINTER);
    let error = naga::back::msl::write_string(
        &module,
        &info,
        &naga::back::msl::Options::default(),
        &naga::back::msl::PipelineOptions::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error, naga::back::msl::Error::NeedsInline(_)),
        "{error}"
    );
}
