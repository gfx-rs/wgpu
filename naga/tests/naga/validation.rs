//! Tests of the module validator.
//!
//! There are also some validation tests in [`wgsl_errors`](super::wgsl_errors).

use naga::{
    ir::{self, Expression, Function, Module, Scalar},
    valid::{self, Capabilities, ModuleInfo, ValidationFlags},
};

#[derive(Default)]
struct TestSpanGenerator(u32);

impl TestSpanGenerator {
    fn next(&mut self) -> naga::Span {
        let span = naga::Span::new(self.0, self.0 + 1);
        self.0 += 1;
        span
    }
}

#[track_caller]
fn expect_validation_error_impl<I: IntoIterator<Item = naga::Span>>(
    module: &Module,
    validation_flags: valid::ValidationFlags,
    capabilities: valid::Capabilities,
    spans: Option<I>,
) -> naga::valid::ValidationError {
    let err = valid::Validator::new(validation_flags, capabilities)
        .validate(module)
        .expect_err("module should be invalid");

    if let Some(expected_spans_iter) = spans {
        let actual_spans = err.spans().map(|sctx| sctx.0).collect::<Vec<_>>();
        let expected_spans = expected_spans_iter.into_iter().collect::<Vec<_>>();
        assert_eq!(
            actual_spans, expected_spans,
            "expected error spans to be {expected_spans:?}, got {actual_spans:?}",
        );
    }

    err.into_inner()
}

/// Validate `module` with the given `validation_flags` and `capabilities`.
///
/// Panics if validation succeeds or fails with an error not associated with
/// `span`. Otherwise, returns the validation error.
///
/// Note that only the span is checked, not the associated context string.
#[track_caller]
fn expect_validation_error_with_span(
    module: &Module,
    validation_flags: valid::ValidationFlags,
    capabilities: valid::Capabilities,
    span: naga::Span,
) -> naga::valid::ValidationError {
    expect_validation_error_impl(
        module,
        validation_flags,
        capabilities,
        Some(core::iter::once(span)),
    )
}

/// Validation should fail if `AtomicResult` expressions are not
/// populated by `Atomic` statements.
#[test]
fn populate_atomic_result() {
    use naga::{Module, Type, TypeInner};

    /// Different variants of the test case that we want to exercise.
    enum Variant {
        /// An `AtomicResult` expression with an `Atomic` statement
        /// that populates it: valid.
        Atomic,

        /// An `AtomicResult` expression visited by an `Emit`
        /// statement: invalid.
        Emit,

        /// An `AtomicResult` expression visited by no statement at
        /// all: invalid
        None,
    }

    // Looking at uses of `variant` should make it easy to identify
    // the differences between the test cases.
    fn try_variant(
        variant: Variant,
    ) -> Result<ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let span = naga::Span::default();
        let mut module = Module::default();
        let ty_u32 = module.types.insert(
            Type {
                name: Some("u32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            span,
        );
        let ty_atomic_u32 = module.types.insert(
            Type {
                name: Some("atomic<u32>".into()),
                inner: TypeInner::Atomic(Scalar::U32),
            },
            span,
        );
        let var_atomic = module.global_variables.append(
            naga::GlobalVariable {
                name: Some("atomic_global".into()),
                space: naga::AddressSpace::WorkGroup,
                binding: None,
                ty: ty_atomic_u32,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            span,
        );

        let mut fun = Function::default();
        let ex_global = fun
            .expressions
            .append(Expression::GlobalVariable(var_atomic), span);
        let ex_42 = fun
            .expressions
            .append(Expression::Literal(naga::Literal::U32(42)), span);
        let ex_result = fun.expressions.append(
            Expression::AtomicResult {
                ty: ty_u32,
                comparison: false,
            },
            span,
        );

        match variant {
            Variant::Atomic => {
                fun.body.push(
                    naga::Statement::Atomic {
                        pointer: ex_global,
                        fun: naga::AtomicFunction::Add,
                        value: ex_42,
                        result: Some(ex_result),
                    },
                    span,
                );
            }
            Variant::Emit => {
                fun.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(ex_result, ex_result)),
                    span,
                );
            }
            Variant::None => {}
        }

        module.functions.append(fun, span);

        valid::Validator::new(
            valid::ValidationFlags::default(),
            valid::Capabilities::all(),
        )
        .validate(&module)
    }

    try_variant(Variant::Atomic).expect("module should validate");
    assert!(try_variant(Variant::Emit).is_err());
    assert!(try_variant(Variant::None).is_err());
}

#[test]
fn populate_call_result() {
    use naga::{Module, Type, TypeInner};

    /// Different variants of the test case that we want to exercise.
    enum Variant {
        /// A `CallResult` expression with an `Call` statement that
        /// populates it: valid.
        Call,

        /// A `CallResult` expression visited by an `Emit` statement:
        /// invalid.
        Emit,

        /// A `CallResult` expression visited by no statement at all:
        /// invalid
        None,
    }

    // Looking at uses of `variant` should make it easy to identify
    // the differences between the test cases.
    fn try_variant(
        variant: Variant,
    ) -> Result<ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let span = naga::Span::default();
        let mut module = Module::default();
        let ty_u32 = module.types.insert(
            Type {
                name: Some("u32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            span,
        );

        let mut fun_callee = Function {
            result: Some(naga::FunctionResult {
                ty: ty_u32,
                binding: None,
            }),
            ..Function::default()
        };
        let ex_42 = fun_callee
            .expressions
            .append(Expression::Literal(naga::Literal::U32(42)), span);
        fun_callee
            .body
            .push(naga::Statement::Return { value: Some(ex_42) }, span);
        let fun_callee = module.functions.append(fun_callee, span);

        let mut fun_caller = Function::default();
        let ex_result = fun_caller
            .expressions
            .append(Expression::CallResult(fun_callee), span);

        match variant {
            Variant::Call => {
                fun_caller.body.push(
                    naga::Statement::Call {
                        function: fun_callee,
                        arguments: vec![],
                        result: Some(ex_result),
                    },
                    span,
                );
            }
            Variant::Emit => {
                fun_caller.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(ex_result, ex_result)),
                    span,
                );
            }
            Variant::None => {}
        }

        module.functions.append(fun_caller, span);

        valid::Validator::new(
            valid::ValidationFlags::default(),
            valid::Capabilities::all(),
        )
        .validate(&module)
    }

    try_variant(Variant::Call).expect("should validate");
    assert!(try_variant(Variant::Emit).is_err());
    assert!(try_variant(Variant::None).is_err());
}

#[test]
fn emit_workgroup_uniform_load_result() {
    use naga::{Module, Type, TypeInner};

    // We want to ensure that the *only* problem with the code is the
    // use of an `Emit` statement instead of an `Atomic` statement. So
    // validate two versions of the module varying only in that
    // aspect.
    //
    // Looking at uses of the `wg_load` makes it easy to identify the
    // differences between the two variants.
    fn variant(wg_load: bool) -> Result<ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let span = naga::Span::default();
        let mut module = Module::default();
        let ty_u32 = module.types.insert(
            Type {
                name: Some("u32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            span,
        );
        let var_workgroup = module.global_variables.append(
            naga::GlobalVariable {
                name: Some("workgroup_global".into()),
                space: naga::AddressSpace::WorkGroup,
                binding: None,
                ty: ty_u32,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            span,
        );

        let mut fun = Function::default();
        let ex_global = fun
            .expressions
            .append(Expression::GlobalVariable(var_workgroup), span);
        let ex_result = fun
            .expressions
            .append(Expression::WorkGroupUniformLoadResult { ty: ty_u32 }, span);

        if wg_load {
            fun.body.push(
                naga::Statement::WorkGroupUniformLoad {
                    pointer: ex_global,
                    result: ex_result,
                },
                span,
            );
        } else {
            fun.body.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(ex_result, ex_result)),
                span,
            );
        }

        module.functions.append(fun, span);

        valid::Validator::new(
            valid::ValidationFlags::default(),
            valid::Capabilities::all(),
        )
        .validate(&module)
    }

    variant(true).expect("module should validate");
    assert!(variant(false).is_err());
}

#[test]
fn builtin_cross_product_args() {
    use naga::{MathFunction, Module, Type, TypeInner, VectorSize};

    // We want to ensure that the *only* problem with the code is the
    // arity of the call, or the size of the vectors passed to
    // `cross`. So validate different versions of the module varying
    // only in those aspects.
    //
    // Looking at uses of `size` and `arity` makes it easy to identify
    // the differences between the variants.
    fn variant(
        size: VectorSize,
        arity: usize,
    ) -> Result<ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let span = naga::Span::default();
        let mut module = Module::default();
        let ty_vec3f = module.types.insert(
            Type {
                name: Some("vecnf".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::F32,
                },
            },
            span,
        );
        let ty_vecnf = module.types.insert(
            Type {
                name: Some("vecnf".into()),
                inner: TypeInner::Vector {
                    size,
                    scalar: Scalar::F32,
                },
            },
            span,
        );

        let mut fun = Function {
            result: Some(naga::ir::FunctionResult {
                ty: ty_vec3f,
                binding: None,
            }),
            ..Function::default()
        };
        let ex_zero = fun
            .expressions
            .append(Expression::ZeroValue(ty_vecnf), span);
        let ex_cross = fun.expressions.append(
            Expression::Math {
                fun: MathFunction::Cross,
                arg: ex_zero,
                arg1: (arity >= 2).then_some(ex_zero),
                arg2: (arity >= 3).then_some(ex_zero),
                arg3: (arity >= 4).then_some(ex_zero),
            },
            span,
        );

        fun.body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(ex_cross, ex_cross)),
            span,
        );
        fun.body.push(
            naga::Statement::Return {
                value: Some(ex_cross),
            },
            span,
        );

        module.functions.append(fun, span);

        valid::Validator::new(
            valid::ValidationFlags::default(),
            valid::Capabilities::all(),
        )
        .validate(&module)
    }

    assert!(variant(VectorSize::Bi, 2).is_err());

    assert!(variant(VectorSize::Tri, 1).is_err());
    variant(VectorSize::Tri, 2).expect("module should validate");
    assert!(variant(VectorSize::Tri, 3).is_err());
    assert!(variant(VectorSize::Tri, 4).is_err());

    assert!(variant(VectorSize::Quad, 2).is_err());
}

#[test]
fn incompatible_interpolation_and_sampling_types() {
    use dummy_interpolation_shader::DummyInterpolationShader;

    // NOTE: Things we expect to actually compile are in the `interpolate` snapshot test.
    use itertools::Itertools;

    let invalid_shader_module = |interpolation_and_sampling| {
        let (interpolation, sampling) = interpolation_and_sampling;

        let valid = matches!(
            (interpolation, sampling),
            (_, None)
                | (
                    naga::Interpolation::Perspective | naga::Interpolation::Linear,
                    Some(
                        naga::Sampling::Center | naga::Sampling::Centroid | naga::Sampling::Sample
                    ),
                )
                | (
                    naga::Interpolation::Flat,
                    Some(naga::Sampling::First | naga::Sampling::Either)
                )
                | (naga::Interpolation::PerVertex, Some(naga::Sampling::Center))
        );

        if valid {
            None
        } else {
            let DummyInterpolationShader {
                source,
                module,
                interpolate_attr,
                entry_point: _,
            } = DummyInterpolationShader::new(interpolation, sampling);
            Some((
                source,
                module,
                interpolation,
                sampling.expect("default interpolation sampling should be valid"),
                interpolate_attr,
            ))
        }
    };

    // Note: `PerVertex` is excluded here because its invalid sampling combinations produce
    // `VaryingError::InvalidPerVertexSampling` rather than `InvalidInterpolationSamplingCombination`.
    let invalid_cases = [
        naga::Interpolation::Flat,
        naga::Interpolation::Linear,
        naga::Interpolation::Perspective,
    ]
    .into_iter()
    .cartesian_product(
        [
            naga::Sampling::Either,
            naga::Sampling::First,
            naga::Sampling::Sample,
            naga::Sampling::Center,
            naga::Sampling::Centroid,
        ]
        .into_iter()
        .map(Some)
        .chain([None]),
    )
    .filter_map(invalid_shader_module);

    for (invalid_source, invalid_module, interpolation, sampling, interpolate_attr) in invalid_cases
    {
        let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
            .validate(&invalid_module)
            .expect_err(&format!(
                "module should be invalid for {interpolate_attr:?}"
            ));
        assert!(dbg!(err.emit_to_string(&invalid_source)).contains(&dbg!(
            naga::valid::VaryingError::InvalidInterpolationSamplingCombination {
                interpolation,
                sampling,
            }
            .to_string()
        )),);
    }
}

#[test]
fn no_flat_first_in_glsl() {
    use dummy_interpolation_shader::DummyInterpolationShader;

    let DummyInterpolationShader {
        source: _,
        module,
        interpolate_attr,
        entry_point,
    } = DummyInterpolationShader::new(naga::Interpolation::Flat, Some(naga::Sampling::First));

    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let module_info = validator.validate(&module).unwrap();

    let options = Default::default();
    let pipeline_options = naga::back::glsl::PipelineOptions {
        shader_stage: naga::ShaderStage::Fragment,
        entry_point: entry_point.to_owned(),
        multiview: None,
    };
    let mut glsl_writer = naga::back::glsl::Writer::new(
        String::new(),
        &module,
        &module_info,
        &options,
        &pipeline_options,
        Default::default(),
    )
    .unwrap();

    let err = glsl_writer.write().expect_err(&format!(
        "`{interpolate_attr}` should fail backend validation"
    ));

    assert!(matches!(
        err,
        naga::back::glsl::Error::FirstSamplingNotSupported
    ));
}

mod dummy_interpolation_shader {
    pub struct DummyInterpolationShader {
        pub source: String,
        pub module: naga::Module,
        pub interpolate_attr: String,
        pub entry_point: &'static str,
    }

    impl DummyInterpolationShader {
        pub fn new(interpolation: naga::Interpolation, sampling: Option<naga::Sampling>) -> Self {
            // NOTE: If you have to add variants below, make sure to add them to the
            // `cartesian_product`'d combinations in tests around here!
            let interpolation_str = match interpolation {
                naga::Interpolation::Flat => "flat",
                naga::Interpolation::Linear => "linear",
                naga::Interpolation::Perspective => "perspective",
                naga::Interpolation::PerVertex => "per_vertex",
            };
            let sampling_str = match sampling {
                None => String::new(),
                Some(sampling) => format!(
                    ", {}",
                    match sampling {
                        naga::Sampling::First => "first",
                        naga::Sampling::Either => "either",
                        naga::Sampling::Center => "center",
                        naga::Sampling::Centroid => "centroid",
                        naga::Sampling::Sample => "sample",
                    }
                ),
            };
            let member_type = match interpolation {
                naga::Interpolation::Perspective | naga::Interpolation::Linear => "f32",
                naga::Interpolation::Flat => "u32",
                naga::Interpolation::PerVertex => "array<u32, 3>",
            };

            let enable_extension = if interpolation == naga::Interpolation::PerVertex {
                "enable wgpu_per_vertex;\n\n"
            } else {
                ""
            };
            let interpolate_attr = format!("@interpolate({interpolation_str}{sampling_str})");
            let source = format!(
                "\
                {enable_extension}
                struct VertexOutput {{
    @location(0) {interpolate_attr} member: {member_type},
}}

@fragment
fn main(input: VertexOutput) {{
    // ...
}}
"
            );
            let module = naga::front::wgsl::parse_str(&source).unwrap();

            Self {
                source,
                module,
                interpolate_attr,
                entry_point: "main",
            }
        }
    }
}

struct BindingArrayFixture {
    module: Module,
    span: naga::Span,
    ty_u32: naga::Handle<naga::Type>,
    ty_array: naga::Handle<naga::Type>,
    ty_struct: naga::Handle<naga::Type>,
    validator: naga::valid::Validator,
}

impl BindingArrayFixture {
    fn new() -> Self {
        let mut module = Module::default();
        let span = naga::Span::default();
        let ty_u32 = module.types.insert(
            naga::Type {
                name: Some("u32".into()),
                inner: naga::TypeInner::Scalar(naga::Scalar::U32),
            },
            span,
        );
        let ty_array = module.types.insert(
            naga::Type {
                name: Some("array<u32, 10>".into()),
                inner: naga::TypeInner::Array {
                    base: ty_u32,
                    size: naga::ArraySize::Constant(core::num::NonZeroU32::new(10).unwrap()),
                    stride: 4,
                },
            },
            span,
        );
        let ty_struct = module.types.insert(
            naga::Type {
                name: Some("S".into()),
                inner: naga::TypeInner::Struct {
                    members: vec![naga::StructMember {
                        name: Some("m".into()),
                        ty: ty_u32,
                        binding: None,
                        offset: 0,
                    }],
                    span: 4,
                },
            },
            span,
        );
        let validator = naga::valid::Validator::new(Default::default(), Default::default());
        BindingArrayFixture {
            module,
            span,
            ty_u32,
            ty_array,
            ty_struct,
            validator,
        }
    }
}

#[test]
fn binding_arrays_hold_structs() {
    let mut t = BindingArrayFixture::new();
    let _binding_array = t.module.types.insert(
        naga::Type {
            name: Some("binding_array_of_struct".into()),
            inner: naga::TypeInner::BindingArray {
                base: t.ty_struct,
                size: naga::ArraySize::Dynamic,
            },
        },
        t.span,
    );

    assert!(t.validator.validate(&t.module).is_ok());
}

#[test]
fn binding_arrays_cannot_hold_arrays() {
    let mut t = BindingArrayFixture::new();
    let _binding_array = t.module.types.insert(
        naga::Type {
            name: Some("binding_array_of_array".into()),
            inner: naga::TypeInner::BindingArray {
                base: t.ty_array,
                size: naga::ArraySize::Dynamic,
            },
        },
        t.span,
    );

    assert!(t.validator.validate(&t.module).is_err());
}

#[test]
fn binding_arrays_cannot_hold_scalars() {
    let mut t = BindingArrayFixture::new();
    let _binding_array = t.module.types.insert(
        naga::Type {
            name: Some("binding_array_of_scalar".into()),
            inner: naga::TypeInner::BindingArray {
                base: t.ty_u32,
                size: naga::ArraySize::Dynamic,
            },
        },
        t.span,
    );

    assert!(t.validator.validate(&t.module).is_err());
}

#[test]
fn validation_error_messages() {
    let cases = [(
        r#"@group(0) @binding(0) var my_sampler: sampler;

                fn foo(tex: texture_2d<f32>) -> vec4<f32> {
                    return textureSampleLevel(tex, my_sampler, vec2f(0, 0), 0.0);
                }

                fn main() {
                    foo();
                }
            "#,
        "\
error: Function [1] 'main' is invalid
  ┌─ wgsl:7:17
  │\x20\x20
7 │ ╭                 fn main() {
8 │ │                     foo();
  │ │                     ^^^^^ invalid function call
9 │ │                 }
  │ ╰─────────────────^ naga::ir::Function [1]
  │\x20\x20
  = Call to [0] is invalid
  = Requires 1 arguments, but 0 are provided

",
    )];

    for (source, expected_err) in cases {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
            .validate(&module)
            .expect_err("module should be invalid");
        println!("{}", err.emit_to_string(source));
        assert_eq!(err.emit_to_string(source), expected_err);
    }
}

#[test]
fn bad_texture_dimensions_level() {
    fn validate(level: &str) -> Result<ModuleInfo, naga::valid::ValidationError> {
        let source = format!(
            r#"
            @group(0) @binding(0)
            var t: texture_1d<f32>;
            fn f() -> u32 {{
              return textureDimensions(t, {level});
            }}
       "#
        );
        let module = naga::front::wgsl::parse_str(&source).expect("module should parse");
        valid::Validator::new(Default::default(), valid::Capabilities::all())
            .validate(&module)
            .map_err(|err| err.into_inner()) // discard spans
    }

    fn is_bad_level_error(result: Result<ModuleInfo, naga::valid::ValidationError>) -> bool {
        matches!(
            result,
            Err(naga::valid::ValidationError::Function {
                handle: _,
                name: _,
                source: naga::valid::FunctionError::Expression {
                    handle: _,
                    source: naga::valid::ExpressionError::InvalidImageOtherIndexType(_),
                },
            })
        )
    }

    assert!(is_bad_level_error(validate("true")));
    assert!(is_bad_level_error(validate("1.0")));
    assert!(validate("1").is_ok());
    assert!(validate("1i").is_ok());
    assert!(validate("1").is_ok());
}

// Adds IR for `override len: u32` and the type `array<u32, len>`.
fn make_override_array(module: &mut ir::Module) -> naga::Handle<ir::Type> {
    let span = naga::Span::default();

    let ty_u32 = module.types.insert(
        ir::Type {
            name: Some("u32".into()),
            inner: ir::TypeInner::Scalar(Scalar::U32),
        },
        span,
    );

    let len = module.overrides.append(
        ir::Override {
            name: Some("len".into()),
            id: None,
            ty: ty_u32,
            init: None,
        },
        span,
    );

    module.types.insert(
        ir::Type {
            name: Some("array<u32, len>".into()),
            inner: ir::TypeInner::Array {
                base: ty_u32,
                size: ir::ArraySize::Pending(len),
                stride: 4,
            },
        },
        span,
    )
}

// Adds IR for type `array<u32>`.
fn make_runtime_array(module: &mut ir::Module) -> naga::Handle<ir::Type> {
    let span = naga::Span::default();

    let ty_u32 = module.types.insert(
        ir::Type {
            name: Some("u32".into()),
            inner: ir::TypeInner::Scalar(Scalar::U32),
        },
        span,
    );

    module.types.insert(
        ir::Type {
            name: Some("array<u32>".into()),
            inner: ir::TypeInner::Array {
                base: ty_u32,
                size: ir::ArraySize::Dynamic,
                stride: 4,
            },
        },
        span,
    )
}

// Adds a local variable `var x: ty = ty();`.
fn make_zero_value_local_variable(fun: &mut Function, ty: naga::Handle<ir::Type>) {
    let span = naga::Span::default();

    let ex_zero = fun.expressions.append(Expression::ZeroValue(ty), span);

    fun.local_variables.append(
        naga::LocalVariable {
            name: Some("x".into()),
            ty,
            init: Some(ex_zero),
        },
        span,
    );
}

#[test]
fn invalid_local_var_override_sized_array() {
    // Similar to a test in wgsl_errors::invalid_local_vars.
    // ```
    // override len: u32;
    // var<workgroup> arr: array<u32, len>;
    // fn f() {
    //     var x: array<u32, len> = arr;
    // }
    // ```
    let span = naga::Span::default();
    let mut module = ir::Module::default();

    let ty_array = make_override_array(&mut module);

    let var_arr = module.global_variables.append(
        naga::GlobalVariable {
            name: Some("arr".into()),
            space: naga::AddressSpace::WorkGroup,
            binding: None,
            ty: ty_array,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        span,
    );

    let mut fun = Function {
        name: Some("f".into()),
        ..Default::default()
    };

    let ex_global = fun
        .expressions
        .append(Expression::GlobalVariable(var_arr), span);
    let ex_load = fun
        .expressions
        .append(Expression::Load { pointer: ex_global }, span);

    fun.local_variables.append(
        naga::LocalVariable {
            name: Some("x".into()),
            ty: ty_array,
            init: Some(ex_load),
        },
        span,
    );

    module.functions.append(fun, span);

    let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .expect_err("module should be invalid")
        .into_inner();

    assert!(matches!(
        err,
        valid::ValidationError::Function {
            source: valid::FunctionError::LocalVariable {
                name: local_var_name,
                source: valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        } if local_var_name == "x"
    ));
}

#[test]
fn invalid_zero_value_runtime_array() {
    // Similar to a test in wgsl_errors::invalid_zero_value_constructors.
    // ```
    // fn main() {
    //     var x = array<u32>();
    // }
    // ```
    let span = naga::Span::default();
    let mut module = ir::Module::default();

    let ty_array = make_runtime_array(&mut module);

    let mut fun = Function {
        name: Some("f".into()),
        ..Default::default()
    };

    make_zero_value_local_variable(&mut fun, ty_array);

    module.functions.append(fun, span);

    let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .expect_err("module should be invalid")
        .into_inner();

    assert!(matches!(
        err,
        valid::ValidationError::Function {
            source: valid::FunctionError::LocalVariable {
                name: local_var_name,
                source: valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        } if local_var_name == "x"
    ));
}

#[test]
fn invalid_zero_value_override_array() {
    // Similar to a test in wgsl_errors::invalid_zero_value_constructors.
    // ```
    // override len: u32;
    // fn main() {
    //     var x = array<u32, len>();
    // }
    // ```
    let span = naga::Span::default();
    let mut module = ir::Module::default();

    let ty_array = make_override_array(&mut module);

    let mut fun = Function {
        name: Some("f".into()),
        ..Default::default()
    };

    make_zero_value_local_variable(&mut fun, ty_array);

    module.functions.append(fun, span);

    let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .expect_err("module should be invalid")
        .into_inner();

    assert!(matches!(
        err,
        valid::ValidationError::Function {
            source: valid::FunctionError::LocalVariable {
                name: local_var_name,
                source: valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        } if local_var_name == "x"
    ));
}

#[test]
fn invalid_zero_value_texture() {
    // Similar to a test in wgsl_errors::invalid_zero_value_constructors.
    // ```
    // fn main() {
    //     var x = texture_2d<f32>();
    // }
    // ```
    use naga::{ImageClass, ImageDimension, Module, Type, TypeInner};

    let span = naga::Span::default();
    let mut module = Module::default();

    let ty_texture = module.types.insert(
        Type {
            name: Some("texture_2d<f32>".into()),
            inner: TypeInner::Image {
                dim: ImageDimension::D2,
                arrayed: false,
                class: ImageClass::Sampled {
                    kind: naga::ScalarKind::Float,
                    multi: false,
                },
            },
        },
        span,
    );

    let mut fun = Function {
        name: Some("f".into()),
        ..Default::default()
    };

    make_zero_value_local_variable(&mut fun, ty_texture);

    module.functions.append(fun, span);

    let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .expect_err("module should be invalid")
        .into_inner();

    assert!(matches!(
        err,
        valid::ValidationError::Function {
            source: valid::FunctionError::LocalVariable {
                name: local_var_name,
                source: valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        } if local_var_name == "x"
    ));
}

/// Test for non-zero-value runtime-sized array constructor.
#[test]
fn invalid_constructor_runtime_array() {
    // Similar to a test in wgsl_errors::invalid_zero_value_constructors.
    // ```
    // fn main() {
    //     var x = array<u32>(0, 1, 2);
    // }
    // ```
    let span = naga::Span::default();
    let mut module = ir::Module::default();

    let ty_array = make_runtime_array(&mut module);

    let mut fun = Function {
        name: Some("f".into()),
        ..Default::default()
    };

    // Create component expressions
    let ex_0 = fun
        .expressions
        .append(Expression::Literal(naga::Literal::U32(0)), span);
    let ex_1 = fun
        .expressions
        .append(Expression::Literal(naga::Literal::U32(1)), span);
    let ex_2 = fun
        .expressions
        .append(Expression::Literal(naga::Literal::U32(2)), span);

    // Create a Compose expression to construct the array
    let ex_compose = fun.expressions.append(
        Expression::Compose {
            ty: ty_array,
            components: vec![ex_0, ex_1, ex_2],
        },
        span,
    );

    fun.local_variables.append(
        naga::LocalVariable {
            name: Some("x".into()),
            ty: ty_array,
            init: Some(ex_compose),
        },
        span,
    );

    module.functions.append(fun, span);

    let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .expect_err("module should be invalid")
        .into_inner();

    assert!(matches!(
        err,
        valid::ValidationError::Function {
            source: valid::FunctionError::LocalVariable {
                name: local_var_name,
                source: valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        } if local_var_name == "x"
    ));
}

#[test]
fn invalid_constructor_unsized_struct() {
    // Similar to a test in wgsl_errors::invalid_zero_value_constructors:
    // ```
    // struct Unsized { data: array<f32> }
    // fn main() {
    //     var x: Unsized = Unsized();
    // }
    // ```
    use naga::{Module, StructMember, Type, TypeInner};

    let span = naga::Span::default();
    let mut module = Module::default();

    let ty_array = make_runtime_array(&mut module);

    let ty_unsized = module.types.insert(
        Type {
            name: Some("Unsized".into()),
            inner: TypeInner::Struct {
                members: vec![StructMember {
                    name: Some("data".into()),
                    ty: ty_array,
                    binding: None,
                    offset: 0,
                }],
                span: 4,
            },
        },
        span,
    );

    let mut fun = Function {
        name: Some("main".into()),
        ..Default::default()
    };

    make_zero_value_local_variable(&mut fun, ty_unsized);

    module.functions.append(fun, span);

    let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .expect_err("module should be invalid")
        .into_inner();

    assert!(matches!(
        err,
        valid::ValidationError::Function {
            source: valid::FunctionError::LocalVariable {
                source: valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        },
    ));
}

#[test]
fn arity_check() {
    use ir::MathFunction as Mf;
    use naga::Span;
    let _ = env_logger::builder().is_test(true).try_init();

    type Result = core::result::Result<ModuleInfo, naga::valid::ValidationError>;

    fn validate(fun: ir::MathFunction, args: &[usize]) -> Result {
        let nowhere = Span::default();
        let mut module = ir::Module::default();
        let ty_f32 = module.types.insert(
            ir::Type {
                name: Some("f32".to_string()),
                inner: ir::TypeInner::Scalar(ir::Scalar::F32),
            },
            nowhere,
        );
        let mut f = ir::Function {
            result: Some(ir::FunctionResult {
                ty: ty_f32,
                binding: None,
            }),
            ..ir::Function::default()
        };
        let ex_zero = f
            .expressions
            .append(ir::Expression::ZeroValue(ty_f32), nowhere);
        let ex_pow = f.expressions.append(
            dbg!(ir::Expression::Math {
                fun,
                arg: ex_zero,
                arg1: args.contains(&1).then_some(ex_zero),
                arg2: args.contains(&2).then_some(ex_zero),
                arg3: args.contains(&3).then_some(ex_zero),
            }),
            nowhere,
        );
        f.body = ir::Block::from_vec(vec![
            ir::Statement::Emit(naga::Range::new_from_bounds(ex_pow, ex_pow)),
            ir::Statement::Return {
                value: Some(ex_pow),
            },
        ]);
        module.functions.append(f, nowhere);
        valid::Validator::new(Default::default(), valid::Capabilities::all())
            .validate(&module)
            .map_err(|err| err.into_inner()) // discard spans
    }

    assert!(validate(Mf::Sin, &[]).is_ok());
    assert!(validate(Mf::Sin, &[1]).is_err());
    assert!(validate(Mf::Sin, &[3]).is_err());
    assert!(validate(Mf::Pow, &[1]).is_ok());
    assert!(validate(Mf::Pow, &[3]).is_err());
}

#[test]
fn global_use_scalar() {
    let source = "
@group(0) @binding(0)
var<storage, read_write> global: u32;

@compute @workgroup_size(64)
fn main() {
    let used = &global;
}
    ";

    let module = naga::front::wgsl::parse_str(source).expect("module should parse");
    let info = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .unwrap();

    let global = module.global_variables.iter().next().unwrap().0;
    assert_eq!(
        info.get_entry_point(0)[global],
        naga::valid::GlobalUse::QUERY
    );
}

#[test]
fn global_use_array() {
    let source = "
@group(0) @binding(0)
var<storage, read_write> global: array<f32>;

@compute @workgroup_size(64)
fn main() {
    let used = &global;
}
    ";

    let module = naga::front::wgsl::parse_str(source).expect("module should parse");
    let info = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .unwrap();

    let global = module.global_variables.iter().next().unwrap().0;
    assert_eq!(
        info.get_entry_point(0)[global],
        naga::valid::GlobalUse::QUERY
    );
}

#[test]
fn global_use_array_index() {
    let source = "
@group(0) @binding(0)
var<storage, read_write> global: array<f32>;

@compute @workgroup_size(64)
fn main() {
    let used = &global[0];
}
    ";

    let module = naga::front::wgsl::parse_str(source).expect("module should parse");
    let info = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .unwrap();

    let global = module.global_variables.iter().next().unwrap().0;
    assert_eq!(
        info.get_entry_point(0)[global],
        naga::valid::GlobalUse::QUERY
    );
}

#[test]
fn global_use_phony() {
    let source = "
@group(0) @binding(0)
var<storage, read_write> global: u32;

@compute @workgroup_size(64)
fn main() {
    _ = &global;
}
    ";

    let module = naga::front::wgsl::parse_str(source).expect("module should parse");
    let info = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .unwrap();

    let global = module.global_variables.iter().next().unwrap().0;
    assert_eq!(
        info.get_entry_point(0)[global],
        naga::valid::GlobalUse::QUERY
    );
}

#[test]
fn global_use_unreachable() {
    // We should allow statements after `return`, and such statements should
    // still contribute to global usage. (Unreachable statements should not
    // contribute to uniformity analysis, but there are multiple issues with
    // the current implementation of uniformity analysis, see #4369.)

    let source = "
@group(0) @binding(0)
var<storage, read_write> global: u32;

@compute @workgroup_size(64)
fn main() {
    var used: u32;
    return;
    used = global;
}
    ";

    let module = naga::front::wgsl::parse_str(source).expect("module should parse");
    let mut validator = valid::Validator::new(Default::default(), valid::Capabilities::all());
    let info = validator.validate(&module).unwrap();

    let global = module.global_variables.iter().next().unwrap().0;
    assert_eq!(
        info.get_entry_point(0)[global],
        naga::valid::GlobalUse::READ,
    );
}

/// Parse and validate the module defined in `source`.
///
/// Panics if unsuccessful.
fn parse_validate(source: &str) -> (Module, ModuleInfo) {
    let module = naga::front::wgsl::parse_str(source).expect("module should parse");
    let info = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .unwrap();
    (module, info)
}

/// Helper for `process_overrides` tests.
///
/// The goal of these tests is to verify that `process_overrides` accepts cases
/// where all necessary overrides are specified (even if some unnecessary ones
/// are not), and does not accept cases where necessary overrides are missing.
/// "Necessary" means that the override is referenced in some way by some
/// function reachable from the specified entry point.
///
/// Each test passes a source snippet containing a compute entry point `used`
/// that makes use of the override `ov` in some way. We augment that with (1)
/// the definition of `ov` and (2) a dummy entrypoint that does not use the
/// override, and then test the matrix of (specified or not) x (used or not).
///
/// The optional `unused_body` can introduce additional objects to the module,
/// to verify that they are adjusted correctly by compaction.
fn override_test(test_case: &str, unused_body: Option<&str>) {
    use hashbrown::HashMap;
    use naga::back::pipeline_constants::PipelineConstantError;

    let source = [
        "override ov: u32;\n",
        test_case,
        "@compute @workgroup_size(64)
fn unused() {
",
        unused_body.unwrap_or_default(),
        "}
",
    ]
    .concat();
    let (module, info) = parse_validate(&source);

    let overrides = HashMap::from([(String::from("ov"), 1.)]);

    // Can translate `unused` with or without the override
    naga::back::pipeline_constants::process_overrides(
        &module,
        &info,
        Some((ir::ShaderStage::Compute, "unused")),
        &HashMap::new(),
    )
    .unwrap();
    naga::back::pipeline_constants::process_overrides(
        &module,
        &info,
        Some((ir::ShaderStage::Compute, "unused")),
        &overrides,
    )
    .unwrap();

    // Cannot translate `used` without the override
    let err = naga::back::pipeline_constants::process_overrides(
        &module,
        &info,
        Some((ir::ShaderStage::Compute, "used")),
        &HashMap::new(),
    )
    .unwrap_err();
    assert!(matches!(err, PipelineConstantError::MissingValue(name) if name == "ov"));

    // Can translate `used` if the override is specified
    naga::back::pipeline_constants::process_overrides(
        &module,
        &info,
        Some((ir::ShaderStage::Compute, "used")),
        &overrides,
    )
    .unwrap();
}

#[test]
fn override_in_workgroup_size() {
    override_test(
        "
@compute @workgroup_size(ov)
fn used() {
}
",
        None,
    );
}

#[test]
fn override_in_workgroup_size_nested() {
    // Initializer for override used in workgroup size refers to another
    // override.
    override_test(
        "
override ov2: u32 = ov + 5;

@compute @workgroup_size(ov2)
fn used() {
}
",
        None,
    );
}

#[test]
fn override_in_function() {
    override_test(
        "
fn foo() -> u32 {
    return ov;
}

@compute @workgroup_size(64)
fn used() {
    foo();
}
",
        None,
    );
}

#[test]
fn override_in_entrypoint() {
    override_test(
        "
fn foo() -> u32 {
    return ov;
}

@compute @workgroup_size(64)
fn used() {
    foo();
}
",
        None,
    );
}

#[test]
fn override_in_array_size() {
    override_test(
        "
var<workgroup> arr: array<u32, ov>;

@compute @workgroup_size(64)
fn used() {
    _ = arr[5];
}
",
        None,
    );
}

#[test]
fn override_in_global_init() {
    override_test(
        "
var<private> foo: u32 = ov;

@compute @workgroup_size(64)
fn used() {
    _ = foo;
}
",
        None,
    );
}

#[test]
fn override_with_multiple_globals() {
    // Test that when compaction of the `unused` entrypoint removes `arr1`, the
    // handle to `arr2` is adjusted correctly.
    override_test(
        "
var<workgroup> arr1: array<u32, ov>;
var<workgroup> arr2: array<u32, 4>;

@compute @workgroup_size(64)
fn used() {
    _ = arr1[5];
}
",
        Some("_ = arr2[3];"),
    );
}

/// Expects parsing `input` to succeed and its validation to fail with error equal to `snapshot`.
#[track_caller]
fn check_wgsl_validation_error_message(input: &str, snapshot: &str) {
    let module = naga::front::wgsl::parse_str(input).unwrap();
    let err = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .expect_err("module should be invalid")
        .emit_to_string(input);
    if err != snapshot {
        for diff in diff::lines(snapshot, &err) {
            match diff {
                diff::Result::Left(l) => println!("-{l}"),
                diff::Result::Both(l, _) => println!(" {l}"),
                diff::Result::Right(r) => println!("+{r}"),
            }
        }
        panic!("Error does not match the expected snapshot");
    }
}

#[test]
fn image_store_type_mismatch() {
    check_wgsl_validation_error_message(
        r#"
@group(0) @binding(0)
var input_texture: texture_depth_2d;
@group(0) @binding(1)
var input_sampler: sampler;
@group(0) @binding(2)
var output_texture: texture_storage_2d<r32float,write>;

@compute @workgroup_size(1, 1)
fn main() {
    let d: vec4<f32> = textureGather(input_texture, input_sampler, vec2f(0.0));
    let min_d = min(min(d[0], d[1]), min(d[2], d[3]));
    textureStore(output_texture, vec2u(1), min_d);
}
"#,
        r#"error: Entry point main at Compute is invalid
   ┌─ wgsl:12:17
   │
12 │     let min_d = min(min(d[0], d[1]), min(d[2], d[3]));
   │                 ^^^ this value is of type Scalar(Scalar { kind: Float, width: 4 })
13 │     textureStore(output_texture, vec2u(1), min_d);
   │     ^^^^^^^^^^^^ expects a value argument of type Vector { size: Quad, scalar: Scalar { kind: Float, width: 4 } }
   │
   = Image store value parameter type mismatch

"#,
    );
}

#[test]
fn unexpected_task_payload() {
    let mut test_spans = TestSpanGenerator::default();
    let mut module = Module::default();

    let ty_payload = module.types.insert(
        ir::Type {
            name: Some("u32".into()),
            inner: ir::TypeInner::Scalar(naga::Scalar::U32),
        },
        test_spans.next(),
    );

    let err_span = test_spans.next();
    let payload_handle = module.global_variables.append(
        ir::GlobalVariable {
            name: Some("task_payload".into()),
            space: ir::AddressSpace::TaskPayload,
            binding: None,
            ty: ty_payload,
            init: None,
            memory_decorations: ir::MemoryDecorations::empty(),
        },
        err_span,
    );

    let entry_point = ir::EntryPoint {
        name: "main".into(),
        stage: ir::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [1, 1, 1],
        workgroup_size_overrides: None,
        function: ir::Function::default(),
        mesh_info: None,
        task_payload: Some(payload_handle), // invalid for compute stage
        incoming_ray_payload: None,
    };
    module.entry_points.push(entry_point);

    let err = expect_validation_error_with_span(
        &module,
        ValidationFlags::default(),
        Capabilities::MESH_SHADER,
        err_span,
    );

    assert!(matches!(
        err,
        valid::ValidationError::EntryPoint {
            source: valid::EntryPointError::UnexpectedTaskPayload,
            ..
        }
    ));
}

#[test]
fn coherent_requires_capability() {
    let module = naga::front::wgsl::parse_str(
        "struct S { x: u32 }
         @group(0) @binding(0) @coherent var<storage, read_write> buf: S;",
    )
    .expect("module should parse");

    let err = valid::Validator::new(ValidationFlags::default(), Capabilities::empty())
        .validate(&module)
        .expect_err("should fail without capability");
    assert!(matches!(
        err.into_inner(),
        valid::ValidationError::GlobalVariable {
            source: valid::GlobalVariableError::CoherentNotSupported,
            ..
        }
    ));

    let result = valid::Validator::new(
        ValidationFlags::default(),
        Capabilities::MEMORY_DECORATION_COHERENT,
    )
    .validate(&module);
    assert!(result.is_ok(), "should succeed with capability: {result:?}");
}

#[test]
fn volatile_requires_capability() {
    let module = naga::front::wgsl::parse_str(
        "struct S { x: u32 }
         @group(0) @binding(0) @volatile var<storage, read_write> buf: S;",
    )
    .expect("module should parse");

    let err = valid::Validator::new(ValidationFlags::default(), Capabilities::empty())
        .validate(&module)
        .expect_err("should fail without capability");
    assert!(matches!(
        err.into_inner(),
        valid::ValidationError::GlobalVariable {
            source: valid::GlobalVariableError::VolatileNotSupported,
            ..
        }
    ));

    let result = valid::Validator::new(
        ValidationFlags::default(),
        Capabilities::MEMORY_DECORATION_VOLATILE,
    )
    .validate(&module);
    assert!(result.is_ok(), "should succeed with capability: {result:?}");
}

#[test]
fn memory_decorations_require_storage_address_space() {
    let module = naga::front::wgsl::parse_str("@coherent var<workgroup> wg: array<u32, 4>;")
        .expect("module should parse");

    let err = valid::Validator::new(
        ValidationFlags::default(),
        Capabilities::MEMORY_DECORATION_COHERENT | Capabilities::MEMORY_DECORATION_VOLATILE,
    )
    .validate(&module)
    .expect_err("should fail on non-storage address space");
    assert!(matches!(
        err.into_inner(),
        valid::ValidationError::GlobalVariable {
            source: valid::GlobalVariableError::InvalidMemoryDecorationsAddressSpace,
            ..
        }
    ));
}
