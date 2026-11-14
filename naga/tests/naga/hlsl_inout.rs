#![cfg(all(feature = "wgsl-in", feature = "hlsl-out"))]

use naga::{back::hlsl, valid, *};
const PACKAGE_LABEL: &str = env!("CARGO_PKG_NAME");

fn validate(module: &Module) -> valid::ModuleInfo {
    valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(module)
        .unwrap()
}

fn emit(module: &Module) -> String {
    let options = hlsl::Options {
        shader_model: hlsl::ShaderModel::V6_8,
        force_loop_bounding: false,

        ..Default::default()
    };
    let pipeline = hlsl::PipelineOptions::default();
    let mut text = String::new();
    hlsl::Writer::new(&mut text, &options, &pipeline)
        .write(module, &validate(module), None)
        .unwrap();
    text
}

fn save(name: &str, text: &str) {
    if let (Ok(root), Ok(case)) = (std::env::var("OUTPUT326"), std::env::var("CASE326")) {
        let path = std::path::Path::new(&root).join(format!("{case}-{PACKAGE_LABEL}"));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join(format!("{name}.hlsl")), text).unwrap();
    }
}

fn marked(source: &str) -> Module {
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    for function in module
        .functions
        .iter_mut()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter_mut().map(|ep| &mut ep.function))
    {
        let handles: Vec<_> = function
            .named_expressions
            .iter()
            .filter_map(|(&handle, name)| name.starts_with("ctl_").then_some(handle))
            .collect();
        for handle in handles {
            function
                .expression_controls
                .insert(handle, ArithmeticControl::NoContraction);
        }
    }
    module
}

const STORAGE: &str = "@group(0) @binding(0) var<storage,read_write> output:array<vec4<f32>>;";

#[test]
fn inout_operands_evaluate_once_and_keep_ordinary_neighbors() {
    let source = format!(
        "{STORAGE} {}",
        r#"
        fn effect()->f32 { let old=output[0].x; output[0].x=old+1.; return old; }
        @compute @workgroup_size(1) fn main() {
            let left=effect(); let right=effect();
            let ordinary=left*right;
            let ctl_result=ordinary+right;
            output[0]=vec4<f32>(ctl_result+ctl_result);
        }
    "#
    );
    let module = marked(&source);
    let text = emit(&module);
    assert_eq!(text.matches("= effect();").count(), 2);
    assert_eq!(text.matches("[noinline]").count(), 1);
    assert!(text.contains("inout float operands[2]"));
    assert!(text.contains("precise float result = operands[0] + operands[1]"));
    assert!(text.contains("operands[0] = result;"));
    assert!(text.contains("float ordinary = "));
    assert!(!text.contains("precise float ordinary"));
    assert!(!text.contains("out float output_value"));
    assert!(!text.contains("return result;"));
    assert_eq!(text.matches("float ctl_result = ").count(), 1);
    assert!(text.find("float ordinary").unwrap() < text.find("gen_naga_operands[2]").unwrap());
    save("effects", &text);
}

#[test]
fn inout_body_values_reach_continuing_without_recomputation() {
    let source = format!(
        "{STORAGE} {}",
        r#"
        var<private> count:u32;
        fn effect()->f32 { count+=1u; return f32(count); }
        @compute @workgroup_size(1) fn main() {
            loop {
                let input=effect();
                let ctl_body=input*input;
                if count == 2u { continue; }
                continuing {
                    let ctl_continuing=ctl_body+input;
                    output[0]=vec4<f32>(ctl_continuing);
                    break if count > 4u;
                }
            }
        }
    "#
    );
    let text = emit(&marked(&source));
    let loop_start = text.find("while(true)").unwrap();
    assert!(text.find("float ctl_body;").unwrap() < loop_start);
    assert!(text.find("ctl_body = ").unwrap() > loop_start);
    assert_eq!(text.matches("= effect();").count(), 1);
    assert_eq!(text.matches("ctl_body = ").count(), 1);
    assert_eq!(text.matches("ctl_continuing = ").count(), 1);
    assert!(text.contains("{(ctl_body), (_e"));
    assert!(text.find("ctl_continuing = ").unwrap() < text.find("ctl_body = ").unwrap());
    save("continuing-body", &text);
}

#[test]
fn inout_branches_nested_loops_switches_and_break_if() {
    let source = format!(
        "{STORAGE} {}",
        r#"
        @compute @workgroup_size(1) fn main() {
            var i=0u;
            loop {
                let x=output[i].x;
                if i == 0u { let ctl_branch=x*x; output[i].y=ctl_branch; }
                switch i {
                    case 0u: { let ctl_case=x+1.; output[i].z=ctl_case; }
                    default: { if i > 5u { break; } }
                }
                loop { let ctl_nested=x-1.; output[i].w=ctl_nested; break; }
                continuing {
                    let ctl_continuing=x+x;
                    output[i].x=ctl_continuing;
                    i+=1u;
                    break if i >= 8u;
                }
            }
        }
    "#
    );
    let text = emit(&marked(&source));
    assert_eq!(text.matches("[noinline]").count(), 4);
    assert!(text.find("if (").unwrap() < text.find("ctl_branch = ").unwrap());
    assert!(text.find("switch(").unwrap() < text.find("ctl_case = ").unwrap());
    assert_eq!(text.matches("ctl_nested = ").count(), 1);
    save("flow", &text);
}

fn typed(scalar: Scalar, size: Option<VectorSize>, op: BinaryOperator, dot: bool) -> Module {
    let mut module = Module::default();
    let scalar_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(scalar),
        },
        Span::UNDEFINED,
    );
    let ty = size.map_or(scalar_ty, |size| {
        module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector { size, scalar },
            },
            Span::UNDEFINED,
        )
    });
    let mut function = Function {
        name: Some("typed".into()),
        ..Default::default()
    };
    function.arguments = vec![
        FunctionArgument {
            name: Some("left".into()),
            ty,
            binding: None,
        },
        FunctionArgument {
            name: Some("right".into()),
            ty,
            binding: None,
        },
    ];
    function.result = Some(FunctionResult {
        ty: if dot { scalar_ty } else { ty },
        binding: None,
    });
    let left = function
        .expressions
        .append(Expression::FunctionArgument(0), Span::UNDEFINED);
    let right = function
        .expressions
        .append(Expression::FunctionArgument(1), Span::UNDEFINED);
    let result = function.expressions.append(
        if dot {
            Expression::Math {
                fun: MathFunction::Dot,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            }
        } else {
            Expression::Binary { op, left, right }
        },
        Span::UNDEFINED,
    );
    function
        .expression_controls
        .insert(result, ArithmeticControl::NoContraction);
    function.body.push(
        Statement::Emit(naga::Range::new_from_bounds(result, result)),
        Span::UNDEFINED,
    );
    function.body.push(
        Statement::Return {
            value: Some(result),
        },
        Span::UNDEFINED,
    );
    let handle = module.functions.append(function, Span::UNDEFINED);
    let mut entry = Function::default();
    let zero = entry
        .expressions
        .append(Expression::ZeroValue(ty), Span::UNDEFINED);
    let call = entry
        .expressions
        .append(Expression::CallResult(handle), Span::UNDEFINED);
    entry.body.push(
        Statement::Call {
            function: handle,
            arguments: vec![zero, zero],
            result: Some(call),
        },
        Span::UNDEFINED,
    );
    let mut ep = naga::front::wgsl::parse_str("@compute @workgroup_size(1) fn main() {}")
        .unwrap()
        .entry_points
        .remove(0);
    ep.function = entry;
    module.entry_points.push(ep);
    module
}

#[test]
fn inout_width_shape_operator_matrix() {
    for kind in [ScalarKind::Float, ScalarKind::Sint, ScalarKind::Uint] {
        for width in [2, 4, 8] {
            let scalar = Scalar { kind, width };
            for size in [
                None,
                Some(VectorSize::Bi),
                Some(VectorSize::Tri),
                Some(VectorSize::Quad),
            ] {
                for op in [
                    BinaryOperator::Add,
                    BinaryOperator::Subtract,
                    BinaryOperator::Multiply,
                ] {
                    let text = emit(&typed(scalar, size, op, false));
                    assert_eq!(text.matches("[noinline]").count(), 1);
                    let calls = text
                        .lines()
                        .filter(|s| s.trim_start().starts_with("gen_naga_no_contraction("))
                        .count();
                    assert_eq!(calls, size.map_or(1, |size| size as usize));
                    let name = format!(
                        "{kind:?}{width}-{}-{op:?}",
                        size.map_or(1, |size| size as usize)
                    );
                    save(&name, &text);
                }
            }
            if kind == ScalarKind::Float {
                for size in [VectorSize::Bi, VectorSize::Tri, VectorSize::Quad] {
                    let module = typed(scalar, Some(size), BinaryOperator::Multiply, true);
                    if width == 8 {
                        let mut text = String::new();
                        let options = hlsl::Options {
                            shader_model: hlsl::ShaderModel::V6_8,
                            ..Default::default()
                        };
                        let pipeline = hlsl::PipelineOptions::default();
                        assert!(matches!(
                            hlsl::Writer::new(&mut text, &options, &pipeline).write(
                                &module,
                                &validate(&module),
                                None
                            ),
                            Err(hlsl::Error::UnsupportedArithmeticControls)
                        ));
                    } else {
                        let text = emit(&module);
                        assert_eq!(text.matches(" = dot(").count(), 1);
                        assert!(text.contains(&format!("operands[{}]", 2 * size as usize)));
                        save(&format!("dot-f{}-{}", width * 8, size as usize), &text);
                    }
                }
            }
        }
    }
}

fn check_atomic_load_continuing(address: &str) {
    for (kind, literal, scalar) in [("u32", "u", "uint"), ("i32", "i", "int")] {
        let declaration = if address == "workgroup" {
            format!("var<workgroup> counter: atomic<{kind}>;")
        } else {
            format!("@group(0) @binding(1) var<storage, read_write> counter: atomic<{kind}>;")
        };
        let source = format!(
            r#"
            {declaration}
            @group(0) @binding(0) var<storage, read_write> output: array<{kind}>;
            @compute @workgroup_size(1) fn main() {{
                atomicStore(&counter, 7{literal});
                loop {{
                    let sampled = atomicLoad(&counter);
                    let ctl_body = sampled + 1{literal};
                    continuing {{
                        output[0] = ctl_body + sampled;
                        break if true;
                    }}
                }}
            }}
            "#
        );
        let text = emit(&marked(&source));
        save(&format!("atomic-{address}-{kind}"), &text);
        let declaration = format!("{scalar} sampled;");
        assert_eq!(text.matches(&declaration).count(), 1, "{text}");
        let declaration = text.find(&declaration).unwrap();
        let loop_start = text.find("while(true)").unwrap();
        let atomic_read = text.find("InterlockedOr(").unwrap();
        assert!(
            declaration < loop_start && loop_start < atomic_read,
            "{text}"
        );
        assert_eq!(text.matches("InterlockedOr(").count(), 1, "{text}");
        assert!(text[atomic_read..].contains(", 0, sampled);"), "{text}");
        assert!(text[loop_start..atomic_read].contains("sampled"), "{text}");
    }
}

#[test]
fn inout_atomic_workgroup_load_reuses_hoisted_storage_in_continuing() {
    check_atomic_load_continuing("workgroup");
}

#[test]
fn inout_atomic_storage_load_reuses_hoisted_storage_in_continuing() {
    check_atomic_load_continuing("storage");
}

const BUFFERS: &str = r#"
    struct Input { a: vec4<f32>, b: vec4<f32>, index: u32 }
    @group(0) @binding(0) var<storage, read_write> output: array<vec4<f32>>;
    @group(0) @binding(1) var<uniform> input: Input;
"#;

/// Writes `body` as a compute entry point and returns, for each helper,
/// whether it is `[noinline]`.
fn helper_attributes(declarations: &str, body: &str, name: &str) -> Vec<bool> {
    let source =
        format!("{BUFFERS} {declarations} @compute @workgroup_size(1) fn main() {{ {body} }}");
    let text = emit(&marked(&source));
    save(name, &text);
    let lines: Vec<_> = text.lines().collect();
    let helpers: Vec<_> = (0..lines.len())
        .filter(|&line| lines[line].starts_with("void gen_naga_no_contraction"))
        .map(|line| line > 0 && lines[line - 1] == "[noinline]")
        .collect();
    let kept = helpers.iter().filter(|&&kept| kept).count();
    assert_eq!(text.matches("[noinline]").count(), kept, "{text}");
    assert_eq!(text.matches("precise ").count(), helpers.len(), "{text}");
    helpers
}

#[test]
fn inout_helpers_are_inlined_when_isolated() {
    // Controlled operations whose operands are loads, other controlled
    // operations and operations that `precise` cannot change.
    let helpers = helper_attributes(
        "",
        r#"
        let a = input.a;
        let b = input.b;
        let ctl_product = a * b;
        let ctl_sum = ctl_product + a;
        let ctl_dot = dot(ctl_sum.xyz, b.xyz);
        let exact = floor(abs(-a.x));
        let ctl_scalar = select(exact, b.y, input.index == 1u) * f32(input.index);
        var local = ctl_scalar;
        let ctl_mix = ctl_dot - local;
        local = ctl_mix;
        let ctl_last = local - trunc(b.w);
        // Consumers of a controlled result are not part of what it reads.
        output[0] = vec4<f32>(ctl_last * 2.0) + ctl_sum * b;
        "#,
        "inline-isolated",
    );
    assert_eq!(helpers, [false; 6]);

    // An integer control is isolated by integer operands.
    let helpers = helper_attributes(
        "",
        r#"
        let i = input.index;
        let ctl_integer = (i >> 2u) * countOneBits(i);
        output[0] = vec4<f32>(f32(ctl_integer) * input.a.x);
        "#,
        "inline-integer",
    );
    assert_eq!(helpers, [false]);

    // A private variable that only the entry point stores to.
    let helpers = helper_attributes(
        "var<private> scratch: f32; fn unrelated() -> f32 { return input.a.x * input.a.y; }",
        r#"
        scratch = input.a.x;
        let ctl_private = scratch + input.b.x;
        output[0] = vec4<f32>(ctl_private);
        "#,
        "inline-private",
    );
    assert_eq!(helpers, [false]);
}

#[test]
fn inout_helpers_stay_noinline_when_precise_would_spread() {
    let cases = [
        // Uncontrolled floating-point arithmetic feeds the controlled operation.
        (
            "operand",
            "",
            "let t = input.a.x * input.b.x; let ctl_r = t + input.a.y; output[0] = vec4<f32>(ctl_r);",
        ),
        // The same through a vector lane that the control does not read:
        // DXC applies `precise` to whole vector values.
        (
            "lane",
            "",
            "let v = input.a * input.b; let ctl_r = v.yxwz.x + input.a.y; output[0] = vec4<f32>(ctl_r);",
        ),
        // Through a local variable.
        (
            "local",
            "",
            "var v = input.a.x * input.b.x; let ctl_r = v + input.a.y; output[0] = vec4<f32>(ctl_r);",
        ),
        // Through a store that may alias a load.
        (
            "alias",
            "",
            "let ctl_r = output[input.index].x + input.a.y; output[input.index + 1u].x = ctl_r * input.b.x;",
        ),
        // Operations whose result can depend on `precise`.
        (
            "comparison",
            "",
            "let ctl_r = select(input.a.x, input.b.x, input.a.y < input.b.y) + input.a.z; output[0] = vec4<f32>(ctl_r);",
        ),
        (
            "minimum",
            "",
            "let ctl_r = min(input.a.x, input.b.x) + input.a.z; output[0] = vec4<f32>(ctl_r);",
        ),
        (
            "square-root",
            "",
            "let ctl_r = sqrt(input.a.x) + input.a.z; output[0] = vec4<f32>(ctl_r);",
        ),
        (
            "unpack",
            "",
            "let ctl_r = unpack4x8unorm(input.index).x + input.a.z; output[0] = vec4<f32>(ctl_r);",
        ),
        // Control flow and calls are not modeled, wherever they are.
        (
            "branch",
            "",
            "let ctl_r = input.a.x + input.a.y; if input.index == 0u { output[0] = vec4<f32>(ctl_r); }",
        ),
        (
            "call",
            "fn side() { output[1] = vec4<f32>(1.0); }",
            "let ctl_r = input.a.x + input.a.y; output[0] = vec4<f32>(ctl_r); side();",
        ),
        // Another function stores a computed value to a private variable.
        (
            "private",
            "var<private> scratch: f32; fn writer() { scratch = input.a.x * input.b.x; }",
            "let ctl_r = scratch + input.a.y; output[0] = vec4<f32>(ctl_r);",
        ),
    ];
    for (name, declarations, body) in cases {
        let helpers = helper_attributes(declarations, body, &format!("noinline-{name}"));
        assert_eq!(helpers, [true], "{name}");
    }
}

/// The rules by which DXC applies `precise` to vector components and local
/// variables. Every case has one controlled operation. `product` is an
/// uncontrolled multiplication, so a case keeps `[noinline]` exactly when
/// `precise` on the controlled result would reach `product`.
///
/// Compiling each saved case with DXC, with and without `[noinline]`, shows
/// the same: the cases named `inline-` mark the same operations either way,
/// those named `noinline-` do not.
#[test]
fn inout_helpers_follow_dxc_value_rules() {
    const PRODUCT: &str = "let product = input.a.x * input.b.x;";
    const REGISTER: &str = "var r = vec4<u32>();";
    let cases = [
        // A component read sees through constructors and swizzles.
        (
            true,
            "compose-component",
            "let v = vec4<f32>(product, input.a.y, 0.0, 1.0); let ctl_r = v.y + input.a.z; output[0] = v + ctl_r;",
        ),
        (
            true,
            "compose-vector-component",
            "let v = vec4<f32>(vec2<f32>(product, input.a.y), input.a.zw); let ctl_r = v.y + v.w; output[0] = v + ctl_r;",
        ),
        (
            true,
            "swizzle-component",
            "let v = vec4<f32>(product, input.a.yzw); let w = v.yxwz; let ctl_r = w.zw.y + w.x; output[0] = w + ctl_r;",
        ),
        // A vector operation is one value, whichever component is read.
        (
            false,
            "negated-component",
            "let v = -vec2<f32>(product, input.a.y); let ctl_r = v.y + input.a.z; output[0] = vec4<f32>(v, ctl_r, 0.0);",
        ),
        (
            false,
            "bitcast-component",
            "let v = bitcast<vec2<f32>>(vec2<u32>(bitcast<u32>(product), input.index)); let ctl_r = v.y + input.a.z; output[0] = vec4<f32>(v, ctl_r, 0.0);",
        ),
        (
            false,
            "integer-component",
            "let v = vec2<u32>(bitcast<u32>(product), input.index) & vec2<u32>(255u); let ctl_r = bitcast<f32>(v.y) + input.a.z; output[0] = vec4<f32>(vec2<f32>(v), ctl_r, 0.0);",
        ),
        // A swizzle that is the operand of a vector operation reads its whole
        // source, including the components it does not select.
        (
            false,
            "swizzle-operand",
            "let v = vec4<f32>(product, input.a.yzw); let w = abs(v.yy); let ctl_r = w.x + input.a.z; output[0] = v + ctl_r;",
        ),
        // A local variable is read as the stores that reach the load.
        (
            true,
            "later-store",
            "var v = input.a.x; let ctl_r = v + input.a.y; v = ctl_r * input.b.x; output[0] = vec4<f32>(v);",
        ),
        (
            true,
            "overwritten-store",
            "var v = product; output[1] = vec4<f32>(v); v = input.a.x; let ctl_r = v + input.a.y; output[0] = vec4<f32>(ctl_r);",
        ),
        (
            true,
            "component-store",
            "var v = vec4<f32>(); v.x = product; v.y = input.a.y; let ctl_r = v.y + input.a.z; output[0] = v + ctl_r;",
        ),
        (
            true,
            "component-store-whole-load",
            "var v = vec4<f32>(); v.x = product; v.y = input.a.y; let w = v; let ctl_r = w.y + input.a.z; output[0] = w + ctl_r;",
        ),
        (
            true,
            "later-component-store",
            "var v = input.a; let w = v; let ctl_r = w.x + input.a.y; v.x = product; output[0] = v + ctl_r;",
        ),
        // Loaded and used as a vector, a variable is every component store
        // since the last store to the whole variable, and that store.
        (
            false,
            "register-other-component",
            "r.x = bitcast<u32>(product); r.y = input.index; let ctl_r = bitcast<vec2<f32>>(r.yy).x + input.a.z; output[0] = vec4<f32>(r) + ctl_r;",
        ),
        (
            false,
            "register-overwritten-component",
            "r.y = bitcast<u32>(product); output[1] = vec4<f32>(r); r.y = input.index; let ctl_r = bitcast<vec2<f32>>(r.yy).x + input.a.z; output[0] = vec4<f32>(ctl_r);",
        ),
        (
            false,
            "register-whole-store",
            "r = vec4<u32>(bitcast<u32>(product)); output[1] = vec4<f32>(r); r.x = 1u; r.y = input.index; r.z = 2u; r.w = 3u; let ctl_r = bitcast<vec2<f32>>(r.yy).x + input.a.z; output[0] = vec4<f32>(ctl_r);",
        ),
        (
            true,
            "register-component-read",
            "r.x = bitcast<u32>(product); r.y = input.index; let w = r; let ctl_r = bitcast<f32>(w.y) + input.a.z; output[0] = vec4<f32>(w) + ctl_r;",
        ),
        (
            true,
            "register-reset",
            "r.x = bitcast<u32>(product); output[1] = vec4<f32>(r); r = vec4<u32>(input.index); r.z = 1u; let ctl_r = bitcast<vec2<f32>>(r.yy).x + input.a.z; output[0] = vec4<f32>(ctl_r);",
        ),
        // A dynamic index makes every store to the variable visible to
        // every load of it.
        (
            false,
            "dynamic-component",
            "var v = input.a; let ctl_r = v.z + input.a.y; v[input.index] = product; output[0] = v + ctl_r;",
        ),
        (
            true,
            "array-element",
            "var v: array<f32, 3>; v[0] = product; v[1] = input.a.y; let ctl_r = v[1] + input.a.z; output[0] = vec4<f32>(v[0], v[1], v[2], ctl_r);",
        ),
        (
            false,
            "dynamic-element",
            "var v: array<f32, 3>; v[0] = product; v[1] = input.a.y; v[input.index] = input.a.z; let ctl_r = v[1] + input.a.z; output[0] = vec4<f32>(v[0], v[1], v[2], ctl_r);",
        ),
        // A conversion to an integer is written with a clamp.
        (
            false,
            "integer-conversion",
            "let ctl_r = f32(i32(input.a.x)) + input.a.z; output[0] = vec4<f32>(ctl_r + product);",
        ),
    ];
    for (inlined, name, body) in cases {
        let prefix = if inlined { "inline" } else { "noinline" };
        let body = format!("{PRODUCT} {REGISTER} {body}");
        let helpers = helper_attributes("", &body, &format!("{prefix}-{name}"));
        assert_eq!(helpers, [!inlined], "{name}");
    }
}

/// The shape of a program translated from register-based bytecode: a
/// straight-line function that keeps every value in `vec4<u32>` registers,
/// writes them one component at a time and reads them back through swizzles
/// and bit casts. Every arithmetic operation up to the last one is
/// controlled; the last one reuses a register.
const REGISTER_MACHINE: &str = r#"
    struct Bank { rows: array<vec4<u32>, 4> }
    @group(0) @binding(0) var<uniform> bank: Bank;
    var<private> p_position: vec4<f32>;
    var<private> p_out: vec4<f32>;
    var<private> p_extra: vec4<f32>;
    fn function() {
        var r0 = vec4<u32>();
        var r1 = vec4<u32>();
        var r2 = vec4<u32>();
        let position = p_position;
        let raw = bitcast<vec4<u32>>(position);
        r0.x = raw.x;
        r0.y = raw.y;
        r0.z = raw.z;
        r0.w = raw.w;
        let e0 = r0;
        let ctl_dot = dot(bitcast<vec2<f32>>(e0.xy), vec2<f32>(0.36602542));
        r1.x = bitcast<u32>(ctl_dot);
        let e1 = r1;
        let row = bank.rows[3u];
        let ctl_sum = bitcast<vec2<f32>>(e1.xx) + bitcast<vec2<f32>>(row.xz);
        let u1 = bitcast<vec2<u32>>(ctl_sum);
        r1.x = u1.x;
        r1.y = u1.y;
        let e2 = r1;
        let u2 = bitcast<vec2<u32>>(floor(bitcast<vec2<f32>>(e2.xy)));
        r1.z = u2.x;
        r1.w = u2.y;
        let e3 = r1;
        let ctl_product = -bitcast<vec2<f32>>(e3.zw) * bitcast<vec2<f32>>(e3.xy);
        let u3 = bitcast<vec2<u32>>(ctl_product);
        r2.x = u3.x;
        r2.y = u3.y;
        let e4 = r2;
        r2.z = select(e4.x, e4.y, (e4.x & 0x80000000u) != 0u);
        let e5 = r2;
        let e6 = r0;
        let ctl_scale = bitcast<f32>(e5.z) * abs(bitcast<vec3<f32>>(e6.xyz)).z;
        r2 = vec4<u32>(e5.xy, bitcast<u32>(ctl_scale), 0x3f800000u);
        let e7 = r2;
        p_out = bitcast<vec4<f32>>(e7);
        let e8 = r0;
        r0 = bitcast<vec4<u32>>(bitcast<vec4<f32>>(e7) * position + bitcast<vec4<f32>>(e8));
        let e9 = r0;
        p_extra = bitcast<vec4<f32>>(e9);
    }
    struct Output { @builtin(position) position: vec4<f32>, @location(0) extra: vec4<f32> }
"#;

#[test]
fn inout_helpers_are_inlined_in_a_register_machine() {
    let noinline = |name: &str, source: &str| {
        let text = emit(&marked(source));
        save(name, &text);
        assert_eq!(text.matches("precise ").count(), 4, "{text}");
        text.matches("[noinline]").count()
    };
    let main = "@vertex fn main(@location(0) position: vec4<f32>) -> Output {
        p_position = position; function(); return Output(p_out, p_extra);
    }";
    let source = format!("{REGISTER_MACHINE} {main}");
    assert_eq!(noinline("inline-register-machine", &source), 0);

    // The same program with an uncontrolled operation stored to a register
    // before a controlled operation loads that register.
    let early = source.replace(
        "let e6 = r0;",
        "r0.w = bitcast<u32>(position.x * position.y); let e6 = r0;",
    );
    assert_ne!(early, source);
    assert_eq!(noinline("noinline-register-machine-early", &early), 4);
}

#[test]
fn inout_helpers_follow_call_sites() {
    let noinline = |name: &str, source: &str| {
        let text = emit(&marked(&format!("{BUFFERS} {source}")));
        save(name, &text);
        assert_eq!(text.matches("precise ").count(), 1, "{text}");
        text.matches("[noinline]").count()
    };
    // The shape the SPIR-V frontend produces: the entry point copies its
    // arguments to private variables and calls the function unconditionally.
    let callee = r#"
        var<private> p_in: vec4<f32>;
        var<private> p_out: f32;
        fn callee() { let ctl_r = p_in.x * p_in.y; p_out = ctl_r; }
    "#;
    let main = |body: &str| {
        format!(
            "{callee} @fragment fn main(@location(0) x: vec4<f32>) -> @location(0) f32 {{ {body} }}"
        )
    };
    let unconditional = main("p_in = x; callee(); return p_out * x.z;");
    assert_eq!(noinline("call-unconditional", &unconditional), 0);

    // DXC inlines `callee` into `main`, where `precise` also applies to the
    // comparison that the call depends on.
    let conditional = main("p_in = x; if x.z * x.w > 1.0 { callee(); } return p_out;");
    assert_eq!(noinline("call-conditional", &conditional), 1);

    // The entry point stores a computed value to the variable that is read.
    let computed = main("p_in = x * x; callee(); return p_out;");
    assert_eq!(noinline("call-computed-input", &computed), 1);

    // A call from another function is not followed further.
    let nested = format!(
        "{callee} fn middle() {{ callee(); }}
        @fragment fn main(@location(0) x: vec4<f32>) -> @location(0) f32 {{
            p_in = x; middle(); return p_out;
        }}"
    );
    assert_eq!(noinline("call-nested", &nested), 1);

    // An argument of the function is an arbitrary expression in its caller.
    let argument = r#"
        fn callee(v: vec4<f32>) -> f32 { let ctl_r = v.x * v.y; return ctl_r; }
        @fragment fn main(@location(0) x: vec4<f32>) -> @location(0) f32 {
            return callee(x * x);
        }
    "#;
    assert_eq!(noinline("call-argument", argument), 1);
}
