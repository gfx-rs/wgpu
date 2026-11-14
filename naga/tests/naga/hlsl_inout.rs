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

#[test]
fn inout_exp2_preserves_one_scalar_operation_and_operand_evaluation() {
    let source = format!(
        "{STORAGE} {}",
        r#"
        fn effect()->f32 { let old=output[0].x; output[0].x=old+1.; return old; }
        @compute @workgroup_size(1) fn main() {
            let x=effect();
            let ordinary=x*x;
            let exp_result=exp2(ordinary);
            output[0]=vec4<f32>(exp_result+exp_result);
        }
    "#
    );
    let mut module = marked(&source);
    let function = &mut module.entry_points[0].function;
    let handle = *function
        .named_expressions
        .iter()
        .find(|(_, name)| name.as_str() == "exp_result")
        .unwrap()
        .0;
    function
        .expression_controls
        .insert(handle, ArithmeticControl::Exp2F32NoRefactoringV1);
    let text = emit(&module);
    assert_eq!(text.matches("= effect();").count(), 1);
    assert_eq!(text.matches("[noinline]").count(), 1);
    assert!(text.contains("inout float operands[1]"));
    assert!(text.contains("precise float result = exp2(operands[0])"));
    assert_eq!(text.matches("= exp2(").count(), 1);
    assert!(text.find("float ordinary = ").unwrap() < text.find("gen_naga_operands[1]").unwrap());
    assert!(!text.contains("return result;"));
    save("exp2-effects", &text);
}

#[test]
fn inout_exp2_rejects_other_types() {
    for (scalar, size) in [
        (Scalar::F32, Some(VectorSize::Bi)),
        (Scalar::F64, None),
        (
            Scalar {
                kind: ScalarKind::Float,
                width: 2,
            },
            None,
        ),
    ] {
        let mut module = typed(scalar, size, BinaryOperator::Multiply, false);
        let function = module.functions.iter_mut().next().unwrap().1;
        let handle = *function.expression_controls.keys().next().unwrap();
        let arg = match function.expressions[handle] {
            Expression::Binary { left, .. } => left,
            _ => unreachable!(),
        };
        function.expressions[handle] = Expression::Math {
            fun: MathFunction::Exp2,
            arg,
            arg1: None,
            arg2: None,
            arg3: None,
        };
        function
            .expression_controls
            .insert(handle, ArithmeticControl::Exp2F32NoRefactoringV1);
        let error =
            valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
                .validate(&module)
                .unwrap_err();
        assert!(format!("{error:?}").contains("InvalidArithmeticControl"));
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
