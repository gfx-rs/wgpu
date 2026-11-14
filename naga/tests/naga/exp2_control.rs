use naga::*;
const EXP: ArithmeticControl = ArithmeticControl::Exp2F32NoRefactoringV1;
const SRC: &str = r#"
@fragment fn main(@location(0) x:f32)->@location(0) vec4f {
 let up=x*x; let e=exp2(up); let c=e*x; let neighbor=x*3.0;
 let ordinary=exp2(x); let d=dot(vec2f(x,e),vec2f(x,2.0));
 return vec4f(c,neighbor,ordinary,d);
}
@compute @workgroup_size(1) fn empty() {}
"#;
fn validate(m: &Module) -> valid::ModuleInfo {
    valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(m)
        .unwrap()
}
fn fixture() -> Module {
    let mut m = front::wgsl::parse_str(SRC).unwrap();
    let f = &mut m.entry_points[0].function;
    let exp = f
        .expressions
        .iter()
        .find(|(_, e)| {
            matches!(
                e,
                Expression::Math {
                    fun: MathFunction::Exp2,
                    ..
                }
            )
        })
        .unwrap()
        .0;
    for (h, e) in f.expressions.iter() {
        if h == exp {
            f.expression_controls.insert(h, EXP);
        } else if matches!(e,Expression::Binary{op:BinaryOperator::Multiply,left,..} if *left==exp)
            || matches!(
                e,
                Expression::Math {
                    fun: MathFunction::Dot,
                    ..
                }
            )
        {
            f.expression_controls
                .insert(h, ArithmeticControl::NoContraction);
        }
    }
    m
}
fn inst(w: &[u32]) -> Vec<&[u32]> {
    let mut p = 5;
    let mut r = vec![];
    while p < w.len() {
        let n = (w[p] >> 16) as usize;
        assert!(n > 0);
        r.push(&w[p..p + n]);
        p += n;
    }
    r
}
fn options() -> back::spv::Options<'static> {
    back::spv::Options {
        lang_version: (1, 5),
        ..Default::default()
    }
}
fn pipeline(name: &str) -> back::spv::PipelineOptions {
    back::spv::PipelineOptions {
        shader_stage: if name == "main" {
            ShaderStage::Fragment
        } else {
            ShaderStage::Compute
        },
        entry_point: name.into(),
    }
}
fn emit(m: &Module) -> Vec<u32> {
    back::spv::write_vec(m, &validate(m), &options(), Some(&pipeline("main"))).unwrap()
}
fn import(w: &[u32]) -> Module {
    front::spv::Frontend::new(w.iter().copied(), &Default::default())
        .parse()
        .unwrap()
}
fn check(w: &[u32]) -> Module {
    let ins = inst(w);
    let decorations: Vec<_> = ins
        .iter()
        .filter(|i| i[0] & 65535 == 71 && i[2] == 40 && i[3] == 0)
        .collect();
    assert_eq!(decorations.len(), 1);
    let target = decorations[0][1];
    explicit_form(w);
    let exp = ins
        .iter()
        .find(|i| i[0] & 65535 == 12 && i[2] == target)
        .unwrap();
    assert_eq!(exp.len(), 6);
    assert_eq!(exp[4], 29);
    assert!(ins
        .iter()
        .any(|i| i[0] & 65535 == 22 && i[1] == exp[1] && i[2] == 32));
    let set = ins
        .iter()
        .find(|i| i[0] & 65535 == 11 && i[1] == exp[3])
        .unwrap();
    let bytes: Vec<_> = set[2..].iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(bytes.starts_with(b"GLSL.std.450\0"));
    // The controlled producer's sole input is exactly the ordinary x*x result.
    let up = ins
        .iter()
        .find(|i| i[0] & 65535 == 133 && i[2] == exp[5])
        .unwrap();
    assert_eq!(up[3], up[4]);
    assert!(!ins
        .iter()
        .any(|i| i[0] & 65535 == 71 && i[1] == up[2] && (i[2] == 40 || i[2] == 42)));
    let consumer = ins
        .iter()
        .find(|i| i[0] & 65535 == 133 && i[3] == target)
        .unwrap();
    assert_eq!(consumer[4], up[3]);
    assert!(ins
        .iter()
        .any(|i| i[0] & 65535 == 71 && i[1] == consumer[2] && i[2] == 40 && i[3] == 7));
    assert!(!ins
        .iter()
        .any(|i| i[0] & 65535 == 71 && i[1] == target && i[2] == 42));
    let ordinary = ins
        .iter()
        .find(|i| i[0] & 65535 == 12 && i[4] == 29 && i[2] != target)
        .unwrap();
    assert_eq!(ordinary[5], up[3]);
    assert!(!ins
        .iter()
        .any(|i| i[0] & 65535 == 71 && i[1] == ordinary[2] && (i[2] == 40 || i[2] == 42)));
    assert!(ins.iter().any(|i| i[0] & 65535 == 17 && i[1] == 6029));
    let im = import(w);
    validate(&im);
    let f = im
        .functions
        .iter()
        .map(|(_, f)| f)
        .find(|f| f.expression_controls.values().any(|c| *c == EXP))
        .unwrap();
    let h = *f
        .expression_controls
        .iter()
        .find(|(_, c)| **c == EXP)
        .unwrap()
        .0;
    let Expression::Math {
        fun: MathFunction::Exp2,
        arg,
        ..
    } = f.expressions[h]
    else {
        panic!()
    };
    assert!(
        matches!(f.expressions[arg],Expression::Binary{op:BinaryOperator::Multiply,left,right}if left==right)
    );
    assert!(f.expressions.iter().any(
        |(p, e)| matches!(e,Expression::Binary{op:BinaryOperator::Multiply,left,..}if *left==h)
            && f.expression_controls.get(&p) == Some(&ArithmeticControl::NoContraction)
    ));
    im
}
/// A module that declares FloatControls2 is in the explicit form: every entry
/// point sets `FPFastMathDefault` 0x7000F for every float width, and nothing
/// is decorated `NoContraction`.
fn explicit_form(w: &[u32]) {
    let ins = inst(w);
    assert!(ins.iter().any(|i| i[0] & 65535 == 17 && i[1] == 6029));
    assert!(!ins.iter().any(|i| i[0] & 65535 == 71 && i[2] == 42));
    let floats: Vec<_> = ins.iter().filter(|i| i[0] & 65535 == 22).collect();
    assert!(!floats.is_empty());
    let defaults: Vec<_> = ins.iter().filter(|i| i[0] & 65535 == 331).collect();
    let entries: Vec<_> = ins.iter().filter(|i| i[0] & 65535 == 15).collect();
    assert_eq!(defaults.len(), entries.len() * floats.len());
    for entry in &entries {
        for float in &floats {
            let default = defaults
                .iter()
                .find(|i| i[1] == entry[2] && i[3] == float[1])
                .unwrap();
            assert_eq!((default.len(), default[2]), (5, 6028));
            let mask = ins
                .iter()
                .find(|i| i[0] & 65535 == 43 && i[2] == default[4])
                .unwrap();
            assert_eq!(mask[3], 0x7000F);
            assert!(ins
                .iter()
                .any(|i| i[0] & 65535 == 21 && i[1] == mask[1] && i[2] == 32));
        }
    }
}
fn spirv_val(w: &[u32]) {
    use std::io::Write as _;
    let mut child = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .stdin(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect(
            "Failed to execute spirv-val. It can be installed \
            by installing the Vulkan SDK and adding it to your path.",
        );
    let bytes: Vec<_> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "spirv-val rejected naga output:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn refused(w: Vec<u32>) -> bool {
    front::spv::Frontend::new(w.into_iter(), &Default::default())
        .parse()
        .is_err()
}
fn hlsl(m: &Module, sm: back::hlsl::ShaderModel) -> Result<String, back::hlsl::Error> {
    let mut s = String::new();
    let opt = back::hlsl::Options {
        shader_model: sm,
        ..Default::default()
    };
    let pipe = back::hlsl::PipelineOptions {
        entry_point: Some((ShaderStage::Fragment, "main".into())),
        ..Default::default()
    };
    back::hlsl::Writer::new(&mut s, &opt, &pipe).write(m, &validate(m), None)?;
    Ok(s)
}
fn refusals(m: &Module, w: &[u32]) {
    let f = &m.entry_points[0].function;
    let h = *f
        .expression_controls
        .iter()
        .find(|(_, c)| **c == EXP)
        .unwrap()
        .0;
    let mut extra_argument = m.clone();
    if let Expression::Math { arg, arg1, .. } =
        &mut extra_argument.entry_points[0].function.expressions[h]
    {
        *arg1 = Some(*arg);
    }
    assert!(
        valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
            .validate(&extra_argument)
            .is_err()
    );
    let mut bad = m.clone();
    bad.entry_points[0]
        .function
        .expression_controls
        .insert(h, ArithmeticControl::NoContraction);
    assert!(
        valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
            .validate(&bad)
            .is_err()
    );
    for target in ["sin", "vec2"] {
        let src = if target == "sin" {
            "@fragment fn main(@location(0)x:f32)->@location(0)f32{return sin(x);}"
        } else {
            "@fragment fn main(@location(0)x:vec2f)->@location(0)vec2f{return exp2(x);}"
        };
        let mut b = front::wgsl::parse_str(src).unwrap();
        let f = &mut b.entry_points[0].function;
        let h = f
            .expressions
            .iter()
            .find(|(_, e)| matches!(e, Expression::Math { .. }))
            .unwrap()
            .0;
        f.expression_controls.insert(h, EXP);
        assert!(
            valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
                .validate(&b)
                .is_err()
        );
    }
    assert!(hlsl(m, back::hlsl::ShaderModel::V5_1).is_err());
    assert!(
        back::msl::write_string(m, &validate(m), &Default::default(), &Default::default()).is_err()
    );
    let mut gl = String::new();
    let go = back::glsl::Options {
        version: back::glsl::Version::Desktop(450),
        ..Default::default()
    };
    let gp = back::glsl::PipelineOptions {
        shader_stage: ShaderStage::Fragment,
        entry_point: "main".into(),
        multiview: None,
    };
    let gi = validate(m);
    assert!(
        back::glsl::Writer::new(&mut gl, m, &gi, &go, &gp, Default::default())
            .unwrap()
            .write()
            .is_err()
    );
    let mut low = options();
    low.lang_version = (1, 0);
    assert!(back::spv::write_vec(m, &validate(m), &low, Some(&pipeline("main"))).is_err());
    let mut lowwords = w.to_vec();
    lowwords[1] = 0x10000;
    assert!(
        front::spv::Frontend::new(lowwords.into_iter(), &Default::default())
            .parse()
            .is_err()
    );
    assert!(back::wgsl::write_string(m, &validate(m), back::wgsl::WriterFlags::empty()).is_err());
    let mut opt = options();
    opt.capabilities = Some([back::spv::Capability::Shader].into_iter().collect());
    assert!(back::spv::write_vec(m, &validate(m), &opt, Some(&pipeline("main"))).is_err());
    // Independently mutate actual authored words, with no reliance on spirv-val.
    let chunks = inst(w);
    let decoration = chunks
        .iter()
        .find(|i| i[0] & 65535 == 71 && i.len() == 4 && i[2] == 40 && i[3] == 0)
        .unwrap();
    let target = decoration[1];
    for case in 0..15 {
        let mut bad = w[..5].to_vec();
        for i in &chunks {
            let mut v = i.to_vec();
            if i[0] & 65535 == 71 && i.len() == 4 && i[2] == 40 && i[1] == target {
                match case {
                    0 => v[3] = 1,
                    1 => v[1] = w[3] + 50,
                    2 => v[1] = chunks.iter().find(|i| i[0] & 65535 == 22).unwrap()[1],
                    _ => {}
                }
            }
            if case == 3 && i[0] & 65535 == 17 && i[1] == 6029 {
                continue;
            }
            if case == 4 && i[0] & 65535 == 10 {
                continue;
            }
            if i[0] & 65535 == 12 && i[2] == target {
                if case == 5 {
                    v[4] = 13;
                }
                if case == 6 {
                    v[4] = 30;
                }
            }
            if case == 9 && i[0] & 65535 == 71 && i[1] == target {
                v = vec![(5 << 16) | 72, target, 0, 40, 0];
            }
            if case == 11 && i[0] & 65535 == 12 && i[2] == target {
                v[0] += 1 << 16;
                v.push(i[5]);
            }
            if (case == 12 || case == 13) && i[0] & 65535 == 22 {
                v[2] = if case == 12 { 16 } else { 64 };
            }
            if case == 14 && i[0] & 65535 == 12 && i[2] == target {
                v[1] = chunks.iter().find(|i| i[0] & 65535 == 23).unwrap()[1];
            }
            bad.extend(v);
            if case == 8 && i[0] & 65535 == 71 && i[1] == target {
                bad.extend(i.iter().copied());
            }
            if case == 10 && i[0] & 65535 == 71 && i[1] == target {
                bad.extend([
                    (4 << 16) | 71,
                    chunks.iter().find(|i| i[0] & 65535 == 133).unwrap()[2],
                    40,
                    0,
                ]);
            }
            if case == 7 && i[0] & 65535 == 71 && i[1] == target {
                bad.extend([(3 << 16) | 71, target, 42]);
            }
        }
        assert!(
            front::spv::Frontend::new(bad.into_iter(), &Default::default())
                .parse()
                .is_err(),
            "case {case}"
        );
    }
}
#[cfg(all(feature = "serialize", feature = "deserialize"))]
#[test]
fn binary_round_trip_preserves_controls() {
    let module = fixture();
    let controls = &module.entry_points[0].function.expression_controls;
    assert!(controls.values().any(|control| *control == EXP));
    assert!(controls
        .values()
        .any(|control| *control == ArithmeticControl::NoContraction));
    let bytes = bincode::serde::encode_to_vec(&module, bincode::config::standard()).unwrap();
    let (decoded, consumed) =
        bincode::serde::decode_from_slice::<Module, _>(&bytes, bincode::config::standard())
            .unwrap();
    assert_eq!(consumed, bytes.len());
    validate(&decoded);
    assert_eq!(
        controls,
        &decoded.entry_points[0].function.expression_controls
    );
    assert!(decoded.entry_points[1]
        .function
        .expression_controls
        .is_empty());
    assert_eq!(
        bytes,
        bincode::serde::encode_to_vec(&decoded, bincode::config::standard()).unwrap()
    );
    assert_eq!(emit(&module), emit(&decoded));
}

#[test]
fn lifecycle() {
    let mut m = front::wgsl::parse_str("@compute @workgroup_size(1) fn main() {}").unwrap();
    let ty = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        },
        Span::UNDEFINED,
    );
    let make = || {
        let mut f = Function {
            arguments: vec![FunctionArgument {
                name: None,
                ty,
                binding: None,
            }],
            result: Some(FunctionResult { ty, binding: None }),
            ..Default::default()
        };
        f.expressions
            .append(Expression::Literal(Literal::F32(99.)), Span::UNDEFINED);
        let a = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        let start = f.expressions.len();
        let dead = f.expressions.append(
            Expression::Math {
                fun: MathFunction::Exp2,
                arg: a,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        let p = f.expressions.append(
            Expression::Math {
                fun: MathFunction::Exp2,
                arg: a,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        f.expression_controls.insert(dead, EXP);
        f.expression_controls.insert(p, EXP);
        f.body.push(
            Statement::Emit(f.expressions.range_from(start)),
            Span::UNDEFINED,
        );
        f.body
            .push(Statement::Return { value: Some(p) }, Span::UNDEFINED);
        f
    };
    m.functions.append(make(), Span::UNDEFINED);
    let live = m.functions.append(make(), Span::UNDEFINED);
    let f = &mut m.entry_points[0].function;
    let a = f
        .expressions
        .append(Expression::Literal(Literal::F32(2.)), Span::UNDEFINED);
    let result = f
        .expressions
        .append(Expression::CallResult(live), Span::UNDEFINED);
    f.body = Block::new();
    f.body.push(
        Statement::Call {
            function: live,
            arguments: vec![a],
            result: Some(result),
        },
        Span::UNDEFINED,
    );
    validate(&m);
    compact::compact(&mut m, compact::KeepUnused::No);
    validate(&m);
    assert_eq!(m.functions.len(), 1);
    let (owner, f) = m.functions.iter().next().unwrap();
    assert_eq!(owner.index(), 0);
    assert_ne!(owner, live);
    assert_eq!(f.expression_controls.len(), 1);
    let h = *f.expression_controls.keys().next().unwrap();
    assert_eq!(h.index(), 1);
    assert!(
        matches!(f.expressions[h],Expression::Math{fun:MathFunction::Exp2,arg,..}if arg.index()==0)
    );

    let mut s = front::wgsl::parse_str(
        "override x:f32=2.0; @fragment fn main()->@location(0)f32{return exp2(x);}",
    )
    .unwrap();
    let f = &mut s.entry_points[0].function;
    let h = f
        .expressions
        .iter()
        .find(|(_, e)| {
            matches!(
                e,
                Expression::Math {
                    fun: MathFunction::Exp2,
                    ..
                }
            )
        })
        .unwrap()
        .0;
    f.expression_controls.insert(h, EXP);
    let info = validate(&s);
    let (special, _) = back::pipeline_constants::process_overrides(
        &s,
        &info,
        Some((ShaderStage::Fragment, "main")),
        &Default::default(),
    )
    .unwrap();
    assert!(!special.entry_points[0]
        .function
        .expressions
        .iter()
        .any(|(_, e)| matches!(e, Expression::Override(_))));
    let f = &special.entry_points[0].function;
    let h = *f.expression_controls.keys().next().unwrap();
    assert!(matches!(
        f.expressions[h],
        Expression::Math {
            fun: MathFunction::Exp2,
            ..
        }
    ));
    let mut bad = special.clone().into_owned();
    let f = &mut bad.entry_points[0].function;
    let ty = bad.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        },
        Span::UNDEFINED,
    );
    let local = f.local_variables.append(
        LocalVariable {
            name: None,
            ty,
            init: Some(h),
        },
        Span::UNDEFINED,
    );
    assert!(
        valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
            .validate(&bad)
            .is_err()
    );
    bad.entry_points[0].function.local_variables[local].init = None;
    let f = &mut bad.entry_points[0].function;
    let ptr = f
        .expressions
        .append(Expression::LocalVariable(local), Span::UNDEFINED);
    let mut body = Block::new();
    for st in f.body.iter() {
        if !matches!(st, Statement::Return { .. }) {
            body.push(st.clone(), Span::UNDEFINED);
        }
    }
    body.push(
        Statement::Store {
            pointer: ptr,
            value: h,
        },
        Span::UNDEFINED,
    );
    body.push(Statement::Return { value: Some(h) }, Span::UNDEFINED);
    f.body = body;
    validate(&bad);
}
#[test]
fn selection_and_types() {
    // A valid Exp2 shape of any other scalar width is not this transport tag.
    for scalar in [Scalar::F16, Scalar::F64] {
        let mut m = Module::default();
        let ty = m.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(scalar),
            },
            Span::UNDEFINED,
        );
        let mut f = Function {
            arguments: vec![FunctionArgument {
                name: None,
                ty,
                binding: None,
            }],
            result: Some(FunctionResult { ty, binding: None }),
            ..Default::default()
        };
        let a = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        let h = f.expressions.append(
            Expression::Math {
                fun: MathFunction::Exp2,
                arg: a,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Emit(f.expressions.range_from(1)),
            Span::UNDEFINED,
        );
        f.body
            .push(Statement::Return { value: Some(h) }, Span::UNDEFINED);
        f.expression_controls.insert(h, EXP);
        m.functions.append(f, Span::UNDEFINED);
        let error =
            valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
                .validate(&m)
                .unwrap_err();
        assert!(format!("{error:?}").contains("InvalidArithmeticControl"));
    }
    let mut m = fixture();
    let mut f = m.entry_points[0].function.clone();
    f.name = Some("unused_body".into());
    f.arguments[0].binding = None;
    f.result.as_mut().unwrap().binding = None;
    let _h = m.functions.append(f, Span::UNDEFINED);
    // HLSL selects only reachable helpers; SPIR-V retains compatible unused
    // functions under its preexisting policy and must control their actual output.
    validate(&m);
    let hs = hlsl(&m, back::hlsl::ShaderModel::V6_0).unwrap();
    assert!(!hs.contains("unused_body"));
    assert_eq!(hs.matches("[noinline]").count(), 3);
    let w = emit(&m);
    let ins = inst(&w);
    let targets: Vec<_> = ins
        .iter()
        .filter(|i| i[0] & 65535 == 71 && i[2] == 40 && i[3] == 0)
        .map(|i| i[1])
        .collect();
    assert_eq!(targets.len(), 2);
    for id in targets {
        assert!(ins
            .iter()
            .any(|i| i[0] & 65535 == 12 && i[2] == id && i[4] == 29));
    }
}
#[test]
fn transport_and_refusals() {
    let m = fixture();
    let words = emit(&m);
    let imported = check(&words);
    let rewritten =
        back::spv::write_vec(&imported, &validate(&imported), &options(), None).unwrap();
    check(&rewritten);
    refusals(&m, &words);
    let empty =
        back::spv::write_vec(&m, &validate(&m), &options(), Some(&pipeline("empty"))).unwrap();
    assert!(!inst(&empty)
        .iter()
        .any(|i| i[0] & 65535 == 17 && i[1] == 6029));
    assert!(!inst(&empty)
        .iter()
        .any(|i| i[0] & 65535 == 71 && i[2] == 40));
}
fn mutate(w: &[u32], mut f: impl FnMut(&[u32]) -> Option<Vec<u32>>) -> Vec<u32> {
    let mut out = w[..5].to_vec();
    for i in inst(w) {
        out.extend(f(i).unwrap_or_else(|| i.to_vec()));
    }
    out
}
fn controls(m: &Module) -> Vec<ArithmeticControl> {
    let mut all: Vec<_> = m
        .entry_points
        .iter()
        .map(|e| &e.function)
        .chain(m.functions.iter().map(|(_, f)| f))
        .flat_map(|f| f.expression_controls.values().copied())
        .collect();
    all.sort_by_key(|c| *c == EXP);
    all
}
// miri cannot shell out to spirv-val
#[cfg_attr(miri, ignore)]
#[test]
fn explicit_float_controls2_form() {
    use ArithmeticControl::NoContraction as NOC;
    let m = fixture();
    let all = back::spv::write_vec(&m, &validate(&m), &options(), None).unwrap();
    explicit_form(&all);
    spirv_val(&all);
    let words = emit(&m);
    explicit_form(&words);
    spirv_val(&words);
    assert_eq!(controls(&import(&words)), [NOC, NOC, EXP]);

    // Without the Exp2 control nothing declares FloatControls2, and the
    // module keeps the plain NoContraction form.
    let mut plain = fixture();
    plain.entry_points[0]
        .function
        .expression_controls
        .retain(|_, c| *c != EXP);
    let legacy = emit(&plain);
    spirv_val(&legacy);
    let ins = inst(&legacy);
    assert!(!ins.iter().any(|i| i[0] & 65535 == 17 && i[1] == 6029));
    assert!(!ins.iter().any(|i| i[0] & 65535 == 331));
    assert!(!ins.iter().any(|i| i[0] & 65535 == 71 && i[2] == 40));
    assert_eq!(
        ins.iter()
            .filter(|i| i[0] & 65535 == 71 && i[2] == 42)
            .count(),
        2
    );
    assert_eq!(controls(&import(&legacy)), [NOC, NOC]);

    // Every float width gets a default. An integer result cannot carry
    // FPFastMathMode, and its NoContraction is dropped: the wrapping
    // operation is exact.
    let mut mixed = front::wgsl::parse_str(
        "enable f16;
@fragment fn main(@location(0) x:f32,@location(1) @interpolate(flat) i:i32)->@location(0) vec4f {
 let n=i*i; let h=f16(x)*f16(x); let e=exp2(x); let p=x*x;
 return vec4f(f32(n),f32(h),e,p);
}",
    )
    .unwrap();
    let f = &mut mixed.entry_points[0].function;
    let muls: Vec<_> = f
        .expressions
        .iter()
        .filter(|(_, e)| {
            matches!(
                e,
                Expression::Binary {
                    op: BinaryOperator::Multiply,
                    ..
                }
            )
        })
        .map(|(h, _)| h)
        .collect();
    assert_eq!(muls.len(), 3);
    for h in muls {
        f.expression_controls.insert(h, NOC);
    }
    let legacy = emit(&mixed);
    spirv_val(&legacy);
    let ins = inst(&legacy);
    assert!(!ins.iter().any(|i| i[0] & 65535 == 17 && i[1] == 6029));
    let decorated = |ins: &[&[u32]], op: u32, decoration: &[u32]| {
        ins.iter()
            .filter(|i| i[0] & 65535 == op)
            .filter(|i| {
                ins.iter()
                    .any(|d| d[0] & 65535 == 71 && d[1] == i[2] && d[2..] == *decoration)
            })
            .count()
    };
    assert_eq!(decorated(&ins, 132, &[42]), 1);
    assert_eq!(decorated(&ins, 133, &[42]), 2);
    let f = &mut mixed.entry_points[0].function;
    let exp = f
        .expressions
        .iter()
        .find(|(_, e)| matches!(e, Expression::Math { .. }))
        .unwrap()
        .0;
    f.expression_controls.insert(exp, EXP);
    let explicit = emit(&mixed);
    explicit_form(&explicit);
    spirv_val(&explicit);
    let ins = inst(&explicit);
    assert_eq!(ins.iter().filter(|i| i[0] & 65535 == 22).count(), 2);
    assert_eq!(decorated(&ins, 133, &[40, 7]), 2);
    let imul = ins.iter().find(|i| i[0] & 65535 == 132).unwrap();
    assert!(!ins.iter().any(|i| i[0] & 65535 == 71 && i[1] == imul[2]));
    // The writer also declares StorageInputOutput16 for an f16 module.
    let lenient = front::spv::Options {
        strict_capabilities: false,
        ..Default::default()
    };
    let imported = front::spv::Frontend::new(explicit.iter().copied(), &lenient)
        .parse()
        .unwrap();
    assert_eq!(controls(&imported), [NOC, NOC, EXP]);

    // Import keeps what it can map and refuses the rest.
    let default = |i: &[u32]| i[0] & 65535 == 331;
    let mul_control = |i: &[u32]| i[0] & 65535 == 71 && i[2] == 40 && i[3] == 7;
    // No default at all is the client-API form and keeps its meaning.
    let no_default = mutate(&words, |i| default(i).then(Vec::new));
    assert_eq!(controls(&import(&no_default)), [NOC, NOC, EXP]);
    // A default that withholds a permission, or that is missing or repeated
    // for an entry point.
    for mask in [0, 7, 0x3000F, 0x70007] {
        assert!(refused(mutate(&words, |i| {
            (i[0] & 65535 == 43 && i[3] == 0x7000F).then(|| vec![i[0], i[1], i[2], mask])
        })));
    }
    let mut seen = false;
    assert!(refused(mutate(&all, |i| {
        (default(i) && !core::mem::replace(&mut seen, true)).then(Vec::new)
    })));
    let mut seen = false;
    assert!(refused(mutate(&words, |i| {
        (default(i) && !core::mem::replace(&mut seen, true)).then(|| [i, i].concat())
    })));
    // A default whose target is not a float type, and another execution mode.
    let uint = inst(&words).iter().find(|i| i[0] & 65535 == 21).unwrap()[1];
    assert!(refused(mutate(&words, |i| {
        default(i).then(|| vec![i[0], i[1], i[2], uint, i[4]])
    })));
    assert!(refused(mutate(&words, |i| {
        default(i).then(|| vec![i[0], i[1], 38, i[3], i[4]])
    })));
    // NoContraction next to a default, and per-instruction masks other than
    // the two forms above.
    let mut seen = false;
    assert!(refused(mutate(&words, |i| {
        (mul_control(i) && !core::mem::replace(&mut seen, true))
            .then(|| vec![(3 << 16) | 71, i[1], 42])
    })));
    for mask in [1, 0xF, 0x10007, 0x7000F] {
        let mut seen = false;
        assert!(refused(mutate(&words, |i| {
            (mul_control(i) && !core::mem::replace(&mut seen, true))
                .then(|| vec![i[0], i[1], 40, mask])
        })));
    }
    assert!(refused(mutate(&words, |i| {
        (i[0] & 65535 == 71 && i[2] == 40 && i[3] == 0).then(|| vec![i[0], i[1], 40, 7])
    })));
    // The explicit form needs the capability, the extension and SPIR-V 1.2.
    assert!(refused(mutate(&words, |i| {
        (i[0] & 65535 == 17 && i[1] == 6029).then(Vec::new)
    })));
    assert!(refused(
        mutate(&words, |i| (i[0] & 65535 == 10).then(Vec::new))
    ));
    let mut old = words.clone();
    old[1] = 0x10100;
    assert!(refused(old));
}
