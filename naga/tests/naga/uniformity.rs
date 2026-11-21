//! Tests for the WGSL specification's uniformity analysis.

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate(source: &str) -> Result<(), String> {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(err) => panic!("WGSL parse failed:\n{}", err.emit_to_string(source)),
    };
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map(|_| ())
        .map_err(|err| err.emit_to_string(source))
}

#[track_caller]
fn valid(source: &str) {
    if let Err(err) = validate(source) {
        panic!("expected the shader to be valid, got:\n{err}");
    }
}

#[track_caller]
fn invalid(source: &str, expected: &str) {
    match validate(source) {
        Ok(()) => panic!("expected a uniformity error containing {expected:?}"),
        Err(err) => assert!(
            err.contains(expected),
            "expected an error containing {expected:?}, got:\n{err}"
        ),
    }
}

const FRAGMENT_BINDINGS: &str = "
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<uniform> u: f32;
";

const COMPUTE_BINDINGS: &str = "
@group(0) @binding(0) var<storage, read> ro: array<u32>;
@group(0) @binding(1) var<storage, read_write> rw: array<u32>;
var<workgroup> wg: u32;
var<private> p: u32;
";

fn fragment(body: &str) -> String {
    format!(
        "{FRAGMENT_BINDINGS}
@fragment
fn main(@location(0) uv: vec2f) -> @location(0) vec4f {{
    {body}
    return vec4f();
}}"
    )
}

fn compute(body: &str) -> String {
    format!(
        "{COMPUTE_BINDINGS}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3u) {{
    {body}
}}"
    )
}

const SAMPLE_REQUIRES: &str =
    "Uniform control flow is required for a texture sample with an implicit level of detail";
const BARRIER_REQUIRES: &str = "Uniform control flow is required for `workgroupBarrier`";

#[test]
fn texture_sample() {
    invalid(
        &fragment("if uv.x > 0.5 { _ = textureSample(t, s, uv); }"),
        SAMPLE_REQUIRES,
    );
    invalid(
        &fragment("if uv.x > 0.5 { _ = dpdx(uv.x); }"),
        "Uniform control flow is required for a derivative",
    );
    valid(&fragment("if u > 0.5 { _ = textureSample(t, s, uv); }"));
    valid(&fragment(
        "if uv.x > 0.5 { _ = textureSampleLevel(t, s, uv, 0.0); }",
    ));
    valid(&fragment("_ = textureSample(t, s, uv);"));
}

#[test]
fn derivative_uniformity_filter() {
    valid(&format!(
        "diagnostic(off, derivative_uniformity);
        {}",
        fragment("if uv.x > 0.5 { _ = textureSample(t, s, uv); }")
    ));
    valid(&format!(
        "diagnostic(warning, derivative_uniformity);
        {}",
        fragment("if uv.x > 0.5 { _ = textureSample(t, s, uv); }")
    ));
    valid(&format!(
        "{FRAGMENT_BINDINGS}
        @diagnostic(off, derivative_uniformity)
        fn helper(uv: vec2f) -> vec4f {{ return textureSample(t, s, uv); }}
        @fragment
        fn main(@location(0) uv: vec2f) -> @location(0) vec4f {{
            if uv.x > 0.5 {{ return helper(uv); }}
            return vec4f();
        }}"
    ));
}

#[test]
fn barrier() {
    invalid(
        &compute("if lid > 3u { workgroupBarrier(); }"),
        BARRIER_REQUIRES,
    );
    invalid(
        &compute("if rw[0] > 0u { storageBarrier(); }"),
        "`storageBarrier`",
    );
    valid(&compute("if ro[0] > 0u { workgroupBarrier(); }"));
    valid(&compute("if wid.x > 0u { workgroupBarrier(); }"));
    // The diagnostic for synchronization functions can't be filtered.
    invalid(
        &format!(
            "diagnostic(off, derivative_uniformity);
            {}",
            compute("if lid > 3u { workgroupBarrier(); }")
        ),
        BARRIER_REQUIRES,
    );
}

#[test]
fn barrier_without_uniformity_analysis() {
    let module =
        naga::front::wgsl::parse_str(&compute("if lid > 3u { workgroupBarrier(); }")).unwrap();
    let flags = ValidationFlags::all() - ValidationFlags::CONTROL_FLOW_UNIFORMITY;
    Validator::new(flags, Capabilities::all())
        .validate(&module)
        .unwrap();
}

#[test]
fn mutable_globals() {
    invalid(
        &compute("if wg > 0u { workgroupBarrier(); }"),
        BARRIER_REQUIRES,
    );
    invalid(
        &compute("if p > 0u { workgroupBarrier(); }"),
        BARRIER_REQUIRES,
    );
    invalid(
        &compute("if arrayLength(&rw) > 0u { workgroupBarrier(); }"),
        BARRIER_REQUIRES,
    );
    valid(&compute("if arrayLength(&ro) > 0u { workgroupBarrier(); }"));
    valid(&compute(
        "if workgroupUniformLoad(&wg) > 0u { workgroupBarrier(); }",
    ));
    invalid(
        &compute("if lid > 0u { _ = workgroupUniformLoad(&wg); }"),
        "Uniform control flow is required for `workgroupUniformLoad`",
    );
}

#[test]
fn reconvergence() {
    valid(&compute(
        "if lid > 3u { rw[lid] = 0u; }
        workgroupBarrier();",
    ));
    invalid(
        &compute(
            "if lid > 3u { return; }
            workgroupBarrier();",
        ),
        BARRIER_REQUIRES,
    );
    invalid(
        &compute("switch lid { case 0u: { workgroupBarrier(); } default: {} }"),
        BARRIER_REQUIRES,
    );
    valid(&compute(
        "switch lid { case 0u: { rw[0] = 1u; } default: {} }
        workgroupBarrier();",
    ));
}

#[test]
fn unreachable_code() {
    valid(&compute(
        "if lid > 3u {
            return;
            workgroupBarrier();
        }",
    ));
}

#[test]
fn loops() {
    // https://github.com/gfx-rs/wgpu/issues/4316
    valid(&compute(
        "for (var i = 0u; i < 4u; i++) { workgroupBarrier(); }",
    ));
    valid(&compute(
        "var i = 0u;
        loop {
            if i >= 4u { break; }
            workgroupBarrier();
            i++;
        }",
    ));
    valid(&compute(
        "loop { if lid > 3u { break; } }
        workgroupBarrier();",
    ));
    invalid(
        &compute(
            "loop {
                if lid > 3u { break; }
                workgroupBarrier();
            }",
        ),
        BARRIER_REQUIRES,
    );
    // The value of `x` only becomes non-uniform on the second iteration.
    invalid(
        &compute(
            "var x = 0u;
            loop {
                if x > 10u { break; }
                workgroupBarrier();
                x = lid;
            }",
        ),
        BARRIER_REQUIRES,
    );
    invalid(
        &compute(
            "var i = 0u;
            loop {
                workgroupBarrier();
                continuing {
                    i = i + lid;
                    break if i > 4u;
                }
            }",
        ),
        BARRIER_REQUIRES,
    );
}

#[test]
fn local_variables() {
    invalid(
        &compute(
            "var x = 0u;
            if lid > 3u { x = 1u; }
            if x == 0u { workgroupBarrier(); }",
        ),
        BARRIER_REQUIRES,
    );
    valid(&compute(
        "var x = lid;
        x = 0u;
        if x == 0u { workgroupBarrier(); }",
    ));
    invalid(
        &compute(
            "var x = 0u;
            let ptr = &x;
            *ptr = lid;
            if x == 0u { workgroupBarrier(); }",
        ),
        BARRIER_REQUIRES,
    );
    // Partial assignments keep the previous value.
    invalid(
        &compute(
            "var v = vec2u(lid);
            v.x = 0u;
            if v.x == 0u { workgroupBarrier(); }",
        ),
        BARRIER_REQUIRES,
    );
    valid(&compute(
        "var v = vec2u(lid);
        v = vec2u(0u);
        if v.x == 0u { workgroupBarrier(); }",
    ));
}

#[test]
fn short_circuit() {
    invalid(
        &fragment("if uv.x > 0.5 && textureSample(t, s, uv).x > 0.5 { discard; }"),
        SAMPLE_REQUIRES,
    );
    valid(&fragment(
        "if u > 0.5 && textureSample(t, s, uv).x > 0.5 { discard; }",
    ));
}

#[test]
fn discard_does_not_affect_uniformity() {
    valid(&fragment(
        "if uv.x > 0.5 { discard; }
        _ = textureSample(t, s, uv);",
    ));
}

#[test]
fn function_calls() {
    let barrier_fn = "fn sync() { workgroupBarrier(); }";
    invalid(
        &format!("{barrier_fn}{}", compute("if lid > 3u { sync(); }")),
        "Uniform control flow is required for a call to `sync`",
    );
    valid(&format!("{barrier_fn}{}", compute("sync();")));

    let conditional = "fn sync_if(c: bool) { if c { workgroupBarrier(); } }";
    invalid(
        &format!("{conditional}{}", compute("sync_if(lid == 0u);")),
        "Argument 0 of a call to `sync_if` must be a uniform value",
    );
    valid(&format!(
        "{conditional}{}",
        compute("sync_if(wid.x == 0u);")
    ));

    let identity = "fn id(x: u32) -> u32 { return x; }";
    invalid(
        &format!(
            "{identity}{}",
            compute("if id(lid) > 0u { workgroupBarrier(); }")
        ),
        BARRIER_REQUIRES,
    );
    valid(&format!(
        "{identity}{}",
        compute("if id(wid.x) > 0u { workgroupBarrier(); }")
    ));

    let non_uniform = "fn load_rw() -> u32 { return rw[0]; }";
    invalid(
        &format!(
            "{non_uniform}{}",
            compute("if load_rw() > 0u { workgroupBarrier(); }")
        ),
        "because of the result of a call to `load_rw`",
    );
}

#[test]
fn pointer_parameters() {
    let store = "fn store(dst: ptr<function, u32>, value: u32) { *dst = value; }";
    invalid(
        &format!(
            "{store}{}",
            compute(
                "var x = 0u;
                store(&x, lid);
                if x == 0u { workgroupBarrier(); }"
            )
        ),
        BARRIER_REQUIRES,
    );
    valid(&format!(
        "{store}{}",
        compute(
            "var x = lid;
            store(&x, 1u);
            if x == 0u { workgroupBarrier(); }"
        )
    ));

    let keep = "fn keep(src: ptr<function, u32>) { _ = *src; }";
    valid(&format!(
        "{keep}{}",
        compute(
            "var x = 0u;
            if lid > 3u { keep(&x); }
            if x == 0u { workgroupBarrier(); }"
        )
    ));
    invalid(
        &format!(
            "{keep}{}",
            compute(
                "var x = lid;
                keep(&x);
                if x == 0u { workgroupBarrier(); }"
            )
        ),
        BARRIER_REQUIRES,
    );

    let sync_if = "fn sync_if(c: ptr<function, bool>) { if *c { workgroupBarrier(); } }";
    invalid(
        &format!(
            "{sync_if}{}",
            compute(
                "var c = lid == 0u;
                sync_if(&c);"
            )
        ),
        "The value that argument 0 of a call to `sync_if` points to must be uniform",
    );
}

#[test]
fn entry_point_inputs() {
    let source = |input: &str| {
        format!(
            "struct In {{ {input} }}
            @compute @workgroup_size(64)
            fn main(input: In) {{
                if input.a.x > 0u {{ workgroupBarrier(); }}
            }}"
        )
    };
    valid(&source("@builtin(workgroup_id) a: vec3u"));
    invalid(
        &source("@builtin(workgroup_id) a: vec3u, @builtin(local_invocation_index) b: u32"),
        "because of the built-in value `local_invocation_index`",
    );
}

#[test]
fn storage_texture_load() {
    let source = |access: &str| {
        format!(
            "@group(0) @binding(0) var tex: texture_storage_2d<r32uint, {access}>;
            @compute @workgroup_size(64)
            fn main() {{
                if textureLoad(tex, vec2u()).x > 0u {{ workgroupBarrier(); }}
            }}"
        )
    };
    valid(&source("read"));
    invalid(&source("read_write"), BARRIER_REQUIRES);
}

#[test]
fn subgroups() {
    let shader = compute("if lid > 3u { _ = subgroupAdd(lid); }");
    invalid(
        &shader,
        "Uniform control flow is required for a subgroup operation",
    );
    valid(&format!("diagnostic(off, subgroup_uniformity);\n{shader}"));
    invalid(
        &compute("if subgroupAdd(1u) > 1u { workgroupBarrier(); }"),
        "because of the result of a subgroup operation",
    );
    invalid(
        &compute("_ = subgroupShuffleUp(lid, lid);"),
        "Argument 1 of a subgroup operation must be a uniform value",
    );
    valid(&compute("_ = subgroupShuffleUp(lid, wid.x);"));
}

#[test]
fn error_spans() {
    let source = compute(
        "let cond = lid > 3u;
    if cond { workgroupBarrier(); }",
    );
    let err = validate(&source).unwrap_err();
    for label in [
        "requires uniform control flow",
        "control flow depends on this value",
        "this value may be non-uniform",
    ] {
        assert!(err.contains(label), "missing {label:?} in:\n{err}");
    }
}
