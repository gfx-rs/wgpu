use super::parse_str;

#[test]
fn parse_comment() {
    parse_str(
        "//
        ////
        ///////////////////////////////////////////////////////// asda
        //////////////////// dad ////////// /
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        //
    ",
    )
    .unwrap();
}

#[test]
fn parse_types() {
    parse_str("const a : i32 = 2;").unwrap();
    parse_str("const a : u64 = 2lu;").unwrap();
    assert!(parse_str("const a : x32 = 2;").is_err());
    parse_str("var t: texture_2d<f32>;").unwrap();
    parse_str("var t: texture_cube_array<i32>;").unwrap();
    parse_str("var t: texture_multisampled_2d<u32>;").unwrap();
    parse_str("var t: texture_storage_1d<rgba8uint,write>;").unwrap();
    parse_str("var t: texture_storage_3d<r32float,read>;").unwrap();
}

#[test]
fn parse_type_inference() {
    parse_str(
        "
        fn foo() {
            let a = 2u;
            let b: u32 = a;
            var x = 3.;
            var y = vec2<f32>(1, 2);
        }",
    )
    .unwrap();
    assert!(parse_str(
        "
        fn foo() { let c : i32 = 2.0; }",
    )
    .is_err());
}

#[test]
fn parse_type_cast() {
    parse_str(
        "
        const a : i32 = 2;
        fn main() {
            var x: f32 = f32(a);
            x = f32(i32(a + 1) / 2);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            let x: vec2<f32> = vec2<f32>(1.0, 2.0);
            let y: vec2<u32> = vec2<u32>(x);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            let x: vec2<f32> = vec2<f32>(0.0);
        }
    ",
    )
    .unwrap();
    assert!(parse_str(
        "
        fn main() {
            let x: vec2<f32> = vec2<f32>(0i, 0i);
        }
    ",
    )
    .is_err());
}

#[test]
fn parse_struct() {
    parse_str(
        "
        struct Foo { x: i32 }
        struct Bar {
            @size(16) x: vec2<i32>,
            @align(16) y: f32,
            @size(32) @align(128) z: vec3<f32>,
        };
        struct Empty {}
        var<storage,read_write> s: Foo;
    ",
    )
    .unwrap();
}

#[test]
fn parse_standard_fun() {
    parse_str(
        "
        fn main() {
            var x: i32 = min(max(1, 2), 3);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_statement() {
    parse_str(
        "
        fn main() {
            ;
            {}
            {;}
        }
    ",
    )
    .unwrap();

    parse_str(
        "
        fn foo() {}
        fn bar() { foo(); }
    ",
    )
    .unwrap();
}

#[test]
fn parse_if() {
    parse_str(
        "
        fn main() {
            if true {
                discard;
            } else {}
            if 0 != 1 {}
            if false {
                return;
            } else if true {
                return;
            } else {}
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_parentheses_if() {
    parse_str(
        "
        fn main() {
            if (true) {
                discard;
            } else {}
            if (0 != 1) {}
            if (false) {
                return;
            } else if (true) {
                return;
            } else {}
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_loop() {
    parse_str(
        "
        fn main() {
            var i: i32 = 0;
            loop {
                if i == 1 { break; }
                continuing { i = 1; }
            }
            loop {
                if i == 0 { continue; }
                break;
            }
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            var found: bool = false;
            var i: i32 = 0;
            while !found {
                if i == 10 {
                    found = true;
                }

                i = i + 1;
            }
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            while true {
                break;
            }
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            var a: i32 = 0;
            for(var i: i32 = 0; i < 4; i = i + 1) {
                a = a + 2;
            }
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        fn main() {
            for(;;) {
                break;
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_switch() {
    parse_str(
        "
        fn main() {
            var pos: f32;
            switch (3) {
                case 0, 1: { pos = 0.0; }
                case 2: { pos = 1.0; }
                default: { pos = 3.0; }
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_switch_optional_colon_in_case() {
    parse_str(
        "
        fn main() {
            var pos: f32;
            switch (3) {
                case 0, 1 { pos = 0.0; }
                case 2 { pos = 1.0; }
                default { pos = 3.0; }
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_switch_default_in_case() {
    parse_str(
        "
        fn main() {
            var pos: f32;
            switch (3) {
                case 0, 1: { pos = 0.0; }
                case 2: {}
                case default, 3: { pos = 3.0; }
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_parentheses_switch() {
    parse_str(
        "
        fn main() {
            var pos: f32;
            switch pos > 1.0 {
                default: { pos = 3.0; }
            }
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_texture_load() {
    parse_str(
        "
        var t: texture_3d<u32>;
        fn foo() {
            let r: vec4<u32> = textureLoad(t, vec3<u32>(0u, 1u, 2u), 1);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        var t: texture_multisampled_2d_array<i32>;
        fn foo() {
            let r: vec4<i32> = textureLoad(t, vec2<i32>(10, 20), 2, 3);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        var t: texture_storage_1d_array<r32float,read>;
        fn foo() {
            let r: vec4<f32> = textureLoad(t, 10, 2);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_texture_store() {
    parse_str(
        "
        var t: texture_storage_2d<rgba8unorm,write>;
        fn foo() {
            textureStore(t, vec2<i32>(10, 20), vec4<f32>(0.0, 1.0, 2.0, 3.0));
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_texture_query() {
    parse_str(
        "
        var t: texture_multisampled_2d_array<f32>;
        fn foo() {
            var dim: vec2<u32> = textureDimensions(t);
            dim = textureDimensions(t, 0);
            let layers: u32 = textureNumLayers(t);
            let samples: u32 = textureNumSamples(t);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_postfix() {
    parse_str(
        "fn foo() {
        let x: f32 = vec4<f32>(1.0, 2.0, 3.0, 4.0).xyz.rgbr.aaaa.wz.g;
        let y: f32 = fract(vec2<f32>(0.5, x)).x;
    }",
    )
    .unwrap();
}

#[test]
fn parse_expressions() {
    parse_str("fn foo() {
        let x: f32 = select(0.0, 1.0, true);
        let y: vec2<f32> = select(vec2<f32>(1.0, 1.0), vec2<f32>(x, x), vec2<bool>(x < 0.5, x > 0.5));
        let z: bool = !(0.0 == 1.0);
    }").unwrap();
}

#[test]
fn binary_expression_mixed_scalar_and_vector_operands() {
    for (operand, expect_splat) in [
        ('<', false),
        ('>', false),
        ('&', false),
        ('|', false),
        ('+', true),
        ('-', true),
        ('*', false),
        ('/', true),
        ('%', true),
    ] {
        let module = parse_str(&format!(
            "
            @fragment
            fn main(@location(0) some_vec: vec3<f32>) -> @location(0) vec4<f32> {{
                if (all(1.0 {operand} some_vec)) {{
                    return vec4(0.0);
                }}
                return vec4(1.0);
            }}
            "
        ))
        .unwrap();

        let expressions = &&module.entry_points[0].function.expressions;

        let found_expressions = expressions
            .iter()
            .filter(|&(_, e)| {
                if let crate::Expression::Binary { left, .. } = *e {
                    matches!(
                        (expect_splat, &expressions[left]),
                        (false, &crate::Expression::Literal(crate::Literal::F32(..)))
                            | (true, &crate::Expression::Splat { .. })
                    )
                } else {
                    false
                }
            })
            .count();

        assert_eq!(
            found_expressions,
            1,
            "expected `{operand}` expression {} splat",
            if expect_splat { "with" } else { "without" }
        );
    }

    let module = parse_str(
        "@fragment
        fn main(mat: mat3x3<f32>) {
            let vec = vec3<f32>(1.0, 1.0, 1.0);
            let result = mat / vec;
        }",
    )
    .unwrap();
    let expressions = &&module.entry_points[0].function.expressions;
    let found_splat = expressions.iter().any(|(_, e)| {
        if let crate::Expression::Binary { left, .. } = *e {
            matches!(&expressions[left], &crate::Expression::Splat { .. })
        } else {
            false
        }
    });
    assert!(!found_splat, "'mat / vec' should not be splatted");
}

#[test]
fn parse_pointers() {
    parse_str(
        "fn foo(a: ptr<private, f32>) -> f32 { return *a; }
    fn bar() {
        var x: f32 = 1.0;
        let px = &x;
        let py = foo(px);
    }",
    )
    .unwrap();
}

#[test]
fn parse_struct_instantiation() {
    parse_str(
        "
    struct Foo {
        a: f32,
        b: vec3<f32>,
    }

    @fragment
    fn fs_main() {
        var foo: Foo = Foo(0.0, vec3<f32>(0.0, 1.0, 42.0));
    }
    ",
    )
    .unwrap();
}

#[test]
fn parse_array_length() {
    parse_str(
        "
        struct Foo {
            data: array<u32>
        } // this is used as both input and output for convenience

        @group(0) @binding(0)
        var<storage> foo: Foo;

        @group(0) @binding(1)
        var<storage> bar: array<u32>;

        fn baz() {
            var x: u32 = arrayLength(foo.data);
            var y: u32 = arrayLength(bar);
        }
        ",
    )
    .unwrap();
}

#[test]
fn parse_storage_buffers() {
    parse_str(
        "
        @group(0) @binding(0)
        var<storage> foo: array<u32>;
        ",
    )
    .unwrap();
    parse_str(
        "
        @group(0) @binding(0)
        var<storage,read> foo: array<u32>;
        ",
    )
    .unwrap();
    parse_str(
        "
        @group(0) @binding(0)
        var<storage,write> foo: array<u32>;
        ",
    )
    .unwrap();
    parse_str(
        "
        @group(0) @binding(0)
        var<storage,read_write> foo: array<u32>;
        ",
    )
    .unwrap();
}

#[test]
fn parse_alias() {
    parse_str(
        "
        alias Vec4 = vec4<f32>;
        ",
    )
    .unwrap();
}

#[test]
fn parse_texture_load_store_expecting_four_args() {
    for (func, texture) in [
        (
            "textureStore",
            "texture_storage_2d_array<rg11b10float, write>",
        ),
        ("textureLoad", "texture_2d_array<i32>"),
    ] {
        let error = parse_str(&format!(
            "
            @group(0) @binding(0) var tex_los_res: {texture};
            @compute
            @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                var color = vec4(1, 1, 1, 1);
                {func}(tex_los_res, id, color);
            }}
            "
        ))
        .unwrap_err();
        assert_eq!(
            error.message(),
            "wrong number of arguments: expected 4, found 3"
        );
    }
}

#[test]
fn parse_repeated_attributes() {
    use crate::{
        front::wgsl::{error::Error, Frontend},
        Span,
    };

    let template_vs = "@vertex fn vs() -> __REPLACE__ vec4<f32> { return vec4<f32>(0.0); }";
    let template_struct = "struct A { __REPLACE__ data: vec3<f32> }";
    let template_resource = "__REPLACE__ var tex_los_res: texture_2d_array<i32>;";
    let template_stage = "__REPLACE__ fn vs() -> vec4<f32> { return vec4<f32>(0.0); }";
    for (attribute, template) in [
        ("align(16)", template_struct),
        ("binding(0)", template_resource),
        ("builtin(position)", template_vs),
        ("compute", template_stage),
        ("fragment", template_stage),
        ("group(0)", template_resource),
        ("interpolate(flat)", template_vs),
        ("invariant", template_vs),
        ("location(0)", template_vs),
        ("size(16)", template_struct),
        ("vertex", template_stage),
        ("early_depth_test(less_equal)", template_resource),
        ("workgroup_size(1)", template_stage),
    ] {
        let shader = template.replace("__REPLACE__", &format!("@{attribute} @{attribute}"));
        let name_length = attribute.rfind('(').unwrap_or(attribute.len()) as u32;
        let span_start = shader.rfind(attribute).unwrap() as u32;
        let span_end = span_start + name_length;
        let expected_span = Span::new(span_start, span_end);

        let result = Frontend::new()
            .inner_to_ast(&shader)
            .and_then(|v| v.inner_to_module(None));
        assert!(matches!(
            result.unwrap_err(),
            Error::RepeatedAttribute(span) if span == expected_span
        ));
    }
}

#[test]
fn parse_missing_workgroup_size() {
    use crate::{
        front::wgsl::{error::Error, Frontend},
        Span,
    };

    let shader = "@compute fn vs() -> vec4<f32> { return vec4<f32>(0.0); }";
    let result = Frontend::new()
        .inner_to_ast(shader)
        .and_then(|v| v.inner_to_module(None));
    assert!(matches!(
        result.unwrap_err(),
        Error::MissingWorkgroupSize(span) if span == Span::new(1, 8)
    ));
}

#[test]
fn parse_base_module_constant() {
    use crate::front::wgsl::Frontend;

    let base_module = parse_str("const some_constant_value: f32 = 1.0;").unwrap();
    let shader = "fn foo() -> vec4<f32> { return vec4<f32>(some_constant_value); }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

/* Add this test once abstract numerics are supported.
Although, depending on the implementation, this may not work anyways.
#[test]
fn parse_base_module_abstract_numerics() {
    use crate::front::wgsl::Frontend;

    let base_module =
        parse_str("const two = 2;")
            .unwrap();
    let shader = "const four: u32 = 2 * two; const signed_two: i32 = two;";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}
*/
#[test]
fn parse_structs_with_base_module_structs() {
    use crate::front::wgsl::Frontend;

    let base_module = parse_str("struct Bar { a: vec4f, b: Foo }; struct Foo { c: f32 }").unwrap();
    let shader = "fn foo(foo: Foo) -> Bar { return Bar(vec4(1.0), foo); }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

#[test]
fn parse_fn_with_base_module() {
    use crate::front::wgsl::Frontend;

    let base_module =
        parse_str("fn cat() -> Foo { return Foo(1.0); }; struct Foo { c: f32 }").unwrap();
    let shader = "fn foo() -> f32 { return cat().c; }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

#[test]
fn parse_fn_conflict_with_base_module() {
    use crate::front::wgsl::{error::Error, Frontend};

    let base_module = parse_str("fn cat() -> f32 { return 1.0; }").unwrap();
    let shader = "fn cat() -> f32 { return 2.0; }";
    let result = Frontend::new()
        .inner_to_ast(shader)
        .and_then(|v| v.inner_to_module(Some(&base_module)));
    assert!(matches!(result, Err(Error::Redefinition { .. })));
}

#[test]
fn parse_base_module_alias() {
    use crate::front::wgsl::Frontend;

    let base_module =
        parse_str("alias number = u32; struct Bar { a: number, b: Foo }; alias Foo = i32;")
            .unwrap();
    let shader = "fn foo(a: u32) -> Bar { return Bar(a, -1i); }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

#[test]
fn parse_base_module_alias_usage() {
    use crate::front::wgsl::Frontend;

    let base_module = parse_str("alias number = u32; struct Bar { a: u32 };").unwrap();
    let shader = "fn foo(a: number) -> Bar { return Bar(a); }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

/* Add this once https://github.com/gfx-rs/wgpu/issues/5786 is fixed
#[test]
fn parse_base_module_alias_predeclared() {
    use crate::front::wgsl::Frontend;

    let base_module =
        parse_str("alias vec4f = u32; struct Bar { a: vec4f, b: Foo }; alias Foo = i32;").unwrap();
    let shader = "fn foo(a: u32) -> Bar { return Bar(a, -1i); }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}
 */

#[test]
fn parse_base_module_function_predefined() {
    use crate::front::wgsl::Frontend;

    // Changing a predefined function should affect the shader
    let base_module = parse_str(
        "fn bar(a: f32) -> f32 { return cos(vec3f(a)).z; } fn cos(a: vec3f) -> vec3f { return a.xyz; }",
    )
    .unwrap();
    let shader = "fn foo(a: f32) -> f32 { return cos(1.0); }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

#[test]
fn parse_base_module_function_predefined_no_leak() {
    use crate::front::wgsl::Frontend;

    // But changing a predefined function in the shader should not affect the base module
    let base_module = parse_str("fn bar(a: f32) -> f32 { return cos(a); }").unwrap();
    let shader =
        "fn foo(a: f32) -> f32 { return cos(vec3f(a)).z + bar(a); } fn cos(a: vec3f) -> vec3f { return a.xyz; }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

#[test]
fn parse_base_module_twice() {
    use crate::front::wgsl::Frontend;

    let base_module_a = parse_str("fn bar(a: f32) -> f32 { return cos(a); }").unwrap();
    let shader_a = "fn foo(a: f32) -> f32 { return bar(a); }";
    let base_module_b = Frontend::new()
        .parse_to_ast(shader_a)
        .unwrap()
        .to_module(Some(&base_module_a))
        .unwrap();
    let shader_b = "fn foobar(a: f32) -> f32 { return bar(a) + foo(a); }";
    Frontend::new()
        .parse_to_ast(shader_b)
        .unwrap()
        .to_module(Some(&base_module_b))
        .unwrap();
}

#[test]
fn parse_base_module_const_conflict() {
    use crate::front::wgsl::{error::Error, Frontend};

    let base_module = parse_str("const foo: f32 = 1.0;").unwrap();
    let shader = "fn foo() -> f32 { return 2.0; }";
    let result = Frontend::new()
        .inner_to_ast(shader)
        .and_then(|v| v.inner_to_module(Some(&base_module)));
    assert!(matches!(result, Err(Error::Redefinition { .. })));
}

#[test]
fn parse_base_module_const_local() {
    use crate::front::wgsl::Frontend;
    // Const in base module with same name as a local variable in the actual module shouldn't cause conflicts

    let base_module = parse_str("const foo: vec3f = vec3f(1.0);").unwrap();
    let shader = "fn bar() -> f32 { let foo: f32 = 2.0; return foo; }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

#[test]
fn parse_base_module_entry_points() {
    use crate::front::wgsl::Frontend;

    let base_module =
        parse_str("@vertex fn vs() -> @builtin(position) vec4f { return vec4<f32>(0.0); }")
            .unwrap();
    let shader = "@fragment fn fs() -> @location(0) vec4f { return vec4<f32>(1.0); }";
    let result = Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
    assert_eq!(result.entry_points.len(), 2);
}

#[test]
fn parse_base_module_pipeline_overridable_constants() {
    use crate::front::wgsl::Frontend;

    let base_module = parse_str("override diffuse_param: f32 = 2.3;").unwrap();
    let shader = "override specular_param: f32 = 0.1; fn foo() -> f32 { return diffuse_param + specular_param; }";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}

#[test]
fn parse_base_module_storage_buffers() {
    use crate::front::wgsl::Frontend;

    let base_module = parse_str(
        "@group(0) @binding(0)
        var<storage,read_write> foo: array<u32>;",
    )
    .unwrap();
    let shader = "@group(0) @binding(1)
    var<storage,read_write> bar: array<u32>;";
    Frontend::new()
        .parse_to_ast(shader)
        .unwrap()
        .to_module(Some(&base_module))
        .unwrap();
}
