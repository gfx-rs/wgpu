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
    assert!(parse_str("const a : x32 = 2;").is_err());
    parse_str("var t: texture_2d<f32>;").unwrap();
    parse_str("var t: texture_cube_array<i32>;").unwrap();
    parse_str("var t: texture_multisampled_2d<u32>;").unwrap();
    parse_str("var t: [[access(write)]] texture_storage_1d<rgba8uint>;").unwrap();
    parse_str("var t: [[access(read)]] texture_storage_3d<r32float>;").unwrap();
}

#[test]
fn parse_type_inference() {
    parse_str(
        "
        fn foo() {
            const a = 2u;
            const b: u32 = a;
        }",
    )
    .unwrap();
    assert!(parse_str(
        "
        fn foo() { const c : i32 = 2.0; }",
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
            const x: vec2<f32> = vec2<f32>(1.0, 2.0);
            const y: vec2<u32> = vec2<u32>(x);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_struct() {
    parse_str(
        "
        [[block]] struct Foo { x: i32; };
        struct Bar {
            [[size(16)]] x: vec2<i32>;
            [[align(16)]] y: f32;
            [[size(32), align(8)]] z: vec3<f32>;
        };
        struct Empty {};
        var s: [[access(read_write)]] Foo;
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
            if (true) {
                discard;
            } else {}
            if (0 != 1) {}
            if (false) {
                return;
            } elseif (true) {
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
                if (i == 1) { break; }
                continuing { i = 1; }
            }
            loop {
                if (i == 0) { continue; }
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
                case 2: { pos = 1.0; fallthrough; }
                case 3: {}
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
            const r: vec4<u32> = textureLoad(t, vec3<u32>(0.0, 1.0, 2.0), 1);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        var t: texture_multisampled_2d_array<i32>;
        fn foo() {
            const r: vec4<i32> = textureLoad(t, vec2<i32>(10, 20), 2, 3);
        }
    ",
    )
    .unwrap();
    parse_str(
        "
        var t: [[access(read)]] texture_storage_1d_array<r32float>;
        fn foo() {
            const r: vec4<f32> = textureLoad(t, 10, 2);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_texture_store() {
    parse_str(
        "
        var t: [[access(write)]] texture_storage_2d<rgba8unorm>;
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
            var dim: vec2<i32> = textureDimensions(t);
            dim = textureDimensions(t, 0);
            const layers: i32 = textureNumLayers(t);
            const samples: i32 = textureNumSamples(t);
        }
    ",
    )
    .unwrap();
}

#[test]
fn parse_postfix() {
    parse_str(
        "fn foo() {
        const x: f32 = vec4<f32>(1.0, 2.0, 3.0, 4.0).xyz.rgbr.aaaa.wz.g;
        const y: f32 = fract(vec2<f32>(0.5, x)).x;
    }",
    )
    .unwrap();
}

#[test]
fn parse_expressions() {
    parse_str("fn foo() {
        const x: f32 = select(0.0, 1.0, true);
        const y: vec2<f32> = select(vec2<f32>(1.0, 1.0), vec2<f32>(x, x), vec2<bool>(x < 0.5, x > 0.5));
        const z: bool = !(0.0 == 1.0);
    }").unwrap();
}

#[test]
fn parse_pointers() {
    parse_str(
        "fn foo() {
        var x: f32 = 1.0;
        const px = &x;
        const py = frexp(0.5, px);
    }",
    )
    .unwrap();
}
