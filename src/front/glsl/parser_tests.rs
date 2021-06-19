use pp_rs::token::PreprocessorError;

use super::lex::Lexer;
use super::parser;
use super::{ast::Profile, error::ErrorKind};
use super::{ast::Program, SourceMetadata};
use crate::front::glsl::error::ExpectedToken;
use crate::{
    front::glsl::{token::TokenValue, Token},
    ShaderStage,
};

fn parse_program<'a>(
    source: &str,
    entry_points: &'a crate::FastHashMap<String, ShaderStage>,
) -> Result<Program<'a>, ErrorKind> {
    let mut program = Program::new(entry_points);
    let defines = crate::FastHashMap::default();
    let lex = Lexer::new(source, &defines);
    let mut parser = parser::Parser::new(&mut program, lex);

    parser.parse()?;
    Ok(program)
}

#[test]
fn version() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Vertex);
    // invalid versions
    assert_eq!(
        parse_program("#version 99000", &entry_points)
            .err()
            .unwrap(),
        ErrorKind::InvalidVersion(SourceMetadata { start: 9, end: 14 }, 99000),
    );

    assert_eq!(
        parse_program("#version 449", &entry_points).err().unwrap(),
        ErrorKind::InvalidVersion(SourceMetadata { start: 9, end: 12 }, 449)
    );

    assert_eq!(
        parse_program("#version 450 smart", &entry_points)
            .err()
            .unwrap(),
        ErrorKind::InvalidProfile(SourceMetadata { start: 13, end: 18 }, "smart".into())
    );

    assert_eq!(
        parse_program("#version 450\nvoid f(){} #version 450", &entry_points)
            .err()
            .unwrap(),
        ErrorKind::InvalidToken(
            Token {
                value: TokenValue::Unknown(PreprocessorError::UnexpectedHash),
                meta: SourceMetadata { start: 24, end: 25 }
            },
            vec![ExpectedToken::Eof]
        )
    );

    // valid versions
    let program = parse_program("  #  version 450\nvoid main() {}", &entry_points).unwrap();
    assert_eq!((program.version, program.profile), (450, Profile::Core));

    let program = parse_program("#version 450\nvoid main() {}", &entry_points).unwrap();
    assert_eq!((program.version, program.profile), (450, Profile::Core));

    let program = parse_program("#version 450 core\nvoid main() {}", &entry_points).unwrap();
    assert_eq!((program.version, program.profile), (450, Profile::Core));
}

#[test]
fn control_flow() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Vertex);

    let _program = parse_program(
        r#"
        #  version 450
        void main() {
            if (true) {
                return 1;
            } else {
                return 2;
            }
        }
        "#,
        &entry_points,
    )
    .unwrap();

    let _program = parse_program(
        r#"
        #  version 450
        void main() {
            if (true) {
                return 1;
            }
        }
        "#,
        &entry_points,
    )
    .unwrap();

    let _program = parse_program(
        r#"
        #  version 450
        void main() {
            int x;
            int y = 3;
            switch (5) {
                case 2:
                    x = 2;
                case 5:
                    x = 5;
                    y = 2;
                    break;
                default:
                    x = 0;
            }
        }
        "#,
        &entry_points,
    )
    .unwrap();
    let _program = parse_program(
        r#"
        #  version 450
        void main() {
            int x = 0;
            while(x < 5) {
                x = x + 1;
            }
            do {
                x = x - 1;
            } while(x >= 4)
        }
        "#,
        &entry_points,
    )
    .unwrap();

    let _program = parse_program(
        r#"
        #  version 450
        void main() {
            int x = 0;
            for(int i = 0; i < 10;) {
                x = x + 2;
            }
            for(;;);
            return x;
        }
        "#,
        &entry_points,
    )
    .unwrap();
}

#[test]
fn declarations() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Fragment);

    let _program = parse_program(
        r#"
        #version 450
        layout(location = 0) in vec2 v_uv;
        layout(location = 0) out vec4 o_color;
        layout(set = 1, binding = 1) uniform texture2D tex;
        layout(set = 1, binding = 2) uniform sampler tex_sampler;

        layout(early_fragment_tests) in;
        "#,
        &entry_points,
    )
    .unwrap();

    let _program = parse_program(
        r#"
        #version 450
        layout(std140, set = 2, binding = 0)
        uniform u_locals {
            vec3 model_offs;
            float load_time;
            ivec4 atlas_offs;
        };
        "#,
        &entry_points,
    )
    .unwrap();

    let _program = parse_program(
        r#"
        #version 450
        layout(push_constant, set = 2, binding = 0)
        uniform u_locals {
            vec3 model_offs;
            float load_time;
            ivec4 atlas_offs;
        };
        "#,
        &entry_points,
    )
    .unwrap();

    let _program = parse_program(
        r#"
        #version 450
        layout(std430, set = 2, binding = 0)
        uniform u_locals {
            vec3 model_offs;
            float load_time;
            ivec4 atlas_offs;
        };
        "#,
        &entry_points,
    )
    .unwrap();

    let _program = parse_program(
        r#"
        #version 450
        layout(std140, set = 2, binding = 0)
        uniform u_locals {
            vec3 model_offs;
            float load_time;
        } block_var;

        void main() {
            load_time * model_offs;
            block_var.load_time * block_var.model_offs;
        }
        "#,
        &entry_points,
    )
    .unwrap();

    let _program = parse_program(
        r#"
        #version 450
        float vector = vec4(1.0 / 17.0,  9.0 / 17.0,  3.0 / 17.0, 11.0 / 17.0);
        "#,
        &entry_points,
    )
    .unwrap();
}

#[test]
fn textures() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Fragment);

    let _program = parse_program(
        r#"
        #version 450
        layout(location = 0) in vec2 v_uv;
        layout(location = 0) out vec4 o_color;
        layout(set = 1, binding = 1) uniform texture2D tex;
        layout(set = 1, binding = 2) uniform sampler tex_sampler;
        void main() {
            o_color = texture(sampler2D(tex, tex_sampler), v_uv);
        }
        "#,
        &entry_points,
    )
    .unwrap();
}

#[test]
fn functions() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Vertex);

    parse_program(
        r#"
        #  version 450
        void test1(float);
        void test1(float) {}

        void main() {}
        "#,
        &entry_points,
    )
    .unwrap();

    parse_program(
        r#"
        #  version 450
        void test2(float a) {}
        void test3(float a, float b) {}
        void test4(float, float) {}

        void main() {}
        "#,
        &entry_points,
    )
    .unwrap();

    parse_program(
        r#"
        #  version 450
        float test(float a) { return a; }

        void main() {}
        "#,
        &entry_points,
    )
    .unwrap();

    parse_program(
        r#"
        #  version 450
        float test(vec4 p) {
            return p.x;
        }
        "#,
        &entry_points,
    )
    .unwrap();

    // Function overloading
    parse_program(
        r#"
        #  version 450
        float test(vec2 p);
        float test(vec3 p);
        float test(vec4 p);

        float test(vec2 p) {
            return p.x;
        }

        float test(vec3 p) {
            return p.x;
        }

        float test(vec4 p) {
            return p.x;
        }
        "#,
        &entry_points,
    )
    .unwrap();

    assert_eq!(
        parse_program(
            r#"
                #  version 450
                int test(vec4 p) {
                    return p.x;
                }

                float test(vec4 p) {
                    return p.x;
                }
                "#,
            &entry_points
        )
        .err()
        .unwrap(),
        ErrorKind::SemanticError(
            SourceMetadata {
                start: 134,
                end: 152
            },
            "Function already defined".into()
        )
    );

    println!();

    let _program = parse_program(
        r#"
        #  version 450
        float callee(uint q) {
            return float(q);
        }

        float caller() {
            callee(1u);
        }
        "#,
        &entry_points,
    )
    .unwrap();

    // Nested function call
    let _program = parse_program(
        r#"
            #  version 450
            layout(set = 0, binding = 1) uniform texture2D t_noise;
            layout(set = 0, binding = 2) uniform sampler s_noise;

            void main() {
                textureLod(sampler2D(t_noise, s_noise), vec2(1.0), 0);
            }
        "#,
        &entry_points,
    )
    .unwrap();

    parse_program(
        r#"
        #  version 450
        void fun(vec2 in_parameter, out float out_parameter) {
            ivec2 _ = ivec2(in_parameter);
        }

        void main() {
            float a;
            fun(vec2(1.0), a);
        }
        "#,
        &entry_points,
    )
    .unwrap();
}

#[test]
fn constants() {
    use crate::{Constant, ConstantInner, ScalarValue};

    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Vertex);

    let program = parse_program(
        r#"
        #  version 450
        const float a = 1.0;
        float global = a;
        const float b = a;
        "#,
        &entry_points,
    )
    .unwrap();

    let mut constants = program.module.constants.iter();

    assert_eq!(
        constants.next().unwrap().1,
        &Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Float(1.0)
            }
        }
    );

    assert!(constants.next().is_none());
}

#[test]
fn implicit_conversions() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Vertex);

    parse_program(
        r#"
        #  version 450
        void main() {
            mat4 a = mat4(1);
            float b = 1u;
            float c = 1 + 2.0;
        }
        "#,
        &entry_points,
    )
    .unwrap();
}

#[test]
fn structs() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Fragment);

    parse_program(
        r#"
        #  version 450
        Test {
            vec4 pos;
          } xx;
        "#,
        &entry_points,
    )
    .unwrap_err();

    parse_program(
        r#"
        #  version 450
        struct Test {
            vec4 pos;
        };
        "#,
        &entry_points,
    )
    .unwrap();

    parse_program(
        r#"
        #  version 450
        const int NUM_VECS = 42;
        struct Test {
            vec4 vecs[NUM_VECS];
        };
        "#,
        &entry_points,
    )
    .unwrap();

    parse_program(
        r#"
        #  version 450
        struct Hello {
            vec4 test;
        } test() {
            return Hello( vec4(1.0) );
        }
        "#,
        &entry_points,
    )
    .unwrap();

    parse_program(
        r#"
        #  version 450
        struct Test {};
        "#,
        &entry_points,
    )
    .unwrap_err();

    parse_program(
        r#"
        #  version 450
        inout struct Test {
            vec4 x;
        };
        "#,
        &entry_points,
    )
    .unwrap_err();
}

#[test]
fn swizzles() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Fragment);

    parse_program(
        r#"
        #  version 450
        void main() {
            vec4 v = vec4(1);
            v.xyz = vec3(2);
            v.x = 5.0;
            v.xyz.zxy.yx.xy = vec2(5.0, 1.0);
        }
        "#,
        &entry_points,
    )
    .unwrap();

    parse_program(
        r#"
        #  version 450
        void main() {
            vec4 v = vec4(1);
            v.xx = vec2(5.0);
        }
        "#,
        &entry_points,
    )
    .unwrap_err();

    parse_program(
        r#"
        #  version 450
        void main() {
            vec3 v = vec3(1);
            v.w = 2.0;
        }
        "#,
        &entry_points,
    )
    .unwrap_err();
}
