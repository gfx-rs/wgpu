use super::ast::Program;
use super::error::ErrorKind;
use super::lex::Lexer;
use super::parser;
use crate::ShaderStage;

fn parse_program<'a>(
    source: &str,
    entry_points: &'a crate::FastHashMap<String, ShaderStage>,
) -> Result<Program<'a>, ErrorKind> {
    let mut program = Program::new(entry_points);
    let defines = crate::FastHashMap::default();
    let lex = Lexer::new(source, &defines);
    let mut parser = parser::Parser::new(&mut program);

    for token in lex {
        parser.parse(token)?;
    }
    parser.end_of_input()?;
    Ok(program)
}

#[test]
fn version() {
    let mut entry_points = crate::FastHashMap::default();
    entry_points.insert("".to_string(), ShaderStage::Vertex);
    // invalid versions
    assert_eq!(
        format!(
            "{:?}",
            parse_program("#version 99000", &entry_points)
                .err()
                .unwrap()
        ),
        "InvalidVersion(TokenMetadata { line: 1, chars: 9..10 }, 99000)" //TODO: location
    );

    assert_eq!(
        format!(
            "{:?}",
            parse_program("#version 449", &entry_points).err().unwrap()
        ),
        "InvalidVersion(TokenMetadata { line: 1, chars: 9..10 }, 449)" //TODO: location
    );

    assert_eq!(
        format!(
            "{:?}",
            parse_program("#version 450 smart", &entry_points)
                .err()
                .unwrap()
        ),
        "InvalidProfile(TokenMetadata { line: 1, chars: 13..14 }, \"smart\")" //TODO: location
    );

    assert_eq!(
        format!(
            "{:?}",
            parse_program("#version 450\nvoid f(){} #version 450", &entry_points)
                .err()
                .unwrap()
        ),
        "InvalidToken(Unknown((TokenMetadata { line: 2, chars: 11..12 }, UnexpectedHash)))"
    );

    // valid versions
    let program = parse_program("  #  version 450\nvoid main() {}", &entry_points).unwrap();
    assert_eq!(
        format!("{:?}", (program.version, program.profile)),
        "(450, Core)"
    );

    let program = parse_program("#version 450\nvoid main() {}", &entry_points).unwrap();
    assert_eq!(
        format!("{:?}", (program.version, program.profile)),
        "(450, Core)"
    );

    let program = parse_program("#version 450 core\nvoid main() {}", &entry_points).unwrap();
    assert_eq!(
        format!("{:?}", (program.version, program.profile)),
        "(450, Core)"
    );
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
            return x;
        }
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

    // TODO: Add support for function prototypes
    // parse_program(
    //     r#"
    //     #  version 450
    //     void test1(float);
    //     void test1(float) {}

    //     void main() {}
    //     "#,
    //     ShaderStage::Vertex,
    // )
    // .unwrap();

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
        const flat float b = a;
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

    assert_eq!(
        constants.next().unwrap().1,
        &Constant {
            name: Some(String::from("a")),
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Float(1.0)
            }
        }
    );

    assert_eq!(
        constants.next().unwrap().1,
        &Constant {
            name: Some(String::from("b")),
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Float(1.0)
            }
        }
    );

    assert!(constants.next().is_none());
}
