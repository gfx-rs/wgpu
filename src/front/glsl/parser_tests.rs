use super::ast::Program;
use super::error::ErrorKind;
use super::lex::Lexer;
use super::parser;
use crate::ShaderStage;

fn parse_program(source: &str, stage: ShaderStage) -> Result<Program, ErrorKind> {
    let mut program = Program::new(stage, "");
    let lex = Lexer::new(source);
    let mut parser = parser::Parser::new(&mut program);

    for token in lex {
        parser.parse(token)?;
    }
    parser.end_of_input()?;
    Ok(program)
}

#[test]
fn version() {
    // invalid versions
    assert_eq!(
        format!(
            "{:?}",
            parse_program("#version 99000", ShaderStage::Vertex)
                .err()
                .unwrap()
        ),
        "InvalidVersion(TokenMetadata { line: 0, chars: 9..14 }, 99000)"
    );

    assert_eq!(
        format!(
            "{:?}",
            parse_program("#version 449", ShaderStage::Vertex)
                .err()
                .unwrap()
        ),
        "InvalidVersion(TokenMetadata { line: 0, chars: 9..12 }, 449)"
    );

    assert_eq!(
        format!(
            "{:?}",
            parse_program("#version 450 smart", ShaderStage::Vertex)
                .err()
                .unwrap()
        ),
        "InvalidProfile(TokenMetadata { line: 0, chars: 13..18 }, \"smart\")"
    );

    assert_eq!(
        format!(
            "{:?}",
            parse_program("#version 450\nvoid f(){} #version 450", ShaderStage::Vertex)
                .err()
                .unwrap()
        ),
        "InvalidToken(Unknown((TokenMetadata { line: 1, chars: 11..12 }, \"#\")))"
    );

    // valid versions
    let program = parse_program("  #  version 450\nvoid main() {}", ShaderStage::Vertex).unwrap();
    assert_eq!(
        format!("{:?}", (program.version, program.profile)),
        "(450, Core)"
    );

    let program = parse_program("#version 450\nvoid main() {}", ShaderStage::Vertex).unwrap();
    assert_eq!(
        format!("{:?}", (program.version, program.profile)),
        "(450, Core)"
    );

    let program = parse_program("#version 450 core\nvoid main() {}", ShaderStage::Vertex).unwrap();
    assert_eq!(
        format!("{:?}", (program.version, program.profile)),
        "(450, Core)"
    );
}
