use crate::{Module, ShaderStage};

mod lex;
#[cfg(test)]
mod lex_tests;

mod ast;
use ast::Program;

use lex::Lexer;
mod error;
use error::ParseError;
mod parser;
#[cfg(test)]
mod parser_tests;
mod token;
mod types;
mod variables;

pub fn parse_str(source: &str, entry: &str, stage: ShaderStage) -> Result<Module, ParseError> {
    log::debug!("------ GLSL-pomelo ------");

    let mut program = Program::new(stage, entry);
    let lex = Lexer::new(source);
    let mut parser = parser::Parser::new(&mut program);

    for token in lex {
        parser.parse(token)?;
    }
    parser.end_of_input()?;

    Ok(program.module)
}
