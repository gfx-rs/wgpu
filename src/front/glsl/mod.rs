pub use error::ErrorKind;
pub use parser::Token;
pub use token::TokenMetadata;

use crate::{FastHashMap, Module, ShaderStage};

mod lex;

mod ast;
use ast::Program;

mod error;
pub use error::ParseError;
mod constants;
mod functions;
mod parser;
#[cfg(test)]
mod parser_tests;
mod token;
mod types;
mod variables;

pub struct Options {
    pub entry_points: FastHashMap<String, ShaderStage>,
    pub defines: FastHashMap<String, String>,
}

pub fn parse_str(source: &str, options: &Options) -> Result<Module, ParseError> {
    let mut program = Program::new(&options.entry_points);

    let lex = lex::Lexer::new(source, &options.defines);
    let mut parser = parser::Parser::new(&mut program);
    for token in lex {
        parser.parse(token)?;
    }
    parser.end_of_input()?;

    Ok(program.module)
}
