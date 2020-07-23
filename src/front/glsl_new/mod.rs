use crate::{EntryPoint, Module, ShaderStage};

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

pub fn parse_str(source: &str, entry: String, stage: ShaderStage) -> Result<Module, ParseError> {
    log::debug!("------ GLSL-pomelo ------");

    let mut program = Program::new(stage);
    let lex = Lexer::new(source);
    let mut parser = parser::Parser::new(&mut program);

    for token in lex {
        parser.parse(token)?;
    }
    parser.end_of_input()?;

    let mut module = Module::generate_empty();
    module.functions = program.functions;
    module.types = program.types;
    module.constants = program.constants;
    module.global_variables = program.global_variables;

    // find entry point
    if let Some(entry_handle) = program.lookup_function.get(&entry) {
        module.entry_points.push(EntryPoint {
            stage,
            name: entry,
            function: *entry_handle,
        });
    }

    Ok(module)
}
