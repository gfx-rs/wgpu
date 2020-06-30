use crate::{Arena, Constant, EntryPoint, Function, GlobalVariable, Header, Module, ShaderStage, Type};

mod lex;
use lex::Lexer;
mod error;
use error::{ErrorKind, ParseError};
mod parser;
mod token;

pub fn parse_str(source: &str, entry: String, stage: ShaderStage) -> Result<Module, ParseError> {
    log::debug!("------ GLSL-pomelo ------");

    let module = Module {
        header: Header {
            version: (1, 0, 0),
            generator: 0,
        },
        types: Arena::<Type>::new(),
        constants: Arena::<Constant>::new(),
        global_variables: Arena::<GlobalVariable>::new(),
        functions: Arena::<Function>::new(),
        entry_points: vec![],
    };

    let lex = Lexer::new(source);
    let mut parser = parser::Parser::new(module);

    for token in lex {
        log::debug!("token: {:#?}", token);
        parser.parse(token).map_err(|_| ErrorKind::InvalidInput)?;
    }
    let (_, mut parsed_module) = parser.end_of_input().map_err(|_| ErrorKind::InvalidInput)?;

    // find entry point
    let entry_func = parsed_module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_ref().filter(|n| **n == entry).is_some());
    if let Some((h, _)) = entry_func {
        parsed_module.entry_points.push(EntryPoint {
            stage,
            name: entry,
            function: h,
        });
    }

    Ok(parsed_module)
}
