pub use ast::Profile;
pub use error::{ErrorKind, ParseError};
pub use token::{SourceMetadata, Token};

use crate::{FastHashMap, Handle, Module, ShaderStage, Type};
use ast::{EntryArg, FunctionDeclaration, GlobalLookup};
use parser::ParsingContext;

mod ast;
mod constants;
mod context;
mod error;
mod functions;
mod lex;
mod offset;
mod parser;
#[cfg(test)]
mod parser_tests;
mod token;
mod types;
mod variables;

type Result<T> = std::result::Result<T, ErrorKind>;

pub struct Options {
    pub stage: ShaderStage,
    pub defines: FastHashMap<String, String>,
}

#[derive(Debug)]
pub struct ShaderMetadata {
    pub version: u16,
    pub profile: Profile,
    pub stage: ShaderStage,

    pub workgroup_size: [u32; 3],
    pub early_fragment_tests: bool,

    pub extensions: FastHashMap<String, String>,
}

impl ShaderMetadata {
    fn reset(&mut self, stage: ShaderStage) {
        self.version = 0;
        self.profile = Profile::Core;
        self.stage = stage;
        self.workgroup_size = [if stage == ShaderStage::Compute { 1 } else { 0 }; 3];
        self.early_fragment_tests = false;
        self.extensions.clear();
    }
}

impl Default for ShaderMetadata {
    fn default() -> Self {
        ShaderMetadata {
            version: 0,
            profile: Profile::Core,
            stage: ShaderStage::Vertex,
            workgroup_size: [0; 3],
            early_fragment_tests: false,
            extensions: FastHashMap::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Parser {
    meta: ShaderMetadata,

    lookup_function: FastHashMap<String, Vec<FunctionDeclaration>>,
    lookup_type: FastHashMap<String, Handle<Type>>,

    global_variables: Vec<(String, GlobalLookup)>,

    entry_args: Vec<EntryArg>,

    module: Module,
}

impl Parser {
    fn reset(&mut self, stage: ShaderStage) {
        self.meta.reset(stage);

        self.lookup_function.clear();
        self.lookup_type.clear();
        self.global_variables.clear();
        self.entry_args.clear();

        // This is necessary because if the last parsing errored out, the module
        // wouldn't have been swapped
        self.module = Module::default();
    }

    pub fn parse(
        &mut self,
        options: &Options,
        source: &str,
    ) -> std::result::Result<Module, ParseError> {
        self.reset(options.stage);

        let lexer = lex::Lexer::new(source, &options.defines);
        let mut ctx = ParsingContext::new(lexer);

        ctx.parse(self).map_err(|kind| ParseError { kind })?;

        let mut module = Module::default();
        std::mem::swap(&mut self.module, &mut module);
        Ok(module)
    }

    pub fn metadata(&self) -> &ShaderMetadata {
        &self.meta
    }
}
