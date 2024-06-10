/*!
Frontend for [WGSL][wgsl] (WebGPU Shading Language).

[wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html
*/

mod error;
mod index;
mod lower;
mod parse;
#[cfg(test)]
mod tests;
mod to_wgsl;

use crate::front::wgsl::error::Error;
use crate::front::wgsl::parse::Parser;
use thiserror::Error;

pub use crate::front::wgsl::error::ParseError;
use crate::front::wgsl::lower::Lowerer;
use crate::Scalar;

use self::parse::ast::TranslationUnit;

pub struct Frontend {
    parser: Parser,
}

impl Frontend {
    pub const fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }

    pub fn parse(&mut self, source: &str) -> Result<crate::Module, ParseError> {
        self.parse_to_ast(source)?.to_module(None)
    }

    /// Two-step module conversion, can be used to compile with a "base module".
    pub fn parse_to_ast<'a>(&mut self, source: &'a str) -> Result<ParsedWgsl<'a>, ParseError> {
        self.inner_to_ast(source)
            .map_err(|x| x.as_parse_error(source))
    }

    fn inner_to_ast<'a>(&mut self, source: &'a str) -> Result<ParsedWgsl<'a>, Error<'a>> {
        let translation_unit = self.parser.parse(source)?;
        let index = index::Index::generate(&translation_unit)?;
        Ok(ParsedWgsl {
            source,
            translation_unit,
            index,
        })
    }
}

pub struct ParsedWgsl<'a> {
    source: &'a str,
    translation_unit: TranslationUnit<'a>,
    index: index::Index<'a>,
}
impl<'a> ParsedWgsl<'a> {
    pub fn to_module(
        &self,
        base_module: Option<&crate::Module>,
    ) -> Result<crate::Module, ParseError> {
        self.inner_to_module(base_module)
            .map_err(|x| x.as_parse_error(self.source))
    }
    fn inner_to_module(
        &self,
        base_module: Option<&'a crate::Module>,
    ) -> Result<crate::Module, Error<'a>> {
        Lowerer::new(&self.index).lower(&self.translation_unit, base_module)
    }
}
/// <div class="warning">
// NOTE: Keep this in sync with `wgpu::Device::create_shader_module`!
// NOTE: Keep this in sync with `wgpu_core::Global::device_create_shader_module`!
///
/// This function may consume a lot of stack space. Compiler-enforced limits for parsing recursion
/// exist; if shader compilation runs into them, it will return an error gracefully. However, on
/// some build profiles and platforms, the default stack size for a thread may be exceeded before
/// this limit is reached during parsing. Callers should ensure that there is enough stack space
/// for this, particularly if calls to this method are exposed to user input.
///
/// </div>
pub fn parse_str(source: &str) -> Result<crate::Module, ParseError> {
    Frontend::new().parse(source)
}
