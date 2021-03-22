//! Parsers which load shaders into memory.

#[cfg(feature = "glsl-in")]
pub mod glsl;
#[cfg(feature = "spv-in")]
pub mod spv;
#[cfg(feature = "wgsl-in")]
pub mod wgsl;

use crate::{
    arena::{Arena, Handle},
    proc::{ResolveContext, ResolveError, TypeResolution},
};

/// Helper class to emit expressions
#[allow(dead_code)]
#[derive(Default)]
struct Emitter {
    start_len: Option<usize>,
}

#[allow(dead_code)]
impl Emitter {
    fn start(&mut self, arena: &Arena<crate::Expression>) {
        if self.start_len.is_some() {
            unreachable!("Emitting has already started!");
        }
        self.start_len = Some(arena.len());
    }
    #[must_use]
    fn finish(&mut self, arena: &Arena<crate::Expression>) -> Option<crate::Statement> {
        let start_len = self.start_len.take().unwrap();
        if start_len != arena.len() {
            Some(crate::Statement::Emit(arena.range_from(start_len)))
        } else {
            None
        }
    }
}

#[allow(dead_code)]
impl super::ConstantInner {
    fn boolean(value: bool) -> Self {
        Self::Scalar {
            width: super::BOOL_WIDTH,
            value: super::ScalarValue::Bool(value),
        }
    }
}

/// Helper processor that derives the types of all expressions.
#[derive(Debug)]
pub struct Typifier {
    resolutions: Vec<TypeResolution>,
}

impl Typifier {
    pub fn new() -> Self {
        Typifier {
            resolutions: Vec::new(),
        }
    }

    pub fn get<'a>(
        &'a self,
        expr_handle: Handle<crate::Expression>,
        types: &'a Arena<crate::Type>,
    ) -> &'a crate::TypeInner {
        self.resolutions[expr_handle.index()].inner_with(types)
    }

    pub fn grow(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        types: &mut Arena<crate::Type>,
        ctx: &ResolveContext,
    ) -> Result<(), ResolveError> {
        if self.resolutions.len() <= expr_handle.index() {
            for (eh, expr) in expressions.iter().skip(self.resolutions.len()) {
                let resolution = ctx.resolve(expr, types, |h| &self.resolutions[h.index()])?;
                log::debug!("Resolving {:?} = {:?} : {:?}", eh, expr, resolution);
                self.resolutions.push(resolution);
            }
        }
        Ok(())
    }
}
