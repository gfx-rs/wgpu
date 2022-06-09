/*!
Frontend parsers that consume binary and text shaders and load them into [`Module`](super::Module)s.
*/

mod interpolator;

#[cfg(feature = "glsl-in")]
pub mod glsl;
#[cfg(feature = "spv-in")]
pub mod spv;
#[cfg(feature = "wgsl-in")]
pub mod wgsl;

use crate::{
    arena::{Arena, Handle, UniqueArena},
    proc::{ResolveContext, ResolveError, TypeResolution},
};
use std::ops;

/// Helper class to emit expressions
#[allow(dead_code)]
#[derive(Default, Debug)]
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
    fn finish(
        &mut self,
        arena: &Arena<crate::Expression>,
    ) -> Option<(crate::Statement, crate::span::Span)> {
        let start_len = self.start_len.take().unwrap();
        if start_len != arena.len() {
            #[allow(unused_mut)]
            let mut span = crate::span::Span::default();
            let range = arena.range_from(start_len);
            #[cfg(feature = "span")]
            for handle in range.clone() {
                span.subsume(arena.get_span(handle))
            }
            Some((crate::Statement::Emit(range), span))
        } else {
            None
        }
    }
}

#[allow(dead_code)]
impl super::ConstantInner {
    const fn boolean(value: bool) -> Self {
        Self::Scalar {
            width: super::BOOL_WIDTH,
            value: super::ScalarValue::Bool(value),
        }
    }
}

/// Helper processor that derives the types of all expressions.
#[derive(Debug, Default)]
pub struct Typifier {
    resolutions: Vec<TypeResolution>,
}

impl Typifier {
    pub const fn new() -> Self {
        Typifier {
            resolutions: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.resolutions.clear()
    }

    pub fn get<'a>(
        &'a self,
        expr_handle: Handle<crate::Expression>,
        types: &'a UniqueArena<crate::Type>,
    ) -> &'a crate::TypeInner {
        self.resolutions[expr_handle.index()].inner_with(types)
    }

    pub fn grow(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        ctx: &ResolveContext,
    ) -> Result<(), ResolveError> {
        if self.resolutions.len() <= expr_handle.index() {
            for (eh, expr) in expressions.iter().skip(self.resolutions.len()) {
                //Note: the closure can't `Err` by construction
                let resolution = ctx.resolve(expr, |h| Ok(&self.resolutions[h.index()]))?;
                log::debug!("Resolving {:?} = {:?} : {:?}", eh, expr, resolution);
                self.resolutions.push(resolution);
            }
        }
        Ok(())
    }

    /// Invalidates the cached type resolution for `expr_handle` forcing a recomputation
    ///
    /// If the type of the expression hasn't yet been calculated a
    /// [`grow`](Self::grow) is performed instead
    pub fn invalidate(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        ctx: &ResolveContext,
    ) -> Result<(), ResolveError> {
        if self.resolutions.len() <= expr_handle.index() {
            self.grow(expr_handle, expressions, ctx)
        } else {
            let expr = &expressions[expr_handle];
            //Note: the closure can't `Err` by construction
            let resolution = ctx.resolve(expr, |h| Ok(&self.resolutions[h.index()]))?;
            self.resolutions[expr_handle.index()] = resolution;
            Ok(())
        }
    }
}

impl ops::Index<Handle<crate::Expression>> for Typifier {
    type Output = TypeResolution;
    fn index(&self, handle: Handle<crate::Expression>) -> &Self::Output {
        &self.resolutions[handle.index()]
    }
}
