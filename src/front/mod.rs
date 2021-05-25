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

    pub fn get_handle(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        types: &mut Arena<crate::Type>,
    ) -> Handle<crate::Type> {
        let mut dummy = TypeResolution::Value(crate::TypeInner::Sampler { comparison: false });
        let res = &mut self.resolutions[expr_handle.index()];

        std::mem::swap(&mut dummy, res);

        let v = match dummy {
            TypeResolution::Handle(h) => h,
            TypeResolution::Value(inner) => {
                let h = types.fetch_or_append(crate::Type { name: None, inner });
                dummy = TypeResolution::Handle(h);
                h
            }
        };

        std::mem::swap(&mut dummy, res);

        v
    }

    pub fn grow(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        ctx: &ResolveContext,
    ) -> Result<(), ResolveError> {
        if self.resolutions.len() <= expr_handle.index() {
            for (eh, expr) in expressions.iter().skip(self.resolutions.len()) {
                let resolution = ctx.resolve(expr, |h| &self.resolutions[h.index()])?;
                log::debug!("Resolving {:?} = {:?} : {:?}", eh, expr, resolution);
                self.resolutions.push(resolution);
            }
        }
        Ok(())
    }
}
