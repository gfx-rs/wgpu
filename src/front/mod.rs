//! Parsers which load shaders into memory.

#[cfg(feature = "glsl-in")]
pub mod glsl;
#[cfg(feature = "spv-in")]
pub mod spv;
#[cfg(feature = "wgsl-in")]
pub mod wgsl;

/// Helper class to emit expressions
#[allow(dead_code)]
#[derive(Default)]
struct Emitter {
    start_len: Option<usize>,
}

#[allow(dead_code)]
impl Emitter {
    fn start(&mut self, arena: &crate::Arena<crate::Expression>) {
        if self.start_len.is_some() {
            unreachable!("Emitting has already started!");
        }
        self.start_len = Some(arena.len());
    }
    #[must_use]
    fn finish(&mut self, arena: &crate::Arena<crate::Expression>) -> Option<crate::Statement> {
        let start_len = self.start_len.take().unwrap();
        if start_len != arena.len() {
            Some(crate::Statement::Emit(arena.range_from(start_len)))
        } else {
            None
        }
    }
}
