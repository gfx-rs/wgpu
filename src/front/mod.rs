//! Parsers which load shaders into memory.

#[cfg(feature = "glsl-in")]
pub mod glsl;
#[cfg(feature = "spv-in")]
pub mod spv;
#[cfg(feature = "wgsl-in")]
pub mod wgsl;

use crate::arena::Arena;

pub const GENERATOR: u32 = 0;

impl crate::Module {
    pub fn from_header(header: crate::Header) -> Self {
        crate::Module {
            header,
            types: Arena::new(),
            constants: Arena::new(),
            global_variables: Arena::new(),
            functions: Arena::new(),
            entry_points: crate::FastHashMap::default(),
        }
    }

    pub fn generate_empty() -> Self {
        Self::from_header(crate::Header {
            version: (1, 0, 0),
            generator: GENERATOR,
        })
    }
}
