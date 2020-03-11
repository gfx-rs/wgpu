pub mod spirv;
pub mod wgsl;

use crate::arena::Arena;

pub const GENERATOR: u32 = 0;

impl crate::Module {
    fn from_header(header: crate::Header) -> Self {
        crate::Module {
            header,
            types: Arena::new(),
            constants: Arena::new(),
            global_variables: Arena::new(),
            functions: Arena::new(),
            entry_points: Vec::new(),
        }
    }

    fn generate_empty() -> Self {
        Self::from_header(crate::Header {
            version: (1, 0, 0),
            generator: GENERATOR,
        })
    }
}
