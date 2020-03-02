pub mod spirv;
pub mod wgsl;

use crate::storage::Storage;

pub const GENERATOR: u32 = 0;

impl crate::Module {
    fn from_header(header: crate::Header) -> Self {
        crate::Module {
            header,
            types: Storage::new(),
            constants: Storage::new(),
            global_variables: Storage::new(),
            functions: Storage::new(),
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
