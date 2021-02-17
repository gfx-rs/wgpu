mod helpers;
mod instructions;
mod layout;
mod writer;

pub use spirv::Capability;
pub use writer::{Error, Writer};

use spirv::Word;

bitflags::bitflags! {
    pub struct WriterFlags: u32 {
        const DEBUG = 0x1;
    }
}

struct PhysicalLayout {
    magic_number: Word,
    version: Word,
    generator: Word,
    bound: Word,
    instruction_schema: Word,
}

#[derive(Default)]
struct LogicalLayout {
    capabilities: Vec<Word>,
    extensions: Vec<Word>,
    ext_inst_imports: Vec<Word>,
    memory_model: Vec<Word>,
    entry_points: Vec<Word>,
    execution_modes: Vec<Word>,
    debugs: Vec<Word>,
    annotations: Vec<Word>,
    declarations: Vec<Word>,
    function_declarations: Vec<Word>,
    function_definitions: Vec<Word>,
}

struct Instruction {
    op: spirv::Op,
    wc: u32,
    type_id: Option<Word>,
    result_id: Option<Word>,
    operands: Vec<Word>,
}

#[derive(Debug, Clone)]
pub struct Options {
    /// (Major, Minor) target version of the SPIR-V.
    pub lang_version: (u8, u8),
    /// Configuration flags for the writer.
    pub flags: WriterFlags,
    /// Set of SPIR-V capabilities.
    pub capabilities: crate::FastHashSet<Capability>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            lang_version: (1, 0),
            flags: WriterFlags::empty(),
            capabilities: Default::default(),
        }
    }
}

pub fn write_vec(
    module: &crate::Module,
    analysis: &crate::proc::analyzer::Analysis,
    options: &Options,
) -> Result<Vec<u32>, Error> {
    let mut words = Vec::new();
    let mut w = Writer::new(options)?;
    w.write(module, analysis, &mut words)?;
    Ok(words)
}
