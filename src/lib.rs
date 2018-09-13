extern crate rspirv;
extern crate spirv_headers;

pub mod msl;

use std::collections::HashMap;


pub struct Transpiler {
}

pub struct Module {
    raw: rspirv::mr::Module,
    entry_points: HashMap<String, EntryPoint>,
}

pub struct EntryPoint {
    pub cleansed_name: String,
    pub exec_model: spirv_headers::ExecutionModel,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoadError {
    Parsing,
}

impl Transpiler {
    pub fn new() -> Self {
        Transpiler {
        }
    }

    pub fn load(&mut self, spv: &[u8]) -> Result<Module, LoadError> {
        let mut loader = rspirv::mr::Loader::new();
        rspirv::binary::Parser::new(spv, &mut loader)
            .parse()
            .map_err(|_| LoadError::Parsing)?;
        let raw = loader.module();

        let entry_points = raw.entry_points
            .iter()
            .map(|ep| {
                let name = match ep.operands[2] {
                    rspirv::mr::Operand::LiteralString(ref name) => name.to_string(),
                    ref other => panic!("Unexpected entry point operand {:?}", other),
                };
                let ep = EntryPoint {
                    cleansed_name: name.clone(), //TODO
                    exec_model: match ep.operands[0] {
                        rspirv::mr::Operand::ExecutionModel(model) => model,
                        ref other => panic!("Unexpected execution model operand {:?}", other),
                    },
                };
                (name, ep)
            })
            .collect();

        Ok(Module {
            raw,
            entry_points,
        })
    }
}

impl Module {
    pub fn entry_points(&self) -> &HashMap<String, EntryPoint> {
        &self.entry_points
    }
}
