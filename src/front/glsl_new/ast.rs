use crate::{Arena, FastHashMap, Function, Handle};

#[derive(Debug)]
pub struct Program {
    pub version: u16,
    pub profile: Profile,
    pub ext_decls: Vec<ExtDecl>,
    pub lookup_function: FastHashMap<String, Handle<Function>>,
    pub functions: Arena<Function>,
}

impl Program {
    pub fn new() -> Program {
        Program {
            version: 0,
            profile: Profile::Core,
            ext_decls: vec![],
            lookup_function: FastHashMap::default(),
            functions: Arena::<Function>::new(),
        }
    }
}

#[derive(Debug)]
pub enum Profile {
    Core,
}

#[derive(Debug)]
pub enum ExtDecl {
    // FunctionDecl,
}
