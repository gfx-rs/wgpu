use crate::{Arena, FastHashMap, Function, Handle, Type};

#[derive(Debug)]
pub struct Program {
    pub version: u16,
    pub profile: Profile,
    pub ext_decls: Vec<ExtDecl>,
    pub lookup_function: FastHashMap<String, Handle<Function>>,
    pub functions: Arena<Function>,
    pub lookup_type: FastHashMap<String, Handle<Type>>,
    pub types: Arena<Type>,
}

impl Program {
    pub fn new() -> Program {
        Program {
            version: 0,
            profile: Profile::Core,
            ext_decls: vec![],
            lookup_function: FastHashMap::default(),
            functions: Arena::<Function>::new(),
            lookup_type: FastHashMap::default(),
            types: Arena::<Type>::new(),
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
