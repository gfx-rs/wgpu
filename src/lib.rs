extern crate spirv_headers as spirv;

pub mod msl;
mod parse;
mod storage;

pub use parse::{Parser, ParseError, parse_u8_slice};

use crate::storage::{Storage, Token};

use std::collections::HashMap;

use smallvec::SmallVec;



#[derive(Debug)]
pub struct Header {
    pub version: (u8, u8, u8),
    pub generator: u32,
}

pub type Bytes = u8;

#[derive(Debug)]
pub struct StructDeclaration {

}

#[derive(Debug)]
pub enum Type {
    Void,
    Int { width: Bytes },
    Uint { width: Bytes },
    Float { width: Bytes },
    Struct(Token<StructDeclaration>),
}

#[derive(Debug)]
pub struct Jump {
    pub target: Token<Block>,
    pub arguments: SmallVec<[Token<Operation>; 1]>,
}

#[derive(Debug)]
pub enum Branch {
    Jump(Jump),
    JumpIf {
        condition: Token<Operation>, //bool
        accept: Jump,
        reject: Jump,
    },
    Switch {
        selector: Token<Operation>, //int
        cases: HashMap<i32, Jump>,
        default: Jump,
    },
    Return {
        value: Option<Token<Operation>>,
    },
}

#[derive(Debug)]
pub enum Operation {
    Arithmetic,
}

#[derive(Debug)]
pub enum Terminator {
    Branch(Branch),
    Kill,
    Unreachable,
}

#[derive(Debug)]
pub struct Block {
    pub label: Option<String>,
    pub argument_types: Vec<Type>,
    pub operations: Storage<Operation>,
    pub terminator: Terminator,
}

#[derive(Debug)]
pub struct Function {
    pub name: Option<String>,
    pub parameter_types: Vec<Type>,
    pub return_type: Type,
    pub blocks: Storage<Block>,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub exec_model: spirv::ExecutionModel,
    pub name: String,
    pub function: Token<Function>,
}

#[derive(Debug)]
pub struct Module {
    pub header: Header,
    pub struct_declarations: Storage<StructDeclaration>,
    pub functions: Storage<Function>,
    pub entry_points: Vec<EntryPoint>,
}
