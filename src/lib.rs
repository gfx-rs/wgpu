extern crate spirv_headers as spirv;

pub mod back;
pub mod front;
mod storage;


use crate::storage::{Storage, Token};

use std::{
    collections::HashMap,
    hash::BuildHasherDefault,
};

type FastHashMap<K, T> = HashMap<K, T, BuildHasherDefault<fxhash::FxHasher>>;



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
pub enum Expression {
    Arithmetic,
}

pub type Block = Vec<Statement>;
#[derive(Debug)]
pub struct FallThrough;

#[derive(Debug)]
pub enum Statement {
    Expression(Expression),
    Block(Block),
    If {
        condition: Expression, //bool
        accept: Block,
        reject: Block,
    },
    Switch {
        selector: Expression, //int
        cases: FastHashMap<i32, (Block, Option<FallThrough>)>,
        default: Block,
    },
    Return {
        value: Option<Expression>,
    },
    Kill,
}

#[derive(Debug)]
pub struct Function {
    pub name: Option<String>,
    pub parameter_types: Vec<Type>,
    pub return_type: Type,
    pub body: Block,
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
