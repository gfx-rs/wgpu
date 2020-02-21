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

#[repr(u8)]
#[derive(Clone, Debug)]
pub enum VectorSize {
    Bi = 2,
    Tri = 3,
    Quad = 4,
}

#[repr(u8)]
#[derive(Clone, Debug)]
pub enum ScalarKind {
    Sint,
    Uint,
    Float,
}

#[derive(Debug)]
pub struct ArrayDeclaration {
    pub base: Type,
    pub length: u32,
}

#[derive(Debug)]
pub struct StructMember {
    pub name: Option<String>,
    pub ty: Type,
}

#[derive(Debug)]
pub struct StructDeclaration {
    pub name: Option<String>,
    pub members: Vec<StructMember>,
}

#[derive(Clone, Debug)]
pub enum Type {
    Void,
    Scalar { kind: ScalarKind, width: Bytes },
    Vector { size: VectorSize, kind: ScalarKind, width: Bytes },
    Array(Token<ArrayDeclaration>),
    Struct(Token<StructDeclaration>),
}

#[derive(Clone, Debug)]
pub enum Constant {
    Sint(i64),
    Uint(u64),
    Float(f64),
}

#[derive(Debug)]
pub enum Expression {
    Constant(Constant),
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
    pub array_declarations: Storage<ArrayDeclaration>,
    pub struct_declarations: Storage<StructDeclaration>,
    pub functions: Storage<Function>,
    pub entry_points: Vec<EntryPoint>,
}
