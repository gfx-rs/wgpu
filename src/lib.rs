pub mod back;
pub mod front;
mod storage;

use crate::storage::{Storage, Token};

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
};


type FastHashMap<K, T> = HashMap<K, T, BuildHasherDefault<fxhash::FxHasher>>;
type FastHashSet<K> = HashSet<K, BuildHasherDefault<fxhash::FxHasher>>;

#[derive(Clone, Debug)]
pub struct Header {
    pub version: (u8, u8, u8),
    pub generator: u32,
}

pub type Bytes = u8;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VectorSize {
    Bi = 2,
    Tri = 3,
    Quad = 4,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarKind {
    Sint,
    Uint,
    Float,
    Bool,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArraySize {
    Static(spirv::Word),
    Dynamic,
}

#[derive(Debug, PartialEq)]
pub struct StructMember {
    pub name: Option<String>,
    pub binding: Option<Binding>,
    pub ty: Token<Type>,
}

bitflags::bitflags! {
    pub struct ImageFlags: u32 {
        const ARRAYED = 0x1;
        const MULTISAMPLED = 0x2;
        const SAMPLED = 0x4;
        const CAN_LOAD = 0x10;
        const CAN_STORE = 0x20;
    }
}

#[derive(Debug)]
pub struct Type {
    pub name: Option<String>,
    pub inner: TypeInner,
}

#[derive(Debug, PartialEq)]
pub enum TypeInner {
    Scalar { kind: ScalarKind, width: Bytes },
    Vector { size: VectorSize, kind: ScalarKind, width: Bytes },
    Matrix { columns: VectorSize, rows: VectorSize, kind: ScalarKind, width: Bytes },
    Pointer { base: Token<Type>, class: spirv::StorageClass },
    Array { base: Token<Type>, size: ArraySize },
    Struct { members: Vec<StructMember> },
    Image { base: Token<Type>, dim: spirv::Dim, flags: ImageFlags },
    Sampler,
}

#[derive(Debug)]
pub struct Constant {
    pub name: Option<String>,
    pub specialization: Option<spirv::Word>,
    pub inner: ConstantInner,
}

#[derive(Debug)]
pub enum ConstantInner {
    Sint(i64),
    Uint(u64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Binding {
    BuiltIn(spirv::BuiltIn),
    Location(spirv::Word),
    Descriptor { set: spirv::Word, binding: spirv::Word },
}

#[derive(Clone, Debug)]
pub struct GlobalVariable {
    pub name: Option<String>,
    pub class: spirv::StorageClass,
    pub binding: Option<Binding>,
    pub ty: Token<Type>,
}

#[derive(Clone, Copy, Debug)]
pub enum BinaryOperator {
    Multiply,
    Add,
    Equals,
    And,
    ExclusiveOr,
    InclusiveOr,
    LogicalAnd,
    LogicalOr,
}

#[derive(Debug)]
pub enum Expression {
    Access {
        base: Token<Expression>,
        index: Token<Expression>, //int
    },
    AccessIndex {
        base: Token<Expression>,
        index: u32,
    },
    Constant(Token<Constant>),
    Compose {
        ty: Token<Type>,
        components: Vec<Token<Expression>>,
    },
    FunctionParameter(u32),
    GlobalVariable(Token<GlobalVariable>),
    Load {
        pointer: Token<Expression>,
    },
    Mul(Token<Expression>, Token<Expression>),
    ImageSample {
        image: Token<Expression>,
        sampler: Token<Expression>,
        coordinate: Token<Expression>,
    },
    Binary {
        op: BinaryOperator,
        left: Token<Expression>,
        right: Token<Expression>,
    },
}

pub type Block = Vec<Statement>;
#[derive(Debug)]
pub struct FallThrough;

#[derive(Debug)]
pub enum Statement {
    Block(Block),
    VariableDeclaration {
        name: String,
        ty: Token<Type>,
        value: Option<Token<Expression>>,
    },
    If {
        condition: Token<Expression>, //bool
        accept: Block,
        reject: Block,
    },
    Switch {
        selector: Token<Expression>, //int
        cases: FastHashMap<i32, (Block, Option<FallThrough>)>,
        default: Block,
    },
    Return {
        value: Option<Token<Expression>>,
    },
    Kill,
    Store {
        pointer: Token<Expression>,
        value: Token<Expression>,
    },
}

#[derive(Debug)]
pub struct Function {
    pub name: Option<String>,
    pub control: spirv::FunctionControl,
    pub parameter_types: Vec<Token<Type>>,
    pub return_type: Option<Token<Type>>,
    pub expressions: Storage<Expression>,
    pub body: Block,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub exec_model: spirv::ExecutionModel,
    pub name: String,
    pub inputs: Vec<Token<GlobalVariable>>,
    pub outputs: Vec<Token<GlobalVariable>>,
    pub function: Token<Function>,
}

#[derive(Debug)]
pub struct Module {
    pub header: Header,
    pub types: Storage<Type>,
    pub constants: Storage<Constant>,
    pub global_variables: Storage<GlobalVariable>,
    pub functions: Storage<Function>,
    pub entry_points: Vec<EntryPoint>,
}
