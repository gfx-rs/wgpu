#![allow(clippy::new_without_default)]

mod arena;
pub mod back;
pub mod front;
pub mod proc;

use crate::arena::{Arena, Handle};

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
};


pub type FastHashMap<K, T> = HashMap<K, T, BuildHasherDefault<fxhash::FxHasher>>;
pub type FastHashSet<K> = HashSet<K, BuildHasherDefault<fxhash::FxHasher>>;

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
    pub ty: Handle<Type>,
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
    Pointer { base: Handle<Type>, class: spirv::StorageClass },
    Array { base: Handle<Type>, size: ArraySize },
    Struct { members: Vec<StructMember> },
    Image { base: Handle<Type>, dim: spirv::Dim, flags: ImageFlags },
    Sampler,
}

#[derive(Debug, PartialEq)]
pub struct Constant {
    pub name: Option<String>,
    pub specialization: Option<spirv::Word>,
    pub inner: ConstantInner,
    pub ty: Handle<Type>,
}

#[derive(Debug, PartialEq)]
pub enum ConstantInner {
    Sint(i64),
    Uint(u64),
    Float(f64),
    Bool(bool),
    Composite(Vec<Handle<Constant>>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Binding {
    BuiltIn(spirv::BuiltIn),
    Location(spirv::Word),
    Descriptor { set: spirv::Word, binding: spirv::Word },
}

bitflags::bitflags! {
    pub struct GlobalUse: u8 {
        const LOAD = 0x1;
        const STORE = 0x2;
    }
}

#[derive(Clone, Debug)]
pub struct GlobalVariable {
    pub name: Option<String>,
    pub class: spirv::StorageClass,
    pub binding: Option<Binding>,
    pub ty: Handle<Type>,
}

#[derive(Clone, Debug)]
pub struct LocalVariable {
    pub name: Option<String>,
    pub ty: Handle<Type>,
    pub init: Option<Handle<Expression>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    ExclusiveOr,
    InclusiveOr,
    LogicalAnd,
    LogicalOr,
    ShiftLeftLogical,
    ShiftRightLogical,
    ShiftRightArithmetic,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntrinsicFunction {
    Any,
    All,
    IsNan,
    IsInf,
    IsFinite,
    IsNormal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DerivativeAxis {
    X,
    Y,
    Width,
}

#[derive(Clone, Debug)]
pub enum Expression {
    Access {
        base: Handle<Expression>,
        index: Handle<Expression>, //int
    },
    AccessIndex {
        base: Handle<Expression>,
        index: u32,
    },
    Constant(Handle<Constant>),
    Compose {
        ty: Handle<Type>,
        components: Vec<Handle<Expression>>,
    },
    FunctionParameter(u32),
    GlobalVariable(Handle<GlobalVariable>),
    LocalVariable(Handle<LocalVariable>),
    Load {
        pointer: Handle<Expression>,
    },
    ImageSample {
        image: Handle<Expression>,
        sampler: Handle<Expression>,
        coordinate: Handle<Expression>,
    },
    Unary {
        op: UnaryOperator,
        expr: Handle<Expression>,
    },
    Binary {
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    },
    Intrinsic {
        fun: IntrinsicFunction,
        argument: Handle<Expression>,
    },
    DotProduct(Handle<Expression>, Handle<Expression>),
    CrossProduct(Handle<Expression>, Handle<Expression>),
    Derivative {
        axis: DerivativeAxis,
        //modifier,
        expr: Handle<Expression>,
    },
    Call {
        name: String,
        arguments: Vec<Handle<Expression>>,
    },
}

pub type Block = Vec<Statement>;
#[derive(Debug)]
pub struct FallThrough;

#[derive(Debug)]
pub enum Statement {
    Empty,
    Block(Block),
    If {
        condition: Handle<Expression>, //bool
        accept: Block,
        reject: Block,
    },
    Switch {
        selector: Handle<Expression>, //int
        cases: FastHashMap<i32, (Block, Option<FallThrough>)>,
        default: Block,
    },
    Loop {
        body: Block,
        continuing: Block,
    },
    //TODO: move terminator variations into a separate enum?
    Break,
    Continue,
    Return {
        value: Option<Handle<Expression>>,
    },
    Kill,
    Store {
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    },
}

#[derive(Debug)]
pub struct Function {
    pub name: Option<String>,
    pub control: spirv::FunctionControl,
    pub parameter_types: Vec<Handle<Type>>,
    pub return_type: Option<Handle<Type>>,
    pub global_usage: Vec<GlobalUse>,
    pub local_variables: Arena<LocalVariable>,
    pub expressions: Arena<Expression>,
    pub body: Block,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub exec_model: spirv::ExecutionModel,
    pub name: String,
    pub function: Handle<Function>,
}

#[derive(Debug)]
pub struct Module {
    pub header: Header,
    pub types: Arena<Type>,
    pub constants: Arena<Constant>,
    pub global_variables: Arena<GlobalVariable>,
    pub functions: Arena<Function>,
    pub entry_points: Vec<EntryPoint>,
}
