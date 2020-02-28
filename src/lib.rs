#[macro_use]
extern crate pest_derive;
extern crate spirv_headers as spirv;

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
}

#[derive(Debug)]
pub struct PointerDeclaration {
    pub name: Option<String>,
    pub base: Type,
    pub class: spirv::StorageClass,
}

#[derive(Debug)]
pub struct ArrayDeclaration {
    pub name: Option<String>,
    pub base: Type,
    pub length: u32,
}

#[derive(Debug)]
pub struct StructMember {
    pub name: Option<String>,
    pub binding: Option<Binding>,
    pub ty: Type,
}

#[derive(Debug)]
pub struct StructDeclaration {
    pub name: Option<String>,
    pub members: Vec<StructMember>,
}

bitflags::bitflags! {
    pub struct ImageFlags: u32 {
        const ARRAYED = 0x1;
        const MULTISAMPLED = 0x2;
        const READABLE = 0x4;
        const WRITABLE = 0x8;
    }
}

#[derive(Debug)]
pub struct ImageDeclaration {
    pub name: Option<String>,
    pub binding: Option<Binding>,
    pub ty: Type,
    pub dim: spirv::Dim,
    pub flags: ImageFlags,
}

#[derive(Debug)]
pub struct SamplerDeclaration {
    pub name: Option<String>,
    pub binding: Option<Binding>,
}

#[derive(Clone, Debug)]
pub enum Type {
    Void,
    Scalar { kind: ScalarKind, width: Bytes },
    Vector { size: VectorSize, kind: ScalarKind, width: Bytes },
    Matrix { columns: VectorSize, rows: VectorSize, kind: ScalarKind, width: Bytes },
    Pointer(Token<PointerDeclaration>),
    Array(Token<ArrayDeclaration>),
    Struct(Token<StructDeclaration>),
    Image(Token<ImageDeclaration>),
    Sampler(Token<SamplerDeclaration>),
}

#[derive(Clone, Debug)]
pub enum Constant {
    Sint(i64),
    Uint(u64),
    Float(f64),
}

#[derive(Clone, Debug)]
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
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub enum Expression {
    Access {
        base: Token<Expression>,
        index: Token<Expression>, //int
    },
    AccessIndex {
        base: Token<Expression>,
        index: u32,
    },
    Constant(Constant),
    Compose {
        ty: Type,
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
}

pub type Block = Vec<Statement>;
#[derive(Debug)]
pub struct FallThrough;

#[derive(Debug)]
pub enum Statement {
    Block(Block),
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
    pub parameter_types: Vec<Type>,
    pub return_type: Type,
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
pub struct ComplexTypes {
    pub pointers: Storage<PointerDeclaration>,
    pub arrays: Storage<ArrayDeclaration>,
    pub structs: Storage<StructDeclaration>,
    pub images: Storage<ImageDeclaration>,
    pub samplers: Storage<SamplerDeclaration>,
}

#[derive(Debug)]
pub struct Module {
    pub header: Header,
    pub complex_types: ComplexTypes,
    pub global_variables: Storage<GlobalVariable>,
    pub functions: Storage<Function>,
    pub entry_points: Vec<EntryPoint>,
}
