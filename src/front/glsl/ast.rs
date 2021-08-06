use super::SourceMetadata;
use crate::{
    BinaryOperator, Binding, Constant, Expression, Function, GlobalVariable, Handle, Interpolation,
    ResourceBinding, Sampling, StorageAccess, StorageClass, Type, UnaryOperator,
};

#[derive(Debug, Clone, Copy)]
pub enum GlobalLookupKind {
    Variable(Handle<GlobalVariable>),
    Constant(Handle<Constant>),
    BlockSelect(Handle<GlobalVariable>, u32),
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalLookup {
    pub kind: GlobalLookupKind,
    pub entry_arg: Option<usize>,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub qualifier: ParameterQualifier,
    /// Wether the parameter should be treated as a depth image instead of a
    /// sampled image
    pub depth: bool,
}

#[derive(Debug)]
pub struct FunctionDeclaration {
    /// Normalized function parameters, modifiers are not applied
    pub parameters: Vec<Handle<Type>>,
    pub parameters_info: Vec<ParameterInfo>,
    pub handle: Handle<Function>,
    /// Wheter this function was already defined or is just a prototype
    pub defined: bool,
    /// Wheter or not this function returns void (nothing)
    pub void: bool,
}

#[derive(Debug)]
pub struct EntryArg {
    pub name: Option<String>,
    pub binding: Binding,
    pub handle: Handle<GlobalVariable>,
    pub storage: StorageQualifier,
}

#[derive(Debug, Clone)]
pub struct VariableReference {
    pub expr: Handle<Expression>,
    pub load: bool,
    pub mutable: bool,
    pub entry_arg: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub meta: SourceMetadata,
}

#[derive(Debug, Clone)]
pub enum HirExprKind {
    Access {
        base: Handle<HirExpr>,
        index: Handle<HirExpr>,
    },
    Select {
        base: Handle<HirExpr>,
        field: String,
    },
    Constant(Handle<Constant>),
    Binary {
        left: Handle<HirExpr>,
        op: BinaryOperator,
        right: Handle<HirExpr>,
    },
    Unary {
        op: UnaryOperator,
        expr: Handle<HirExpr>,
    },
    Variable(VariableReference),
    Call(FunctionCall),
    Conditional {
        condition: Handle<HirExpr>,
        accept: Handle<HirExpr>,
        reject: Handle<HirExpr>,
    },
    Assign {
        tgt: Handle<HirExpr>,
        value: Handle<HirExpr>,
    },
    IncDec {
        increment: bool,
        postfix: bool,
        expr: Handle<HirExpr>,
    },
}

#[derive(Debug)]
pub enum TypeQualifier {
    StorageQualifier(StorageQualifier),
    Interpolation(Interpolation),
    ResourceBinding(ResourceBinding),
    Location(u32),
    WorkGroupSize(usize, u32),
    Sampling(Sampling),
    Layout(StructLayout),
    Precision(Precision),
    EarlyFragmentTests,
    StorageAccess(StorageAccess),
}

#[derive(Debug, Clone)]
pub enum FunctionCallKind {
    TypeConstructor(Handle<Type>),
    Function(String),
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub kind: FunctionCallKind,
    pub args: Vec<Handle<HirExpr>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageQualifier {
    StorageClass(StorageClass),
    Input,
    Output,
    Const,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructLayout {
    Std140,
    Std430,
}

// TODO: Encode precision hints in the IR
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Precision {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum ParameterQualifier {
    In,
    Out,
    InOut,
    Const,
}

impl ParameterQualifier {
    /// Returns true if the argument should be passed as a lhs expression
    pub fn is_lhs(&self) -> bool {
        match *self {
            ParameterQualifier::Out | ParameterQualifier::InOut => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Profile {
    Core,
}
