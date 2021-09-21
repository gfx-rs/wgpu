use std::fmt;

use super::{builtins::MacroCall, context::ExprPos, Span};
use crate::{
    BinaryOperator, Binding, Constant, Expression, Function, GlobalVariable, Handle, Interpolation,
    Sampling, StorageAccess, StorageClass, Type, UnaryOperator,
};

#[derive(Debug, Clone, Copy)]
pub enum GlobalLookupKind {
    Variable(Handle<GlobalVariable>),
    Constant(Handle<Constant>, Handle<Type>),
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

/// How the function is implemented
#[derive(Clone, Copy)]
pub enum FunctionKind {
    /// The function is user defined
    Call(Handle<Function>),
    /// The function is a buitin
    Macro(MacroCall),
}

impl fmt::Debug for FunctionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Call(_) => write!(f, "Call"),
            Self::Macro(_) => write!(f, "Macro"),
        }
    }
}

#[derive(Debug)]
pub struct Overload {
    /// Normalized function parameters, modifiers are not applied
    pub parameters: Vec<Handle<Type>>,
    pub parameters_info: Vec<ParameterInfo>,
    /// How the function is implemented
    pub kind: FunctionKind,
    /// Wheter this function was already defined or is just a prototype
    pub defined: bool,
    /// Wheter or not this function returns void (nothing)
    pub void: bool,
}

#[derive(Debug, Default)]
pub struct FunctionDeclaration {
    pub overloads: Vec<Overload>,
    /// Wether or not this function has the name of a builtin
    pub builtin: bool,
    /// In case [`builtin`](Self::builtin) is true, this field indicates wether
    /// this function already has double overloads added or not, otherwise is unused
    pub double: bool,
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
    pub constant: Option<(Handle<Constant>, Handle<Type>)>,
    pub entry_arg: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub meta: Span,
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
    /// A prefix/postfix operator like `++`
    PrePostfix {
        /// The operation to be performed
        op: BinaryOperator,
        /// Whether this is a postfix or a prefix
        postfix: bool,
        /// The target expression
        expr: Handle<HirExpr>,
    },
}

#[derive(Debug)]
pub enum TypeQualifier {
    StorageQualifier(StorageQualifier),
    Interpolation(Interpolation),
    Set(u32),
    Binding(u32),
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
/// A precision hint used in glsl declarations
///
/// Precision hints can be used to either speed up shader execution or control
/// the precision of arithmetic operations.
///
/// To use a precision hint simply add it before the type in the declaration.
/// ```glsl
/// mediump float a;
/// ```
///
/// The default when no precision is declared is `highp` which means that all
/// operations operate with the type defined width.
///
/// For `mediump` and `lowp` operations follow the spir-v
/// [`RelaxedPrecision`][RelaxedPrecision] decoration semantics.
///
/// [RelaxedPrecision]: https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#_a_id_relaxedprecisionsection_a_relaxed_precision
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Precision {
    /// `lowp` precision
    Low,
    /// `mediump` precision
    Medium,
    /// `highp` precision
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

    /// Converts from a parameter qualifier into a [`ExprPos`](ExprPos)
    pub fn as_pos(&self) -> ExprPos {
        match *self {
            ParameterQualifier::Out | ParameterQualifier::InOut => ExprPos::Lhs,
            _ => ExprPos::Rhs,
        }
    }
}

/// The glsl profile used by a shader
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Profile {
    /// The `core` profile, default when no profile is specified.
    Core,
}
