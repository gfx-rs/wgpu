//! Front end for consuming [WebGPU Shading Language][wgsl].
//!
//! [wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html

mod conv;
mod lexer;

use crate::{
    arena::{Arena, Handle},
    proc::{ResolveContext, ResolveError, Typifier},
    FastHashMap,
};

use self::lexer::Lexer;
use thiserror::Error;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Separator(char),
    DoubleColon,
    Paren(char),
    DoubleParen(char),
    Number(&'a str),
    String(&'a str),
    Word(&'a str),
    Operation(char),
    LogicalOperation(char),
    ShiftOperation(char),
    Arrow,
    Unknown(char),
    UnterminatedString,
    End,
}

#[derive(Clone, Debug, Error)]
pub enum Error<'a> {
    #[error("unexpected token: {0:?}")]
    Unexpected(Token<'a>),
    #[error("unable to parse `{0}` as integer: {1}")]
    BadInteger(&'a str, std::num::ParseIntError),
    #[error("unable to parse `{1}` as float: {1}")]
    BadFloat(&'a str, std::num::ParseFloatError),
    #[error("bad field accessor `{0}`")]
    BadAccessor(&'a str),
    #[error(transparent)]
    InvalidResolve(ResolveError),
    #[error("unknown import: `{0}`")]
    UnknownImport(&'a str),
    #[error("unknown storage class: `{0}`")]
    UnknownStorageClass(&'a str),
    #[error("unknown decoration: `{0}`")]
    UnknownDecoration(&'a str),
    #[error("unknown scalar kind: `{0}`")]
    UnknownScalarKind(&'a str),
    #[error("unknown builtin: `{0}`")]
    UnknownBuiltin(&'a str),
    #[error("unknown shader stage: `{0}`")]
    UnknownShaderStage(&'a str),
    #[error("unknown identifier: `{0}`")]
    UnknownIdent(&'a str),
    #[error("unknown scalar type: `{0}`")]
    UnknownScalarType(&'a str),
    #[error("unknown type: `{0}`")]
    UnknownType(&'a str),
    #[error("unknown function: `{0}`")]
    UnknownFunction(&'a str),
    #[error("unknown storage format: `{0}`")]
    UnknownStorageFormat(&'a str),
    #[error("missing offset for structure member `{0}`")]
    MissingMemberOffset(&'a str),
    #[error("array stride must not be 0")]
    ZeroStride,
    #[error("not a composite type: {0:?}")]
    NotCompositeType(Handle<crate::Type>),
    #[error("function redefinition: `{0}`")]
    FunctionRedefinition(&'a str),
    #[error("other error")]
    Other,
}

trait StringValueLookup<'a> {
    type Value;
    fn lookup(&self, key: &'a str) -> Result<Self::Value, Error<'a>>;
}
impl<'a> StringValueLookup<'a> for FastHashMap<&'a str, Handle<crate::Expression>> {
    type Value = Handle<crate::Expression>;
    fn lookup(&self, key: &'a str) -> Result<Self::Value, Error<'a>> {
        self.get(key).cloned().ok_or(Error::UnknownIdent(key))
    }
}

struct StatementContext<'input, 'temp, 'out> {
    lookup_ident: &'temp mut FastHashMap<&'input str, Handle<crate::Expression>>,
    typifier: &'temp mut Typifier,
    variables: &'out mut Arena<crate::LocalVariable>,
    expressions: &'out mut Arena<crate::Expression>,
    types: &'out mut Arena<crate::Type>,
    constants: &'out mut Arena<crate::Constant>,
    global_vars: &'out Arena<crate::GlobalVariable>,
    arguments: &'out [crate::FunctionArgument],
}

impl<'a> StatementContext<'a, '_, '_> {
    fn reborrow(&mut self) -> StatementContext<'a, '_, '_> {
        StatementContext {
            lookup_ident: self.lookup_ident,
            typifier: self.typifier,
            variables: self.variables,
            expressions: self.expressions,
            types: self.types,
            constants: self.constants,
            global_vars: self.global_vars,
            arguments: self.arguments,
        }
    }

    fn as_expression(&mut self) -> ExpressionContext<'a, '_, '_> {
        ExpressionContext {
            lookup_ident: self.lookup_ident,
            typifier: self.typifier,
            expressions: self.expressions,
            types: self.types,
            constants: self.constants,
            global_vars: self.global_vars,
            local_vars: self.variables,
            arguments: self.arguments,
        }
    }
}

struct ExpressionContext<'input, 'temp, 'out> {
    lookup_ident: &'temp FastHashMap<&'input str, Handle<crate::Expression>>,
    typifier: &'temp mut Typifier,
    expressions: &'out mut Arena<crate::Expression>,
    types: &'out mut Arena<crate::Type>,
    constants: &'out mut Arena<crate::Constant>,
    global_vars: &'out Arena<crate::GlobalVariable>,
    local_vars: &'out Arena<crate::LocalVariable>,
    arguments: &'out [crate::FunctionArgument],
}

impl<'a> ExpressionContext<'a, '_, '_> {
    fn reborrow(&mut self) -> ExpressionContext<'a, '_, '_> {
        ExpressionContext {
            lookup_ident: self.lookup_ident,
            typifier: self.typifier,
            expressions: self.expressions,
            types: self.types,
            constants: self.constants,
            global_vars: self.global_vars,
            local_vars: self.local_vars,
            arguments: self.arguments,
        }
    }

    fn resolve_type(
        &mut self,
        handle: Handle<crate::Expression>,
    ) -> Result<&crate::TypeInner, Error<'a>> {
        let functions = Arena::new(); //TODO
        let resolve_ctx = ResolveContext {
            constants: self.constants,
            global_vars: self.global_vars,
            local_vars: self.local_vars,
            functions: &functions,
            arguments: self.arguments,
        };
        match self
            .typifier
            .grow(handle, self.expressions, self.types, &resolve_ctx)
        {
            Err(e) => Err(Error::InvalidResolve(e)),
            Ok(()) => Ok(self.typifier.get(handle, self.types)),
        }
    }

    fn parse_binary_op(
        &mut self,
        lexer: &mut Lexer<'a>,
        classifier: impl Fn(Token<'a>) -> Option<crate::BinaryOperator>,
        mut parser: impl FnMut(
            &mut Lexer<'a>,
            ExpressionContext<'a, '_, '_>,
        ) -> Result<Handle<crate::Expression>, Error<'a>>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        let mut left = parser(lexer, self.reborrow())?;
        while let Some(op) = classifier(lexer.peek()) {
            let _ = lexer.next();
            let expression = crate::Expression::Binary {
                op,
                left,
                right: parser(lexer, self.reborrow())?,
            };
            left = self.expressions.append(expression);
        }
        Ok(left)
    }
}

enum Composition {
    Single(crate::Expression),
    Multi(crate::VectorSize, Vec<Handle<crate::Expression>>),
}

impl Composition {
    fn make<'a>(
        base: Handle<crate::Expression>,
        base_size: crate::VectorSize,
        name: &'a str,
        expressions: &mut Arena<crate::Expression>,
    ) -> Result<Self, Error<'a>> {
        const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];

        Ok(if name.len() > 1 {
            let mut components = Vec::with_capacity(name.len());
            for ch in name.chars() {
                let expr = crate::Expression::AccessIndex {
                    base,
                    index: MEMBERS[..base_size as usize]
                        .iter()
                        .position(|&m| m == ch)
                        .ok_or(Error::BadAccessor(name))? as u32,
                };
                components.push(expressions.append(expr));
            }

            let size = match name.len() {
                2 => crate::VectorSize::Bi,
                3 => crate::VectorSize::Tri,
                4 => crate::VectorSize::Quad,
                _ => return Err(Error::BadAccessor(name)),
            };
            Composition::Multi(size, components)
        } else {
            let ch = name.chars().next().ok_or(Error::BadAccessor(name))?;
            let index = MEMBERS[..base_size as usize]
                .iter()
                .position(|&m| m == ch)
                .ok_or(Error::BadAccessor(name))? as u32;
            Composition::Single(crate::Expression::AccessIndex { base, index })
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scope {
    Decoration,
    ImportDecl,
    VariableDecl,
    TypeDecl,
    FunctionDecl,
    Block,
    Statement,
    ConstantExpr,
    PrimaryExpr,
    SingularExpr,
    GeneralExpr,
}

struct ParsedVariable<'a> {
    name: &'a str,
    class: Option<crate::StorageClass>,
    ty: Handle<crate::Type>,
    access: crate::StorageAccess,
    init: Option<Handle<crate::Constant>>,
}

#[derive(Clone, Debug, Error)]
#[error("error while parsing WGSL in scopes {scopes:?} at position {pos:?}: {error}")]
pub struct ParseError<'a> {
    pub error: Error<'a>,
    pub scopes: Vec<Scope>,
    pub pos: (usize, usize),
}

pub struct Parser {
    scopes: Vec<Scope>,
    lookup_type: FastHashMap<String, Handle<crate::Type>>,
    function_lookup: FastHashMap<String, Handle<crate::Function>>,
    std_namespace: Option<Vec<String>>,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            scopes: Vec::new(),
            lookup_type: FastHashMap::default(),
            function_lookup: FastHashMap::default(),
            std_namespace: None,
        }
    }

    fn deconstruct_composite_type(
        type_arena: &mut Arena<crate::Type>,
        ty: Handle<crate::Type>,
        index: usize,
    ) -> Result<Handle<crate::Type>, Error<'static>> {
        match type_arena[ty].inner {
            crate::TypeInner::Vector { kind, width, .. } => {
                let inner = crate::TypeInner::Scalar { kind, width };
                Ok(type_arena.fetch_or_append(crate::Type { name: None, inner }))
            }
            crate::TypeInner::Matrix { width, .. } => {
                let inner = crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Float,
                    width,
                };
                Ok(type_arena.fetch_or_append(crate::Type { name: None, inner }))
            }
            crate::TypeInner::Array { base, .. } => Ok(base),
            crate::TypeInner::Struct { ref members } => Ok(members[index].ty),
            _ => Err(Error::NotCompositeType(ty)),
        }
    }

    fn get_constant_inner(
        word: &str,
    ) -> Result<(crate::ConstantInner, crate::ScalarKind), Error<'_>> {
        if word.contains('.') {
            word.parse()
                .map(|f| (crate::ConstantInner::Float(f), crate::ScalarKind::Float))
                .map_err(|err| Error::BadFloat(word, err))
        } else {
            word.parse()
                .map(|i| (crate::ConstantInner::Sint(i), crate::ScalarKind::Sint))
                .map_err(|err| Error::BadInteger(word, err))
        }
    }

    fn parse_function_call<'a>(
        &mut self,
        lexer: &Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<(crate::Expression, Lexer<'a>)>, Error<'a>> {
        let mut lexer = lexer.clone();

        let external_function = if let Some(std_namespaces) = self.std_namespace.as_deref() {
            std_namespaces.iter().all(|namespace| {
                lexer.skip(Token::Word(namespace)) && lexer.skip(Token::DoubleColon)
            })
        } else {
            false
        };

        let origin = if external_function {
            let function = lexer.next_ident()?;
            crate::FunctionOrigin::External(function.to_string())
        } else if let Ok(function) = lexer.next_ident() {
            if let Some(&function) = self.function_lookup.get(function) {
                crate::FunctionOrigin::Local(function)
            } else {
                return Ok(None);
            }
        } else {
            return Ok(None);
        };

        if !lexer.skip(Token::Paren('(')) {
            return Ok(None);
        }

        let mut arguments = Vec::new();
        while !lexer.skip(Token::Paren(')')) {
            if !arguments.is_empty() {
                lexer.expect(Token::Separator(','))?;
            }
            let arg = self.parse_general_expression(&mut lexer, ctx.reborrow())?;
            arguments.push(arg);
        }
        Ok(Some((crate::Expression::Call { origin, arguments }, lexer)))
    }

    fn parse_const_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        self_ty: Handle<crate::Type>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Constant>, Error<'a>> {
        self.scopes.push(Scope::ConstantExpr);
        let inner = match lexer.peek() {
            Token::Word("true") => {
                let _ = lexer.next();
                crate::ConstantInner::Bool(true)
            }
            Token::Word("false") => {
                let _ = lexer.next();
                crate::ConstantInner::Bool(false)
            }
            Token::Number(word) => {
                let _ = lexer.next();
                let (inner, _) = Self::get_constant_inner(word)?;
                inner
            }
            _ => {
                let composite_ty = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                lexer.expect(Token::Paren('('))?;
                let mut components = Vec::new();
                while !lexer.skip(Token::Paren(')')) {
                    if !components.is_empty() {
                        lexer.expect(Token::Separator(','))?;
                    }
                    let ty = Self::deconstruct_composite_type(
                        type_arena,
                        composite_ty,
                        components.len(),
                    )?;
                    let component =
                        self.parse_const_expression(lexer, ty, type_arena, const_arena)?;
                    components.push(component);
                }
                crate::ConstantInner::Composite(components)
            }
        };
        let handle = const_arena.fetch_or_append(crate::Constant {
            name: None,
            specialization: None,
            inner,
            ty: self_ty,
        });
        self.scopes.pop();
        Ok(handle)
    }

    fn parse_primary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::PrimaryExpr);
        let backup = lexer.clone();
        let expression = match lexer.next() {
            Token::Paren('(') => {
                let expr = self.parse_general_expression(lexer, ctx)?;
                lexer.expect(Token::Paren(')'))?;
                self.scopes.pop();
                return Ok(expr);
            }
            Token::Word("true") => {
                let handle = ctx.constants.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner: crate::ConstantInner::Bool(true),
                    ty: ctx.types.fetch_or_append(crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar {
                            kind: crate::ScalarKind::Bool,
                            width: 1,
                        },
                    }),
                });
                crate::Expression::Constant(handle)
            }
            Token::Word("false") => {
                let handle = ctx.constants.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner: crate::ConstantInner::Bool(false),
                    ty: ctx.types.fetch_or_append(crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar {
                            kind: crate::ScalarKind::Bool,
                            width: 1,
                        },
                    }),
                });
                crate::Expression::Constant(handle)
            }
            Token::Number(word) => {
                let (inner, kind) = Self::get_constant_inner(word)?;
                let handle = ctx.constants.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner,
                    ty: ctx.types.fetch_or_append(crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar { kind, width: 4 },
                    }),
                });
                crate::Expression::Constant(handle)
            }
            Token::Word(word) => {
                if let Some(handle) = ctx.lookup_ident.get(word) {
                    self.scopes.pop();
                    return Ok(*handle);
                }
                if let Some((expr, new_lexer)) =
                    self.parse_function_call(&backup, ctx.reborrow())?
                {
                    *lexer = new_lexer;
                    expr
                } else {
                    *lexer = backup;
                    let ty = self.parse_type_decl(lexer, None, ctx.types, ctx.constants)?;
                    lexer.expect(Token::Paren('('))?;
                    let mut components = Vec::new();
                    while !lexer.skip(Token::Paren(')')) {
                        if !components.is_empty() {
                            lexer.expect(Token::Separator(','))?;
                        }
                        let sub_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                        components.push(sub_expr);
                    }
                    crate::Expression::Compose { ty, components }
                }
            }
            other => return Err(Error::Unexpected(other)),
        };
        self.scopes.pop();
        Ok(ctx.expressions.append(expression))
    }

    fn parse_postfix<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
        mut handle: Handle<crate::Expression>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        loop {
            match lexer.peek() {
                Token::Separator('.') => {
                    let _ = lexer.next();
                    let name = lexer.next_ident()?;
                    let expression = match *ctx.resolve_type(handle)? {
                        crate::TypeInner::Struct { ref members } => {
                            let index = members
                                .iter()
                                .position(|m| m.name.as_deref() == Some(name))
                                .ok_or(Error::BadAccessor(name))?
                                as u32;
                            crate::Expression::AccessIndex {
                                base: handle,
                                index,
                            }
                        }
                        crate::TypeInner::Vector { size, kind, width } => {
                            match Composition::make(handle, size, name, ctx.expressions)? {
                                Composition::Multi(size, components) => {
                                    let inner = crate::TypeInner::Vector { size, kind, width };
                                    crate::Expression::Compose {
                                        ty: ctx
                                            .types
                                            .fetch_or_append(crate::Type { name: None, inner }),
                                        components,
                                    }
                                }
                                Composition::Single(expr) => expr,
                            }
                        }
                        crate::TypeInner::Matrix {
                            rows,
                            columns,
                            width,
                        } => match Composition::make(handle, columns, name, ctx.expressions)? {
                            Composition::Multi(columns, components) => {
                                let inner = crate::TypeInner::Matrix {
                                    columns,
                                    rows,
                                    width,
                                };
                                crate::Expression::Compose {
                                    ty: ctx
                                        .types
                                        .fetch_or_append(crate::Type { name: None, inner }),
                                    components,
                                }
                            }
                            Composition::Single(expr) => expr,
                        },
                        _ => return Err(Error::BadAccessor(name)),
                    };
                    handle = ctx.expressions.append(expression);
                }
                Token::Paren('[') => {
                    let _ = lexer.next();
                    let index = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(']'))?;
                    let expr = crate::Expression::Access {
                        base: handle,
                        index,
                    };
                    handle = ctx.expressions.append(expr);
                }
                _ => return Ok(handle),
            }
        }
    }

    fn parse_singular_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::SingularExpr);
        let backup = lexer.clone();
        let expression = match lexer.next() {
            Token::Operation('-') => Some(crate::Expression::Unary {
                op: crate::UnaryOperator::Negate,
                expr: self.parse_singular_expression(lexer, ctx.reborrow())?,
            }),
            Token::Operation('!') => Some(crate::Expression::Unary {
                op: crate::UnaryOperator::Not,
                expr: self.parse_singular_expression(lexer, ctx.reborrow())?,
            }),
            Token::Word(word) => {
                if let Some(fun) = conv::get_intrinsic(word) {
                    lexer.expect(Token::Paren('('))?;
                    let argument = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::Intrinsic { fun, argument })
                } else if let Some(axis) = conv::get_derivative(word) {
                    lexer.expect(Token::Paren('('))?;
                    let expr = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::Derivative { axis, expr })
                } else if let Some((kind, _width)) = conv::get_scalar_type(word) {
                    lexer.expect(Token::Paren('('))?;
                    let expr = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::As {
                        expr,
                        kind,
                        convert: true,
                    })
                } else {
                    match word {
                        "dot" => {
                            lexer.expect(Token::Paren('('))?;
                            let a = self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Separator(','))?;
                            let b = self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Paren(')'))?;
                            Some(crate::Expression::DotProduct(a, b))
                        }
                        "cross" => {
                            lexer.expect(Token::Paren('('))?;
                            let a = self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Separator(','))?;
                            let b = self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Paren(')'))?;
                            Some(crate::Expression::CrossProduct(a, b))
                        }
                        "textureSample" => {
                            lexer.expect(Token::Paren('('))?;
                            let image_name = lexer.next_ident()?;
                            lexer.expect(Token::Separator(','))?;
                            let sampler_name = lexer.next_ident()?;
                            lexer.expect(Token::Separator(','))?;
                            let coordinate =
                                self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Paren(')'))?;
                            Some(crate::Expression::ImageSample {
                                image: ctx.lookup_ident.lookup(image_name)?,
                                sampler: ctx.lookup_ident.lookup(sampler_name)?,
                                coordinate,
                                level: crate::SampleLevel::Auto,
                                depth_ref: None,
                            })
                        }
                        "textureSampleLevel" => {
                            lexer.expect(Token::Paren('('))?;
                            let image_name = lexer.next_ident()?;
                            lexer.expect(Token::Separator(','))?;
                            let sampler_name = lexer.next_ident()?;
                            lexer.expect(Token::Separator(','))?;
                            let coordinate =
                                self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Separator(','))?;
                            let level = self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Paren(')'))?;
                            Some(crate::Expression::ImageSample {
                                image: ctx.lookup_ident.lookup(image_name)?,
                                sampler: ctx.lookup_ident.lookup(sampler_name)?,
                                coordinate,
                                level: crate::SampleLevel::Exact(level),
                                depth_ref: None,
                            })
                        }
                        "textureSampleBias" => {
                            lexer.expect(Token::Paren('('))?;
                            let image_name = lexer.next_ident()?;
                            lexer.expect(Token::Separator(','))?;
                            let sampler_name = lexer.next_ident()?;
                            lexer.expect(Token::Separator(','))?;
                            let coordinate =
                                self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Separator(','))?;
                            let bias = self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Paren(')'))?;
                            Some(crate::Expression::ImageSample {
                                image: ctx.lookup_ident.lookup(image_name)?,
                                sampler: ctx.lookup_ident.lookup(sampler_name)?,
                                coordinate,
                                level: crate::SampleLevel::Bias(bias),
                                depth_ref: None,
                            })
                        }
                        "textureSampleCompare" => {
                            lexer.expect(Token::Paren('('))?;
                            let image_name = lexer.next_ident()?;
                            lexer.expect(Token::Separator(','))?;
                            let sampler_name = lexer.next_ident()?;
                            lexer.expect(Token::Separator(','))?;
                            let coordinate =
                                self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Separator(','))?;
                            let reference = self.parse_primary_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Paren(')'))?;
                            Some(crate::Expression::ImageSample {
                                image: ctx.lookup_ident.lookup(image_name)?,
                                sampler: ctx.lookup_ident.lookup(sampler_name)?,
                                coordinate,
                                level: crate::SampleLevel::Zero,
                                depth_ref: Some(reference),
                            })
                        }
                        "textureLoad" => {
                            lexer.expect(Token::Paren('('))?;
                            let image_name = lexer.next_ident()?;
                            let image = ctx.lookup_ident.lookup(image_name)?;
                            lexer.expect(Token::Separator(','))?;
                            let coordinate =
                                self.parse_primary_expression(lexer, ctx.reborrow())?;
                            let is_storage = match *ctx.resolve_type(image)? {
                                crate::TypeInner::Image {
                                    class: crate::ImageClass::Storage(_),
                                    ..
                                } => true,
                                _ => false,
                            };
                            let index = if is_storage {
                                None
                            } else {
                                lexer.expect(Token::Separator(','))?;
                                let index_name = lexer.next_ident()?;
                                Some(ctx.lookup_ident.lookup(index_name)?)
                            };
                            lexer.expect(Token::Paren(')'))?;
                            Some(crate::Expression::ImageLoad {
                                image,
                                coordinate,
                                index,
                            })
                        }
                        _ => None,
                    }
                }
            }
            _ => None,
        };

        let handle = match expression {
            Some(expr) => ctx.expressions.append(expr),
            None => {
                *lexer = backup;
                let handle = self.parse_primary_expression(lexer, ctx.reborrow())?;
                self.parse_postfix(lexer, ctx, handle)?
            }
        };
        self.scopes.pop();
        Ok(handle)
    }

    fn parse_equality_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        // equality_expression
        context.parse_binary_op(
            lexer,
            |token| match token {
                Token::LogicalOperation('=') => Some(crate::BinaryOperator::Equal),
                Token::LogicalOperation('!') => Some(crate::BinaryOperator::NotEqual),
                _ => None,
            },
            // relational_expression
            |lexer, mut context| {
                context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::Paren('<') => Some(crate::BinaryOperator::Less),
                        Token::Paren('>') => Some(crate::BinaryOperator::Greater),
                        Token::LogicalOperation('<') => Some(crate::BinaryOperator::LessEqual),
                        Token::LogicalOperation('>') => Some(crate::BinaryOperator::GreaterEqual),
                        _ => None,
                    },
                    // shift_expression
                    |lexer, mut context| {
                        context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::ShiftOperation('<') => {
                                    Some(crate::BinaryOperator::ShiftLeft)
                                }
                                Token::ShiftOperation('>') => {
                                    Some(crate::BinaryOperator::ShiftRight)
                                }
                                _ => None,
                            },
                            // additive_expression
                            |lexer, mut context| {
                                context.parse_binary_op(
                                    lexer,
                                    |token| match token {
                                        Token::Operation('+') => Some(crate::BinaryOperator::Add),
                                        Token::Operation('-') => {
                                            Some(crate::BinaryOperator::Subtract)
                                        }
                                        _ => None,
                                    },
                                    // multiplicative_expression
                                    |lexer, mut context| {
                                        context.parse_binary_op(
                                            lexer,
                                            |token| match token {
                                                Token::Operation('*') => {
                                                    Some(crate::BinaryOperator::Multiply)
                                                }
                                                Token::Operation('/') => {
                                                    Some(crate::BinaryOperator::Divide)
                                                }
                                                Token::Operation('%') => {
                                                    Some(crate::BinaryOperator::Modulo)
                                                }
                                                _ => None,
                                            },
                                            |lexer, context| {
                                                self.parse_singular_expression(lexer, context)
                                            },
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )
    }

    fn parse_general_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::GeneralExpr);
        // logical_or_expression
        let handle = context.parse_binary_op(
            lexer,
            |token| match token {
                Token::LogicalOperation('|') => Some(crate::BinaryOperator::LogicalOr),
                _ => None,
            },
            // logical_and_expression
            |lexer, mut context| {
                context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::LogicalOperation('&') => Some(crate::BinaryOperator::LogicalAnd),
                        _ => None,
                    },
                    // inclusive_or_expression
                    |lexer, mut context| {
                        context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::Operation('|') => Some(crate::BinaryOperator::InclusiveOr),
                                _ => None,
                            },
                            // exclusive_or_expression
                            |lexer, mut context| {
                                context.parse_binary_op(
                                    lexer,
                                    |token| match token {
                                        Token::Operation('^') => {
                                            Some(crate::BinaryOperator::ExclusiveOr)
                                        }
                                        _ => None,
                                    },
                                    // and_expression
                                    |lexer, mut context| {
                                        context.parse_binary_op(
                                            lexer,
                                            |token| match token {
                                                Token::Operation('&') => {
                                                    Some(crate::BinaryOperator::And)
                                                }
                                                _ => None,
                                            },
                                            |lexer, context| {
                                                self.parse_equality_expression(lexer, context)
                                            },
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )?;
        self.scopes.pop();
        Ok(handle)
    }

    fn parse_variable_ident_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<(&'a str, Handle<crate::Type>), Error<'a>> {
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
        Ok((name, ty))
    }

    fn parse_variable_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<ParsedVariable<'a>, Error<'a>> {
        self.scopes.push(Scope::VariableDecl);
        let mut class = None;
        if lexer.skip(Token::Paren('<')) {
            let class_str = lexer.next_ident()?;
            class = Some(conv::map_storage_class(class_str)?);
            lexer.expect(Token::Paren('>'))?;
        }
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
        let access = match class {
            Some(crate::StorageClass::Storage) => crate::StorageAccess::all(),
            Some(crate::StorageClass::Handle) => {
                match type_arena[ty].inner {
                    //TODO: RW textures
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Storage(_),
                        ..
                    } => crate::StorageAccess::LOAD,
                    _ => crate::StorageAccess::empty(),
                }
            }
            _ => crate::StorageAccess::empty(),
        };
        let init = if lexer.skip(Token::Operation('=')) {
            let handle = self.parse_const_expression(lexer, ty, type_arena, const_arena)?;
            Some(handle)
        } else {
            None
        };
        lexer.expect(Token::Separator(';'))?;
        self.scopes.pop();
        Ok(ParsedVariable {
            name,
            class,
            ty,
            access,
            init,
        })
    }

    fn parse_struct_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Vec<crate::StructMember>, Error<'a>> {
        let mut members = Vec::new();
        lexer.expect(Token::Paren('{'))?;
        loop {
            let mut offset = !0;
            if lexer.skip(Token::DoubleParen('[')) {
                self.scopes.push(Scope::Decoration);
                let mut ready = true;
                loop {
                    match lexer.next() {
                        Token::DoubleParen(']') => {
                            break;
                        }
                        Token::Separator(',') if !ready => {
                            ready = true;
                        }
                        Token::Word("offset") if ready => {
                            lexer.expect(Token::Paren('('))?;
                            offset = lexer.next_uint_literal()?;
                            lexer.expect(Token::Paren(')'))?;
                            ready = false;
                        }
                        other => return Err(Error::Unexpected(other)),
                    }
                }
                self.scopes.pop();
            }
            let name = match lexer.next() {
                Token::Word(word) => word,
                Token::Paren('}') => return Ok(members),
                other => return Err(Error::Unexpected(other)),
            };
            if offset == !0 {
                return Err(Error::MissingMemberOffset(name));
            }
            lexer.expect(Token::Separator(':'))?;
            let ty = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
            lexer.expect(Token::Separator(';'))?;
            members.push(crate::StructMember {
                name: Some(name.to_owned()),
                origin: crate::MemberOrigin::Offset(offset),
                ty,
            });
        }
    }

    fn parse_type_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        self_name: Option<&'a str>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Type>, Error<'a>> {
        self.scopes.push(Scope::TypeDecl);
        let decoration_lexer = if lexer.skip(Token::DoubleParen('[')) {
            Some(lexer.take_until(Token::DoubleParen(']'))?)
        } else {
            None
        };

        let inner = match lexer.next() {
            Token::Word("f32") => crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Float,
                width: 4,
            },
            Token::Word("i32") => crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Sint,
                width: 4,
            },
            Token::Word("u32") => crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Uint,
                width: 4,
            },
            Token::Word("vec2") => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            Token::Word("vec3") => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            Token::Word("vec4") => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Quad,
                    kind,
                    width,
                }
            }
            Token::Word("mat2x2") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            Token::Word("mat2x3") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            Token::Word("mat2x4") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            Token::Word("mat3x2") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            Token::Word("mat3x3") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            Token::Word("mat3x4") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            Token::Word("mat4x2") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            Token::Word("mat4x3") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            Token::Word("mat4x4") => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            Token::Word("ptr") => {
                lexer.expect(Token::Paren('<'))?;
                let class = conv::map_storage_class(lexer.next_ident()?)?;
                lexer.expect(Token::Separator(','))?;
                let base = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                lexer.expect(Token::Paren('>'))?;
                crate::TypeInner::Pointer { base, class }
            }
            Token::Word("array") => {
                lexer.expect(Token::Paren('<'))?;
                let base = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                let size = match lexer.next() {
                    Token::Separator(',') => {
                        let value = lexer.next_uint_literal()?;
                        lexer.expect(Token::Paren('>'))?;
                        let const_handle = const_arena.fetch_or_append(crate::Constant {
                            name: None,
                            specialization: None,
                            inner: crate::ConstantInner::Uint(value as u64),
                            ty: type_arena.fetch_or_append(crate::Type {
                                name: None,
                                inner: crate::TypeInner::Scalar {
                                    kind: crate::ScalarKind::Uint,
                                    width: 4,
                                },
                            }),
                        });
                        crate::ArraySize::Constant(const_handle)
                    }
                    Token::Paren('>') => crate::ArraySize::Dynamic,
                    other => return Err(Error::Unexpected(other)),
                };

                let mut stride = None;
                if let Some(mut lexer) = decoration_lexer {
                    self.scopes.push(Scope::Decoration);
                    loop {
                        match lexer.next() {
                            Token::Word("stride") => {
                                use std::num::NonZeroU32;
                                stride = Some(
                                    NonZeroU32::new(lexer.next_uint_literal()?)
                                        .ok_or(Error::ZeroStride)?,
                                );
                            }
                            Token::End => break,
                            other => return Err(Error::Unexpected(other)),
                        }
                    }
                    self.scopes.pop();
                }

                crate::TypeInner::Array { base, size, stride }
            }
            Token::Word("struct") => {
                let members = self.parse_struct_body(lexer, type_arena, const_arena)?;
                crate::TypeInner::Struct { members }
            }
            Token::Word("sampler") => crate::TypeInner::Sampler { comparison: false },
            Token::Word("sampler_comparison") => crate::TypeInner::Sampler { comparison: true },
            Token::Word("texture_sampled_1d") => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            Token::Word("texture_sampled_1d_array") => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            Token::Word("texture_sampled_2d") => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            Token::Word("texture_sampled_2d_array") => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            Token::Word("texture_sampled_3d") => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            Token::Word("texture_sampled_cube") => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            Token::Word("texture_sampled_cube_array") => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            Token::Word("texture_multisampled_2d") => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            Token::Word("texture_depth_2d") => crate::TypeInner::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth,
            },
            Token::Word("texture_depth_2d_array") => crate::TypeInner::Image {
                dim: crate::ImageDimension::D2,
                arrayed: true,
                class: crate::ImageClass::Depth,
            },
            Token::Word("texture_depth_cube") => crate::TypeInner::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: false,
                class: crate::ImageClass::Depth,
            },
            Token::Word("texture_depth_cube_array") => crate::TypeInner::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: true,
                class: crate::ImageClass::Depth,
            },
            Token::Word("texture_ro_1d") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_ro_1d_array") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_ro_2d") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_ro_2d_array") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_ro_3d") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_wo_1d") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_wo_1d_array") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_wo_2d") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_wo_2d_array") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word("texture_wo_3d") => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            Token::Word(name) => {
                self.scopes.pop();
                return match self.lookup_type.get(name) {
                    Some(&handle) => Ok(handle),
                    None => Err(Error::UnknownType(name)),
                };
            }
            other => return Err(Error::Unexpected(other)),
        };
        self.scopes.pop();

        let handle = type_arena.fetch_or_append(crate::Type {
            name: self_name.map(|s| s.to_string()),
            inner,
        });
        Ok(handle)
    }

    fn parse_statement<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, '_>,
    ) -> Result<crate::Statement, Error<'a>> {
        let backup = lexer.clone();
        match lexer.next() {
            Token::Separator(';') => Ok(crate::Statement::Block(Vec::new())),
            Token::Word(word) => {
                self.scopes.push(Scope::Statement);
                let statement = match word {
                    "var" => {
                        enum Init {
                            Empty,
                            Constant(Handle<crate::Constant>),
                            Variable(Handle<crate::Expression>),
                        }
                        let (name, ty) = self.parse_variable_ident_decl(
                            lexer,
                            context.types,
                            context.constants,
                        )?;
                        let init = if lexer.skip(Token::Operation('=')) {
                            let value =
                                self.parse_general_expression(lexer, context.as_expression())?;
                            if let crate::Expression::Constant(handle) = context.expressions[value]
                            {
                                Init::Constant(handle)
                            } else {
                                Init::Variable(value)
                            }
                        } else {
                            Init::Empty
                        };
                        lexer.expect(Token::Separator(';'))?;
                        let var_id = context.variables.append(crate::LocalVariable {
                            name: Some(name.to_owned()),
                            ty,
                            init: match init {
                                Init::Constant(value) => Some(value),
                                _ => None,
                            },
                        });
                        let expr_id = context
                            .expressions
                            .append(crate::Expression::LocalVariable(var_id));
                        context.lookup_ident.insert(name, expr_id);
                        match init {
                            Init::Variable(value) => crate::Statement::Store {
                                pointer: expr_id,
                                value,
                            },
                            _ => crate::Statement::Block(Vec::new()),
                        }
                    }
                    "return" => {
                        let value = if lexer.peek() != Token::Separator(';') {
                            Some(self.parse_general_expression(lexer, context.as_expression())?)
                        } else {
                            None
                        };
                        lexer.expect(Token::Separator(';'))?;
                        crate::Statement::Return { value }
                    }
                    "if" => {
                        lexer.expect(Token::Paren('('))?;
                        let condition =
                            self.parse_general_expression(lexer, context.as_expression())?;
                        lexer.expect(Token::Paren(')'))?;
                        let accept = self.parse_block(lexer, context.reborrow())?;
                        let reject = if lexer.skip(Token::Word("else")) {
                            self.parse_block(lexer, context.reborrow())?
                        } else {
                            Vec::new()
                        };
                        crate::Statement::If {
                            condition,
                            accept,
                            reject,
                        }
                    }
                    "loop" => {
                        let mut body = Vec::new();
                        let mut continuing = Vec::new();
                        lexer.expect(Token::Paren('{'))?;
                        loop {
                            if lexer.skip(Token::Word("continuing")) {
                                continuing = self.parse_block(lexer, context.reborrow())?;
                                lexer.expect(Token::Paren('}'))?;
                                break;
                            }
                            if lexer.skip(Token::Paren('}')) {
                                break;
                            }
                            let s = self.parse_statement(lexer, context.reborrow())?;
                            body.push(s);
                        }
                        crate::Statement::Loop { body, continuing }
                    }
                    "break" => crate::Statement::Break,
                    "continue" => crate::Statement::Continue,
                    ident => {
                        // assignment
                        if let Some(&var_expr) = context.lookup_ident.get(ident) {
                            let left =
                                self.parse_postfix(lexer, context.as_expression(), var_expr)?;
                            lexer.expect(Token::Operation('='))?;
                            let value =
                                self.parse_general_expression(lexer, context.as_expression())?;
                            lexer.expect(Token::Separator(';'))?;
                            crate::Statement::Store {
                                pointer: left,
                                value,
                            }
                        } else if let Some((expr, new_lexer)) =
                            self.parse_function_call(&backup, context.as_expression())?
                        {
                            *lexer = new_lexer;
                            context.expressions.append(expr);
                            lexer.expect(Token::Separator(';'))?;
                            crate::Statement::Block(Vec::new())
                        } else {
                            return Err(Error::UnknownIdent(ident));
                        }
                    }
                };
                self.scopes.pop();
                Ok(statement)
            }
            other => Err(Error::Unexpected(other)),
        }
    }

    fn parse_block<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, '_>,
    ) -> Result<Vec<crate::Statement>, Error<'a>> {
        self.scopes.push(Scope::Block);
        lexer.expect(Token::Paren('{'))?;
        let mut statements = Vec::new();
        while !lexer.skip(Token::Paren('}')) {
            let s = self.parse_statement(lexer, context.reborrow())?;
            statements.push(s);
        }
        self.scopes.pop();
        Ok(statements)
    }

    fn parse_function_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        module: &mut crate::Module,
        lookup_global_expression: &FastHashMap<&'a str, crate::Expression>,
    ) -> Result<(crate::Function, &'a str), Error<'a>> {
        self.scopes.push(Scope::FunctionDecl);
        // read function name
        let mut lookup_ident = FastHashMap::default();
        let fun_name = lexer.next_ident()?;
        // populare initial expressions
        let mut expressions = Arena::new();
        for (&name, expression) in lookup_global_expression.iter() {
            let expr_handle = expressions.append(expression.clone());
            lookup_ident.insert(name, expr_handle);
        }
        // read parameter list
        let mut arguments = Vec::new();
        lexer.expect(Token::Paren('('))?;
        while !lexer.skip(Token::Paren(')')) {
            if !arguments.is_empty() {
                lexer.expect(Token::Separator(','))?;
            }
            let (param_name, param_type) =
                self.parse_variable_ident_decl(lexer, &mut module.types, &mut module.constants)?;
            let param_index = arguments.len() as u32;
            let expression_token =
                expressions.append(crate::Expression::FunctionArgument(param_index));
            lookup_ident.insert(param_name, expression_token);
            arguments.push(crate::FunctionArgument {
                name: Some(param_name.to_string()),
                ty: param_type,
            });
        }
        // read return type
        lexer.expect(Token::Arrow)?;
        let return_type = if lexer.skip(Token::Word("void")) {
            None
        } else {
            Some(self.parse_type_decl(lexer, None, &mut module.types, &mut module.constants)?)
        };

        let mut fun = crate::Function {
            name: Some(fun_name.to_string()),
            arguments,
            return_type,
            global_usage: Vec::new(),
            local_variables: Arena::new(),
            expressions,
            body: Vec::new(),
        };

        // read body
        let mut typifier = Typifier::new();
        fun.body = self.parse_block(
            lexer,
            StatementContext {
                lookup_ident: &mut lookup_ident,
                typifier: &mut typifier,
                variables: &mut fun.local_variables,
                expressions: &mut fun.expressions,
                types: &mut module.types,
                constants: &mut module.constants,
                global_vars: &module.global_variables,
                arguments: &fun.arguments,
            },
        )?;
        // done
        fun.fill_global_use(&module.global_variables);
        self.scopes.pop();

        Ok((fun, fun_name))
    }

    fn parse_global_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        module: &mut crate::Module,
        lookup_global_expression: &mut FastHashMap<&'a str, crate::Expression>,
    ) -> Result<bool, Error<'a>> {
        // read decorations
        let mut binding = None;
        // Perspective is the default qualifier.
        let mut interpolation = None;
        let mut stage = None;
        let mut workgroup_size = [0u32; 3];

        if lexer.skip(Token::DoubleParen('[')) {
            let (mut bind_index, mut bind_group) = (None, None);
            self.scopes.push(Scope::Decoration);
            loop {
                match lexer.next_ident()? {
                    "location" => {
                        lexer.expect(Token::Paren('('))?;
                        let loc = lexer.next_uint_literal()?;
                        lexer.expect(Token::Paren(')'))?;
                        binding = Some(crate::Binding::Location(loc));
                    }
                    "builtin" => {
                        lexer.expect(Token::Paren('('))?;
                        let builtin = conv::map_built_in(lexer.next_ident()?)?;
                        lexer.expect(Token::Paren(')'))?;
                        binding = Some(crate::Binding::BuiltIn(builtin));
                    }
                    "binding" => {
                        lexer.expect(Token::Paren('('))?;
                        bind_index = Some(lexer.next_uint_literal()?);
                        lexer.expect(Token::Paren(')'))?;
                    }
                    "group" => {
                        lexer.expect(Token::Paren('('))?;
                        bind_group = Some(lexer.next_uint_literal()?);
                        lexer.expect(Token::Paren(')'))?;
                    }
                    "interpolate" => {
                        lexer.expect(Token::Paren('('))?;
                        interpolation = Some(conv::map_interpolation(lexer.next_ident()?)?);
                        lexer.expect(Token::Paren(')'))?;
                    }
                    "stage" => {
                        lexer.expect(Token::Paren('('))?;
                        stage = Some(conv::map_shader_stage(lexer.next_ident()?)?);
                        lexer.expect(Token::Paren(')'))?;
                    }
                    "workgroup_size" => {
                        lexer.expect(Token::Paren('('))?;
                        for (i, size) in workgroup_size.iter_mut().enumerate() {
                            *size = lexer.next_uint_literal()?;
                            match lexer.next() {
                                Token::Paren(')') => break,
                                Token::Separator(',') if i != 2 => (),
                                other => return Err(Error::Unexpected(other)),
                            }
                        }
                        for size in workgroup_size.iter_mut() {
                            if *size == 0 {
                                *size = 1;
                            }
                        }
                    }
                    word => return Err(Error::UnknownDecoration(word)),
                }
                match lexer.next() {
                    Token::DoubleParen(']') => {
                        break;
                    }
                    Token::Separator(',') => {}
                    other => return Err(Error::Unexpected(other)),
                }
            }
            if let (Some(group), Some(index)) = (bind_group, bind_index) {
                binding = Some(crate::Binding::Resource {
                    group,
                    binding: index,
                });
            }
            self.scopes.pop();
        }
        // read items
        match lexer.next() {
            Token::Separator(';') => {}
            Token::Word("import") => {
                self.scopes.push(Scope::ImportDecl);
                let path = match lexer.next() {
                    Token::String(path) => path,
                    other => return Err(Error::Unexpected(other)),
                };
                lexer.expect(Token::Word("as"))?;
                let mut namespaces = Vec::new();
                loop {
                    namespaces.push(lexer.next_ident()?.to_owned());
                    if lexer.skip(Token::Separator(';')) {
                        break;
                    }
                    lexer.expect(Token::DoubleColon)?;
                }
                match path {
                    "GLSL.std.450" => self.std_namespace = Some(namespaces),
                    _ => return Err(Error::UnknownImport(path)),
                }
                self.scopes.pop();
            }
            Token::Word("type") => {
                let name = lexer.next_ident()?;
                lexer.expect(Token::Operation('='))?;
                let ty = self.parse_type_decl(
                    lexer,
                    Some(name),
                    &mut module.types,
                    &mut module.constants,
                )?;
                self.lookup_type.insert(name.to_owned(), ty);
                lexer.expect(Token::Separator(';'))?;
            }
            Token::Word("const") => {
                let (name, ty) = self.parse_variable_ident_decl(
                    lexer,
                    &mut module.types,
                    &mut module.constants,
                )?;
                lexer.expect(Token::Operation('='))?;
                let const_handle = self.parse_const_expression(
                    lexer,
                    ty,
                    &mut module.types,
                    &mut module.constants,
                )?;
                lexer.expect(Token::Separator(';'))?;
                lookup_global_expression.insert(name, crate::Expression::Constant(const_handle));
            }
            Token::Word("var") => {
                let pvar =
                    self.parse_variable_decl(lexer, &mut module.types, &mut module.constants)?;
                let class = match pvar.class {
                    Some(c) => c,
                    None => match binding {
                        Some(crate::Binding::BuiltIn(builtin)) => match builtin {
                            crate::BuiltIn::GlobalInvocationId => crate::StorageClass::Input,
                            crate::BuiltIn::Position => crate::StorageClass::Output,
                            _ => unimplemented!(),
                        },
                        _ => crate::StorageClass::Handle,
                    },
                };
                let var_handle = module.global_variables.append(crate::GlobalVariable {
                    name: Some(pvar.name.to_owned()),
                    class,
                    binding: binding.take(),
                    ty: pvar.ty,
                    init: pvar.init,
                    interpolation,
                    storage_access: pvar.access,
                });
                lookup_global_expression
                    .insert(pvar.name, crate::Expression::GlobalVariable(var_handle));
            }
            Token::Word("fn") => {
                let (function, name) =
                    self.parse_function_decl(lexer, module, &lookup_global_expression)?;
                let already_declared = match stage {
                    Some(stage) => module
                        .entry_points
                        .insert(
                            (stage, name.to_string()),
                            crate::EntryPoint {
                                early_depth_test: None,
                                workgroup_size,
                                function,
                            },
                        )
                        .is_some(),
                    None => {
                        let fun_handle = module.functions.append(function);
                        self.function_lookup
                            .insert(name.to_string(), fun_handle)
                            .is_some()
                    }
                };
                if already_declared {
                    return Err(Error::FunctionRedefinition(name));
                }
            }
            Token::End => return Ok(false),
            token => return Err(Error::Unexpected(token)),
        }
        match binding {
            None => Ok(true),
            // we had the decoration but no var?
            Some(_) => Err(Error::Other),
        }
    }

    pub fn parse<'a>(&mut self, source: &'a str) -> Result<crate::Module, ParseError<'a>> {
        self.scopes.clear();
        self.lookup_type.clear();
        self.std_namespace = None;

        let mut module = crate::Module::generate_empty();
        let mut lexer = Lexer::new(source);
        let mut lookup_global_expression = FastHashMap::default();
        loop {
            match self.parse_global_decl(&mut lexer, &mut module, &mut lookup_global_expression) {
                Err(error) => {
                    let pos = lexer.offset_from(source);
                    let (mut rows, mut cols) = (0, 1);
                    for line in source[..pos].lines() {
                        rows += 1;
                        cols = line.len();
                    }
                    return Err(ParseError {
                        error,
                        scopes: std::mem::replace(&mut self.scopes, Vec::new()),
                        pos: (rows, cols),
                    });
                }
                Ok(true) => {}
                Ok(false) => {
                    if !self.scopes.is_empty() {
                        return Err(ParseError {
                            error: Error::Other,
                            scopes: std::mem::replace(&mut self.scopes, Vec::new()),
                            pos: (0, 0),
                        });
                    };
                    return Ok(module);
                }
            }
        }
    }
}

pub fn parse_str(source: &str) -> Result<crate::Module, ParseError> {
    Parser::new().parse(source)
}

#[test]
fn parse_types() {
    assert!(parse_str("const a : i32 = 2;").is_ok());
    assert!(parse_str("const a : x32 = 2;").is_err());
}
