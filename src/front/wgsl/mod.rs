//! Front end for consuming [WebGPU Shading Language][wgsl].
//!
//! [wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html

mod conv;
mod lexer;
#[cfg(test)]
mod tests;

use crate::{
    arena::{Arena, Handle},
    proc::{ensure_block_returns, ResolveContext, ResolveError, Typifier},
    FastHashMap,
};

use self::lexer::Lexer;
use std::num::NonZeroU32;
use thiserror::Error;

const SPACE: &str = "                                                                                                    ";

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Separator(char),
    DoubleColon,
    Paren(char),
    DoubleParen(char),
    Number {
        value: &'a str,
        ty: char,
        width: &'a str,
    },
    String(&'a str),
    Word(&'a str),
    Operation(char),
    LogicalOperation(char),
    ShiftOperation(char),
    Arrow,
    Unknown(char),
    UnterminatedString,
    Trivia,
    End,
}

#[derive(Clone, Debug, Error)]
pub enum Error<'a> {
    #[error("unexpected token {0:?}, expected {1}")]
    Unexpected(Token<'a>, &'a str),
    #[error("unable to parse `{0}` as integer: {1}")]
    BadInteger(&'a str, std::num::ParseIntError),
    #[error("unable to parse `{1}` as float: {1}")]
    BadFloat(&'a str, std::num::ParseFloatError),
    #[error("unable to parse `{0}{1}{2}` as scalar width: {3}")]
    BadScalarWidth(&'a str, char, &'a str, std::num::ParseIntError),
    #[error("bad field accessor `{0}`")]
    BadAccessor(&'a str),
    #[error("bad texture {0}`")]
    BadTexture(&'a str),
    #[error("bad texture coordinate")]
    BadCoordinate,
    #[error("invalid type cast to `{0}`")]
    BadTypeCast(&'a str),
    #[error(transparent)]
    InvalidResolve(ResolveError),
    #[error("invalid statement {0:?}, expected {1}")]
    InvalidStatement(crate::Statement, &'a str),
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
    #[error("unknown access: `{0}`")]
    UnknownAccess(&'a str),
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
    #[error("unknown conservative depth: `{0}`")]
    UnknownConservativeDepth(&'a str),
    #[error("array stride must not be 0")]
    ZeroStride,
    #[error("not a composite type: {0:?}")]
    NotCompositeType(Handle<crate::Type>),
    #[error("function redefinition: `{0}`")]
    FunctionRedefinition(&'a str),
    #[error("call to local `{0}(..)` can't be resolved")]
    UnknownLocalFunction(&'a str),
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
    functions: &'out Arena<crate::Function>,
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
            functions: self.functions,
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
            functions: self.functions,
            arguments: self.arguments,
        }
    }
}

struct SamplingContext {
    image: Handle<crate::Expression>,
    arrayed: bool,
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
    functions: &'out Arena<crate::Function>,
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
            functions: self.functions,
            arguments: self.arguments,
        }
    }

    fn resolve_type(
        &mut self,
        handle: Handle<crate::Expression>,
    ) -> Result<&crate::TypeInner, Error<'a>> {
        let resolve_ctx = ResolveContext {
            constants: self.constants,
            global_vars: self.global_vars,
            local_vars: self.local_vars,
            functions: self.functions,
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

    fn prepare_sampling(&mut self, image_name: &'a str) -> Result<SamplingContext, Error<'a>> {
        let image = self.lookup_ident.lookup(image_name)?;
        Ok(SamplingContext {
            image,
            arrayed: match *self.resolve_type(image)? {
                crate::TypeInner::Image { arrayed, .. } => arrayed,
                _ => return Err(Error::BadTexture(image_name)),
            },
        })
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
    //TODO: could be `const fn` once MSRV allows
    fn letter_pos(letter: char) -> u32 {
        match letter {
            'x' | 'r' => 0,
            'y' | 'g' => 1,
            'z' | 'b' => 2,
            'w' | 'a' => 3,
            _ => !0,
        }
    }

    fn make<'a>(
        base: Handle<crate::Expression>,
        base_size: crate::VectorSize,
        name: &'a str,
        expressions: &mut Arena<crate::Expression>,
    ) -> Result<Self, Error<'a>> {
        Ok(if name.len() > 1 {
            let mut components = Vec::with_capacity(name.len());
            for ch in name.chars() {
                let index = Self::letter_pos(ch);
                if index >= base_size as u32 {
                    return Err(Error::BadAccessor(name));
                }
                let expr = crate::Expression::AccessIndex { base, index };
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
            let index = Self::letter_pos(ch);
            if index >= base_size as u32 {
                return Err(Error::BadAccessor(name));
            }
            Composition::Single(crate::Expression::AccessIndex { base, index })
        })
    }
}

#[derive(Default)]
struct TypeDecoration {
    stride: Option<NonZeroU32>,
    access: crate::StorageAccess,
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

type LocalFunctionCall = (Handle<crate::Function>, Vec<Handle<crate::Expression>>);

struct ParsedVariable<'a> {
    name: &'a str,
    class: Option<crate::StorageClass>,
    ty: Handle<crate::Type>,
    access: crate::StorageAccess,
    init: Option<Handle<crate::Constant>>,
}

#[derive(Clone, Debug, Error)]
#[error("error while parsing WGSL in scopes {scopes:?} at line {line} pos {pos}: {error}")]
pub struct ParseError<'a> {
    pub error: Error<'a>,
    pub scopes: Vec<Scope>,
    pub line: usize,
    pub pos: usize,
}

pub struct Parser {
    scopes: Vec<Scope>,
    lookup_type: FastHashMap<String, Handle<crate::Type>>,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            scopes: Vec::new(),
            lookup_type: FastHashMap::default(),
        }
    }

    fn get_constant_inner<'a>(
        word: &'a str,
        ty: char,
        width: &'a str,
    ) -> Result<crate::ConstantInner, Error<'a>> {
        let value = match ty {
            'i' => word
                .parse()
                .map(crate::ScalarValue::Sint)
                .map_err(|err| Error::BadInteger(word, err))?,
            'u' => word
                .parse()
                .map(crate::ScalarValue::Uint)
                .map_err(|err| Error::BadInteger(word, err))?,
            'f' => word
                .parse()
                .map(crate::ScalarValue::Float)
                .map_err(|err| Error::BadFloat(word, err))?,
            _ => unreachable!(),
        };
        Ok(crate::ConstantInner::Scalar {
            value,
            width: if width.is_empty() {
                4
            } else {
                match width.parse::<crate::Bytes>() {
                    Ok(bits) => bits / 8,
                    Err(e) => return Err(Error::BadScalarWidth(word, ty, width, e)),
                }
            },
        })
    }

    fn parse_local_function_call<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<LocalFunctionCall>, Error<'a>> {
        let fun_handle = match ctx.functions.iter().find(|(_, fun)| match fun.name {
            Some(ref string) => string == name,
            None => false,
        }) {
            Some((fun_handle, _)) => fun_handle,
            None => return Ok(None),
        };

        let mut arguments = Vec::new();
        lexer.expect(Token::Paren('('))?;
        if !lexer.skip(Token::Paren(')')) {
            loop {
                let arg = self.parse_general_expression(lexer, ctx.reborrow())?;
                arguments.push(arg);
                match lexer.next() {
                    Token::Paren(')') => break,
                    Token::Separator(',') => (),
                    other => return Err(Error::Unexpected(other, "argument list separator")),
                }
            }
        }
        Ok(Some((fun_handle, arguments)))
    }

    fn parse_function_call_inner<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<crate::Expression>, Error<'a>> {
        Ok(if let Some(fun) = conv::map_relational_fun(name) {
            lexer.expect(Token::Paren('('))?;
            let argument = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.expect(Token::Paren(')'))?;
            Some(crate::Expression::Relational { fun, argument })
        } else if let Some(axis) = conv::map_derivative_axis(name) {
            lexer.expect(Token::Paren('('))?;
            let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.expect(Token::Paren(')'))?;
            Some(crate::Expression::Derivative { axis, expr })
        } else if let Some(fun) = conv::map_standard_fun(name) {
            lexer.expect(Token::Paren('('))?;
            let arg_count = fun.argument_count();
            let arg = self.parse_general_expression(lexer, ctx.reborrow())?;
            let arg1 = if arg_count > 1 {
                lexer.expect(Token::Separator(','))?;
                Some(self.parse_general_expression(lexer, ctx.reborrow())?)
            } else {
                None
            };
            let arg2 = if arg_count > 2 {
                lexer.expect(Token::Separator(','))?;
                Some(self.parse_general_expression(lexer, ctx.reborrow())?)
            } else {
                None
            };
            lexer.expect(Token::Paren(')'))?;
            Some(crate::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
            })
        } else if name == "select" {
            lexer.expect(Token::Paren('('))?;
            let accept = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.expect(Token::Separator(','))?;
            let reject = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.expect(Token::Separator(','))?;
            let condition = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.expect(Token::Paren(')'))?;
            Some(crate::Expression::Select {
                condition,
                accept,
                reject,
            })
        } else {
            // texture sampling
            match name {
                "textureSample" => {
                    lexer.expect(Token::Paren('('))?;
                    let image_name = lexer.next_ident()?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_name = lexer.next_ident()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, None, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name)?,
                        coordinate,
                        array_index,
                        offset,
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
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let level = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, None, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name)?,
                        coordinate,
                        array_index,
                        offset,
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
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let bias = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, None, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Bias(bias),
                        depth_ref: None,
                    })
                }
                "textureSampleGrad" => {
                    lexer.expect(Token::Paren('('))?;
                    let image_name = lexer.next_ident()?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_name = lexer.next_ident()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let x = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let y = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, None, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Gradient { x, y },
                        depth_ref: None,
                    })
                }
                "textureSampleCompare" => {
                    lexer.expect(Token::Paren('('))?;
                    let image_name = lexer.next_ident()?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_name = lexer.next_ident()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let reference = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, None, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Zero,
                        depth_ref: Some(reference),
                    })
                }
                "textureLoad" => {
                    lexer.expect(Token::Paren('('))?;
                    let image_name = lexer.next_ident()?;
                    let image = ctx.lookup_ident.lookup(image_name)?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let (class, arrayed) = match *ctx.resolve_type(image)? {
                        crate::TypeInner::Image { class, arrayed, .. } => (class, arrayed),
                        _ => return Err(Error::BadTexture(image_name)),
                    };
                    let array_index = if arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    let index = match class {
                        crate::ImageClass::Storage(_) => None,
                        // it's the MSAA index for multi-sampled, and LOD for the others
                        crate::ImageClass::Sampled { .. } | crate::ImageClass::Depth => {
                            lexer.expect(Token::Separator(','))?;
                            Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                        }
                    };
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageLoad {
                        image,
                        coordinate,
                        array_index,
                        index,
                    })
                }
                "textureDimensions" => {
                    lexer.expect(Token::Paren('('))?;
                    let image_name = lexer.next_ident()?;
                    let image = ctx.lookup_ident.lookup(image_name)?;
                    let level = if lexer.skip(Token::Separator(',')) {
                        let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                        Some(expr)
                    } else {
                        None
                    };
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::Size { level },
                    })
                }
                "textureNumLevels" => {
                    lexer.expect(Token::Paren('('))?;
                    let image_name = lexer.next_ident()?;
                    let image = ctx.lookup_ident.lookup(image_name)?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumLevels,
                    })
                }
                "textureNumLayers" => {
                    lexer.expect(Token::Paren('('))?;
                    let image_name = lexer.next_ident()?;
                    let image = ctx.lookup_ident.lookup(image_name)?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumLayers,
                    })
                }
                "textureNumSamples" => {
                    lexer.expect(Token::Paren('('))?;
                    let image_name = lexer.next_ident()?;
                    let image = ctx.lookup_ident.lookup(image_name)?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumSamples,
                    })
                }
                _ => self.parse_local_function_call(lexer, name, ctx)?.map(
                    |(function, arguments)| crate::Expression::Call {
                        function,
                        arguments,
                    },
                ),
            }
        })
    }

    fn parse_const_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        register_name: Option<&'a str>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Constant>, Error<'a>> {
        self.scopes.push(Scope::ConstantExpr);
        let inner = match lexer.next() {
            Token::Word("true") => crate::ConstantInner::Scalar {
                width: 1,
                value: crate::ScalarValue::Bool(true),
            },
            Token::Word("false") => crate::ConstantInner::Scalar {
                width: 1,
                value: crate::ScalarValue::Bool(false),
            },
            Token::Number { value, ty, width } => Self::get_constant_inner(value, ty, width)?,
            Token::Word(name) => {
                // look for an existing constant first
                for (handle, var) in const_arena.iter() {
                    match var.name {
                        Some(ref string) if string == name => {
                            self.scopes.pop();
                            return Ok(handle);
                        }
                        _ => {}
                    }
                }
                let composite_ty = self.parse_type_decl_name(
                    lexer,
                    name,
                    None,
                    TypeDecoration::default(),
                    type_arena,
                    const_arena,
                )?;
                lexer.expect(Token::Paren('('))?;
                let mut components = Vec::new();
                while !lexer.skip(Token::Paren(')')) {
                    if !components.is_empty() {
                        lexer.expect(Token::Separator(','))?;
                    }
                    let component =
                        self.parse_const_expression(lexer, None, type_arena, const_arena)?;
                    components.push(component);
                }
                crate::ConstantInner::Composite {
                    ty: composite_ty,
                    components,
                }
            }
            other => return Err(Error::Unexpected(other, "constant")),
        };

        let handle = if let Some(name) = register_name {
            const_arena.append(crate::Constant {
                name: Some(name.to_string()),
                specialization: None,
                inner,
            })
        } else {
            const_arena.fetch_or_append(crate::Constant {
                name: None,
                specialization: None,
                inner,
            })
        };

        self.scopes.pop();
        Ok(handle)
    }

    fn parse_primary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::PrimaryExpr);
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
                    inner: crate::ConstantInner::Scalar {
                        width: 1,
                        value: crate::ScalarValue::Bool(true),
                    },
                });
                crate::Expression::Constant(handle)
            }
            Token::Word("false") => {
                let handle = ctx.constants.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner: crate::ConstantInner::Scalar {
                        width: 1,
                        value: crate::ScalarValue::Bool(false),
                    },
                });
                crate::Expression::Constant(handle)
            }
            Token::Number { value, ty, width } => {
                let inner = Self::get_constant_inner(value, ty, width)?;
                let handle = ctx.constants.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner,
                });
                crate::Expression::Constant(handle)
            }
            Token::Word(word) => {
                if let Some(handle) = ctx.lookup_ident.get(word) {
                    self.scopes.pop();
                    return Ok(*handle);
                }
                //TODO: resolve the duplicate call in `parse_singular_expression`
                if let Some(expr) = self.parse_function_call_inner(lexer, word, ctx.reborrow())? {
                    expr
                } else {
                    let inner = self.parse_type_decl_impl(
                        lexer,
                        TypeDecoration::default(),
                        word,
                        ctx.types,
                        ctx.constants,
                    )?;
                    let kind = inner.scalar_kind();

                    lexer.expect(Token::Paren('('))?;
                    let mut components = Vec::new();
                    let mut last_component =
                        self.parse_general_expression(lexer, ctx.reborrow())?;
                    while lexer.skip(Token::Separator(',')) {
                        components.push(last_component);
                        last_component = self.parse_general_expression(lexer, ctx.reborrow())?;
                    }
                    lexer.expect(Token::Paren(')'))?;
                    if components.is_empty() {
                        let last_component_inner = ctx.resolve_type(last_component)?;
                        match (&inner, last_component_inner) {
                            (crate::TypeInner::Scalar { .. }, crate::TypeInner::Scalar { .. })
                            | (crate::TypeInner::Matrix { .. }, crate::TypeInner::Matrix { .. })
                            | (crate::TypeInner::Vector { .. }, crate::TypeInner::Vector { .. }) => {
                                crate::Expression::As {
                                    expr: last_component,
                                    kind: kind.ok_or(Error::BadTypeCast(word))?,
                                    convert: true,
                                }
                            }
                            _ => {
                                return Err(Error::BadTypeCast(word));
                            }
                        }
                    } else {
                        components.push(last_component);
                        let ty = ctx.types.fetch_or_append(crate::Type { name: None, inner });
                        crate::Expression::Compose { ty, components }
                    }
                }
            }
            other => return Err(Error::Unexpected(other, "primary expression")),
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
                        crate::TypeInner::Struct { ref members, .. } => {
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
        //TODO: refactor this to avoid backing up
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
            Token::Word(word) => self.parse_function_call_inner(lexer, word, ctx.reborrow())?,
            _ => None,
        };

        let handle = match expression {
            Some(expr) => ctx.expressions.append(expr),
            None => {
                *lexer = backup;
                self.parse_primary_expression(lexer, ctx.reborrow())?
            }
        };

        let post_handle = self.parse_postfix(lexer, ctx, handle)?;
        self.scopes.pop();
        Ok(post_handle)
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
    ) -> Result<(&'a str, Handle<crate::Type>, crate::StorageAccess), Error<'a>> {
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let (ty, access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
        Ok((name, ty, access))
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
        let (ty, access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;

        let init = if lexer.skip(Token::Operation('=')) {
            let handle = self.parse_const_expression(lexer, None, type_arena, const_arena)?;
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
            let mut span = 0;
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
                        Token::Word("span") if ready => {
                            lexer.expect(Token::Paren('('))?;
                            //Note: 0 is not handled
                            span = lexer.next_uint_literal()?;
                            lexer.expect(Token::Paren(')'))?;
                            ready = false;
                        }
                        Token::Word("offset") if ready => {
                            lexer.expect(Token::Paren('('))?;
                            let _offset = lexer.next_uint_literal()?;
                            lexer.expect(Token::Paren(')'))?;
                            ready = false;
                        }
                        other => return Err(Error::Unexpected(other, "decoration separator")),
                    }
                }
                self.scopes.pop();
            }
            let name = match lexer.next() {
                Token::Word(word) => word,
                Token::Paren('}') => return Ok(members),
                other => return Err(Error::Unexpected(other, "field name")),
            };
            lexer.expect(Token::Separator(':'))?;
            let (ty, _access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
            lexer.expect(Token::Separator(';'))?;
            members.push(crate::StructMember {
                name: Some(name.to_owned()),
                span: NonZeroU32::new(span),
                ty,
            });
        }
    }

    fn parse_type_decl_impl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        decoration: TypeDecoration,
        word: &'a str,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<crate::TypeInner, Error<'a>> {
        Ok(match word {
            "f32" => crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Float,
                width: 4,
            },
            "i32" => crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Sint,
                width: 4,
            },
            "u32" => crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Uint,
                width: 4,
            },
            "bool" => crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Bool,
                width: 1,
            },
            "vec2" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            "vec3" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            "vec4" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Quad,
                    kind,
                    width,
                }
            }
            "mat2x2" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            "mat2x3" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            "mat2x4" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            "mat3x2" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            "mat3x3" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            "mat3x4" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            "mat4x2" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            "mat4x3" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            "mat4x4" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            "ptr" => {
                lexer.expect_generic_paren('<')?;
                let class = conv::map_storage_class(lexer.next_ident()?)?;
                lexer.expect(Token::Separator(','))?;
                let (base, _access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                lexer.expect_generic_paren('>')?;
                crate::TypeInner::Pointer { base, class }
            }
            "array" => {
                lexer.expect_generic_paren('<')?;
                let (base, _access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let const_handle =
                        self.parse_const_expression(lexer, None, type_arena, const_arena)?;
                    crate::ArraySize::Constant(const_handle)
                } else {
                    crate::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                crate::TypeInner::Array {
                    base,
                    size,
                    stride: decoration.stride,
                }
            }
            "sampler" => crate::TypeInner::Sampler { comparison: false },
            "sampler_comparison" => crate::TypeInner::Sampler { comparison: true },
            "texture_1d" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_1d_array" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d_array" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_3d" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube_array" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_multisampled_2d" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_multisampled_2d_array" => {
                let (kind, _) = lexer.next_scalar_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_depth_2d" => crate::TypeInner::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth,
            },
            "texture_depth_2d_array" => crate::TypeInner::Image {
                dim: crate::ImageDimension::D2,
                arrayed: true,
                class: crate::ImageClass::Depth,
            },
            "texture_depth_cube" => crate::TypeInner::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: false,
                class: crate::ImageClass::Depth,
            },
            "texture_depth_cube_array" => crate::TypeInner::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: true,
                class: crate::ImageClass::Depth,
            },
            "texture_storage_1d" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            "texture_storage_1d_array" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Storage(format),
                }
            }
            "texture_storage_2d" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            "texture_storage_2d_array" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Storage(format),
                }
            }
            "texture_storage_3d" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            _ => return Err(Error::UnknownType(word)),
        })
    }

    /// Parse type declaration of a given name and decoration.
    fn parse_type_decl_name<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        debug_name: Option<&'a str>,
        decoration: TypeDecoration,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Type>, Error<'a>> {
        Ok(match self.lookup_type.get(name) {
            Some(&handle) => handle,
            None => {
                let inner =
                    self.parse_type_decl_impl(lexer, decoration, name, type_arena, const_arena)?;
                type_arena.fetch_or_append(crate::Type {
                    name: debug_name.map(|s| s.to_string()),
                    inner,
                })
            }
        })
    }

    fn parse_type_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        debug_name: Option<&'a str>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<(Handle<crate::Type>, crate::StorageAccess), Error<'a>> {
        self.scopes.push(Scope::TypeDecl);
        let mut decoration = TypeDecoration::default();

        if lexer.skip(Token::DoubleParen('[')) {
            self.scopes.push(Scope::Decoration);
            loop {
                match lexer.next() {
                    Token::Word("access") => {
                        lexer.expect(Token::Paren('('))?;
                        decoration.access = match lexer.next_ident()? {
                            "read" => crate::StorageAccess::LOAD,
                            "write" => crate::StorageAccess::STORE,
                            "read_write" => crate::StorageAccess::all(),
                            other => return Err(Error::UnknownAccess(other)),
                        };
                        lexer.expect(Token::Paren(')'))?;
                    }
                    Token::Word("stride") => {
                        lexer.expect(Token::Paren('('))?;
                        decoration.stride = Some(
                            NonZeroU32::new(lexer.next_uint_literal()?).ok_or(Error::ZeroStride)?,
                        );
                        lexer.expect(Token::Paren(')'))?;
                    }
                    Token::DoubleParen(']') => break,
                    other => return Err(Error::Unexpected(other, "type decoration")),
                }
            }
            self.scopes.pop();
        }

        let storage_access = decoration.access;
        let name = lexer.next_ident()?;
        let handle = self.parse_type_decl_name(
            lexer,
            name,
            debug_name,
            decoration,
            type_arena,
            const_arena,
        )?;
        self.scopes.pop();
        Ok((handle, storage_access))
    }

    fn parse_statement<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, '_>,
        is_uniform_control_flow: bool,
    ) -> Result<Option<crate::Statement>, Error<'a>> {
        let word = match lexer.next() {
            Token::Separator(';') => return Ok(None),
            Token::Paren('{') => {
                self.scopes.push(Scope::Block);
                let mut statements = Vec::new();
                while !lexer.skip(Token::Paren('}')) {
                    let s =
                        self.parse_statement(lexer, context.reborrow(), is_uniform_control_flow)?;
                    statements.extend(s);
                }
                self.scopes.pop();
                let block = crate::Statement::Block(statements);
                return Ok(Some(block));
            }
            Token::Word(word) => word,
            other => return Err(Error::Unexpected(other, "statement")),
        };

        self.scopes.push(Scope::Statement);
        let statement = match word {
            "const" => {
                let (name, _ty, _access) =
                    self.parse_variable_ident_decl(lexer, context.types, context.constants)?;
                lexer.expect(Token::Operation('='))?;
                let expr_id = self.parse_general_expression(lexer, context.as_expression())?;
                lexer.expect(Token::Separator(';'))?;
                context.lookup_ident.insert(name, expr_id);
                None
            }
            "var" => {
                enum Init {
                    Empty,
                    Constant(Handle<crate::Constant>),
                    Variable(Handle<crate::Expression>),
                }

                let (name, ty, _access) =
                    self.parse_variable_ident_decl(lexer, context.types, context.constants)?;

                let init = if lexer.skip(Token::Operation('=')) {
                    let value = self.parse_general_expression(lexer, context.as_expression())?;
                    match context.expressions[value] {
                        crate::Expression::Constant(handle) if is_uniform_control_flow => {
                            Init::Constant(handle)
                        }
                        _ => Init::Variable(value),
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
                    Init::Variable(value) => Some(crate::Statement::Store {
                        pointer: expr_id,
                        value,
                    }),
                    _ => None,
                }
            }
            "return" => {
                let value = if lexer.peek() != Token::Separator(';') {
                    Some(self.parse_general_expression(lexer, context.as_expression())?)
                } else {
                    None
                };
                lexer.expect(Token::Separator(';'))?;
                Some(crate::Statement::Return { value })
            }
            "if" => {
                lexer.expect(Token::Paren('('))?;
                let condition = self.parse_general_expression(lexer, context.as_expression())?;
                lexer.expect(Token::Paren(')'))?;

                let accept = self.parse_block(lexer, context.reborrow(), false)?;
                let mut elsif_stack = Vec::new();
                while lexer.skip(Token::Word("elseif")) {
                    lexer.expect(Token::Paren('('))?;
                    let other_condition =
                        self.parse_general_expression(lexer, context.as_expression())?;
                    lexer.expect(Token::Paren(')'))?;
                    let other_block = self.parse_block(lexer, context.reborrow(), false)?;
                    elsif_stack.push((other_condition, other_block));
                }
                let mut reject = if lexer.skip(Token::Word("else")) {
                    self.parse_block(lexer, context.reborrow(), false)?
                } else {
                    Vec::new()
                };
                // reverse-fold the else-if blocks
                //Note: we may consider uplifting this to the IR
                for (other_cond, other_block) in elsif_stack.drain(..).rev() {
                    reject = vec![crate::Statement::If {
                        condition: other_cond,
                        accept: other_block,
                        reject,
                    }];
                }

                Some(crate::Statement::If {
                    condition,
                    accept,
                    reject,
                })
            }
            "switch" => {
                lexer.expect(Token::Paren('('))?;
                let selector = self.parse_general_expression(lexer, context.as_expression())?;
                lexer.expect(Token::Paren(')'))?;
                lexer.expect(Token::Paren('{'))?;
                let mut cases = Vec::new();
                let mut default = Vec::new();

                loop {
                    // cases + default
                    match lexer.next() {
                        Token::Word("case") => {
                            let value = loop {
                                // values
                                let value = lexer.next_sint_literal()?;
                                match lexer.next() {
                                    Token::Separator(',') => {
                                        cases.push(crate::SwitchCase {
                                            value,
                                            body: Vec::new(),
                                            fall_through: true,
                                        });
                                    }
                                    Token::Separator(':') => break value,
                                    other => {
                                        return Err(Error::Unexpected(other, "case separator"))
                                    }
                                }
                            };

                            let mut body = Vec::new();
                            lexer.expect(Token::Paren('{'))?;
                            let fall_through = loop {
                                // default statements
                                if lexer.skip(Token::Word("fallthrough")) {
                                    lexer.expect(Token::Separator(';'))?;
                                    lexer.expect(Token::Paren('}'))?;
                                    break true;
                                }
                                if lexer.skip(Token::Paren('}')) {
                                    break false;
                                }
                                let s = self.parse_statement(lexer, context.reborrow(), false)?;
                                body.extend(s);
                            };

                            cases.push(crate::SwitchCase {
                                value,
                                body,
                                fall_through,
                            });
                        }
                        Token::Word("default") => {
                            lexer.expect(Token::Separator(':'))?;
                            default = self.parse_block(lexer, context.reborrow(), false)?;
                        }
                        Token::Paren('}') => break,
                        other => return Err(Error::Unexpected(other, "switch item")),
                    }
                }

                Some(crate::Statement::Switch {
                    selector,
                    cases,
                    default,
                })
            }
            "loop" => {
                let mut body = Vec::new();
                let mut continuing = Vec::new();
                lexer.expect(Token::Paren('{'))?;

                loop {
                    if lexer.skip(Token::Word("continuing")) {
                        continuing = self.parse_block(lexer, context.reborrow(), false)?;
                        lexer.expect(Token::Paren('}'))?;
                        break;
                    }
                    if lexer.skip(Token::Paren('}')) {
                        break;
                    }
                    let s = self.parse_statement(lexer, context.reborrow(), false)?;
                    body.extend(s);
                }

                Some(crate::Statement::Loop { body, continuing })
            }
            "for" => {
                let mut block = Vec::new();

                lexer.expect(Token::Paren('('))?;
                let initialization = if lexer.skip(Token::Separator(';')) {
                    None
                } else {
                    let statement =
                        self.parse_statement(lexer, context.reborrow(), is_uniform_control_flow)?;
                    match statement {
                        Some(crate::Statement::Store { .. })
                        | Some(crate::Statement::Call { .. }) => statement,
                        Some(other) => {
                            return Err(Error::InvalidStatement(
                                other,
                                "variable, assignment or function call",
                            ))
                        }
                        None => None,
                    }
                };
                let condition = if lexer.skip(Token::Separator(';')) {
                    None
                } else {
                    let condition =
                        Some(self.parse_general_expression(lexer, context.as_expression())?);
                    lexer.expect(Token::Separator(';'))?;
                    condition
                };
                let continuing = if let Token::Word(ident) = lexer.peek() {
                    // manually parse the next statement here instead of calling parse_statement
                    // because the statement is not terminated with a semicolon
                    let _ = lexer.next();
                    Some(match context.lookup_ident.get(ident) {
                        Some(&var_expr) => {
                            let left =
                                self.parse_postfix(lexer, context.as_expression(), var_expr)?;
                            lexer.expect(Token::Operation('='))?;
                            let value =
                                self.parse_general_expression(lexer, context.as_expression())?;
                            crate::Statement::Store {
                                pointer: left,
                                value,
                            }
                        }
                        None => {
                            let (function, arguments) = self
                                .parse_local_function_call(lexer, ident, context.as_expression())?
                                .ok_or(Error::UnknownLocalFunction(ident))?;
                            crate::Statement::Call {
                                function,
                                arguments,
                            }
                        }
                    })
                } else {
                    None
                };
                lexer.expect(Token::Paren(')'))?;
                lexer.expect(Token::Paren('{'))?;

                block.extend(initialization);

                let mut body = Vec::new();
                if let Some(condition) = condition {
                    body.push(crate::Statement::If {
                        condition,
                        accept: Vec::new(),
                        reject: [crate::Statement::Break].to_vec(),
                    });
                }
                while !lexer.skip(Token::Paren('}')) {
                    let s = self.parse_statement(lexer, context.reborrow(), false)?;
                    body.extend(s);
                }

                block.push(crate::Statement::Loop {
                    body,
                    continuing: continuing.into_iter().collect(),
                });

                Some(crate::Statement::Block(block))
            }
            "break" => Some(crate::Statement::Break),
            "continue" => Some(crate::Statement::Continue),
            "discard" => Some(crate::Statement::Kill),
            "textureStore" => {
                lexer.expect(Token::Paren('('))?;
                let image_name = lexer.next_ident()?;
                let image = context.lookup_ident.lookup(image_name)?;
                lexer.expect(Token::Separator(','))?;
                let coordinate = self.parse_general_expression(lexer, context.as_expression())?;
                let arrayed = match *context.as_expression().resolve_type(image)? {
                    crate::TypeInner::Image { arrayed, .. } => arrayed,
                    _ => return Err(Error::BadTexture(image_name)),
                };
                let array_index = if arrayed {
                    lexer.expect(Token::Separator(','))?;
                    Some(self.parse_general_expression(lexer, context.as_expression())?)
                } else {
                    None
                };
                lexer.expect(Token::Separator(','))?;
                let value = self.parse_general_expression(lexer, context.as_expression())?;
                lexer.expect(Token::Paren(')'))?;
                Some(crate::Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                })
            }
            // assignment or a function call
            ident => {
                if let Some(&var_expr) = context.lookup_ident.get(ident) {
                    let left = self.parse_postfix(lexer, context.as_expression(), var_expr)?;
                    lexer.expect(Token::Operation('='))?;
                    let value = self.parse_general_expression(lexer, context.as_expression())?;
                    lexer.expect(Token::Separator(';'))?;
                    Some(crate::Statement::Store {
                        pointer: left,
                        value,
                    })
                } else {
                    let (function, arguments) = self
                        .parse_local_function_call(lexer, ident, context.as_expression())?
                        .ok_or(Error::UnknownLocalFunction(ident))?;
                    lexer.expect(Token::Separator(';'))?;
                    Some(crate::Statement::Call {
                        function,
                        arguments,
                    })
                }
            }
        };
        self.scopes.pop();
        Ok(statement)
    }

    fn parse_block<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, '_>,
        is_uniform_control_flow: bool,
    ) -> Result<Vec<crate::Statement>, Error<'a>> {
        self.scopes.push(Scope::Block);
        lexer.expect(Token::Paren('{'))?;
        let mut statements = Vec::new();
        while !lexer.skip(Token::Paren('}')) {
            let s = self.parse_statement(lexer, context.reborrow(), is_uniform_control_flow)?;
            statements.extend(s);
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
        // populate initial expressions
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
            let (param_name, param_type, _access) =
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
        let return_type = if lexer.skip(Token::Arrow) && !lexer.skip(Token::Word("void")) {
            let (handle, _access) =
                self.parse_type_decl(lexer, None, &mut module.types, &mut module.constants)?;
            Some(handle)
        } else {
            None
        };

        let mut fun = crate::Function {
            name: Some(fun_name.to_string()),
            arguments,
            return_type,
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
                functions: &module.functions,
                arguments: &fun.arguments,
            },
            true,
        )?;
        // fixup the IR
        ensure_block_returns(&mut fun.body);
        // done
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
        let mut is_block = false;
        let mut workgroup_size = [0u32; 3];
        let mut early_depth_test = None;

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
                    "block" => {
                        is_block = true;
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
                                other => {
                                    return Err(Error::Unexpected(
                                        other,
                                        "workgroup size separator",
                                    ))
                                }
                            }
                        }
                        for size in workgroup_size.iter_mut() {
                            if *size == 0 {
                                *size = 1;
                            }
                        }
                    }
                    "early_depth_test" => {
                        let conservative = if lexer.skip(Token::Paren('(')) {
                            let value = conv::map_conservative_depth(lexer.next_ident()?)?;
                            lexer.expect(Token::Paren(')'))?;
                            Some(value)
                        } else {
                            None
                        };
                        early_depth_test = Some(crate::EarlyDepthTest { conservative });
                    }
                    word => return Err(Error::UnknownDecoration(word)),
                }
                match lexer.next() {
                    Token::DoubleParen(']') => {
                        break;
                    }
                    Token::Separator(',') => {}
                    other => return Err(Error::Unexpected(other, "decoration separator")),
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
            Token::Word("struct") => {
                let name = lexer.next_ident()?;
                let members =
                    self.parse_struct_body(lexer, &mut module.types, &mut module.constants)?;
                let ty = module.types.fetch_or_append(crate::Type {
                    name: Some(name.to_string()),
                    inner: crate::TypeInner::Struct {
                        block: is_block,
                        members,
                    },
                });
                self.lookup_type.insert(name.to_owned(), ty);
                lexer.expect(Token::Separator(';'))?;
            }
            Token::Word("type") => {
                let name = lexer.next_ident()?;
                lexer.expect(Token::Operation('='))?;
                let (ty, _access) = self.parse_type_decl(
                    lexer,
                    Some(name),
                    &mut module.types,
                    &mut module.constants,
                )?;
                self.lookup_type.insert(name.to_owned(), ty);
                lexer.expect(Token::Separator(';'))?;
            }
            Token::Word("const") => {
                let (name, _ty, _access) = self.parse_variable_ident_decl(
                    lexer,
                    &mut module.types,
                    &mut module.constants,
                )?;
                lexer.expect(Token::Operation('='))?;
                let const_handle = self.parse_const_expression(
                    lexer,
                    Some(name),
                    &mut module.types,
                    &mut module.constants,
                )?;
                //TODO: check `ty` against `const_handle`.
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
                        Some(crate::Binding::Resource { .. }) => {
                            match module.types[pvar.ty].inner {
                                crate::TypeInner::Struct { .. } if pvar.access.is_empty() => {
                                    crate::StorageClass::Uniform
                                }
                                crate::TypeInner::Struct { .. } => crate::StorageClass::Storage,
                                _ => crate::StorageClass::Handle,
                            }
                        }
                        _ => crate::StorageClass::Private,
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
                                early_depth_test,
                                workgroup_size,
                                function,
                            },
                        )
                        .is_some(),
                    None => {
                        if module.functions.iter().any(|(_, fun)| match fun.name {
                            Some(ref string) => string == name,
                            None => false,
                        }) {
                            true
                        } else {
                            module.functions.append(function);
                            false
                        }
                    }
                };
                if already_declared {
                    return Err(Error::FunctionRedefinition(name));
                }
            }
            Token::End => return Ok(false),
            other => return Err(Error::Unexpected(other, "global item")),
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

        let mut module = crate::Module::default();
        let mut lexer = Lexer::new(source);
        let mut lookup_global_expression = FastHashMap::default();
        loop {
            match self.parse_global_decl(&mut lexer, &mut module, &mut lookup_global_expression) {
                Err(error) => {
                    let pos = lexer.offset_from(source);
                    let (mut rows, mut cols) = (0, 1);
                    let (mut prev_line, mut cur_line) = ("", "");
                    for line in source[..pos].lines() {
                        rows += 1;
                        cols = line.len();
                        prev_line = cur_line;
                        cur_line = line;
                    }
                    log::error!("|\t{}", prev_line);
                    log::error!(
                        ">\t{}{}",
                        cur_line,
                        source[pos..].lines().next().unwrap_or_default()
                    );
                    if cols <= SPACE.len() {
                        log::error!("|\t{}^", &SPACE[..cols]);
                    }
                    return Err(ParseError {
                        error,
                        scopes: std::mem::replace(&mut self.scopes, Vec::new()),
                        line: rows,
                        pos: cols,
                    });
                }
                Ok(true) => {}
                Ok(false) => {
                    if !self.scopes.is_empty() {
                        log::error!("Reached the end of file, but scopes are not closed");
                        return Err(ParseError {
                            error: Error::Other,
                            scopes: std::mem::replace(&mut self.scopes, Vec::new()),
                            line: 0,
                            pos: 0,
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
