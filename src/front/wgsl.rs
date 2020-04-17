use crate::{
    arena::{Arena, Handle},
    proc::{Typifier, ResolveError},
    FastHashMap,
};


#[derive(Debug, PartialEq)]
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
    ArithmeticShiftOperation(char),
    Arrow,
    Unknown(char),
    UnterminatedString,
    End,
}

mod lex {
    use super::Token;

    fn _consume_str<'a>(input: &'a str, what: &str) -> Option<&'a str> {
        if input.starts_with(what) {
            Some(&input[what.len() ..])
        } else {
            None
        }
    }

    fn consume_any(input: &str, what: impl Fn(char) -> bool) -> (&str, &str) {
        let pos = input.find(|c| !what(c)).unwrap_or_else(|| input.len());
        input.split_at(pos)
    }

    pub fn consume_token(mut input: &str) -> (Token<'_>, &str) {
        input = input.trim_start();
        let mut chars = input.chars();
        let cur = match chars.next() {
            Some(c) => c,
            None => return (Token::End, input),
        };
        match cur {
            ':' => {
                input = chars.as_str();
                if chars.next() == Some(':') {
                    (Token::DoubleColon, chars.as_str())
                } else {
                    (Token::Separator(cur), input)
                }
            }
            ';' | ',' | '.' => {
                (Token::Separator(cur), chars.as_str())
            }
            '(' | ')' | '{' | '}' => {
                (Token::Paren(cur), chars.as_str())
            }
            '<' | '>' => {
                input = chars.as_str();
                let next = chars.next();
                if next == Some('=') {
                    (Token::LogicalOperation(cur), chars.as_str())
                } else if next == Some(cur) {
                    input = chars.as_str();
                    if chars.next() == Some(cur) {
                        (Token::ArithmeticShiftOperation(cur), chars.as_str())
                    } else {
                        (Token::ShiftOperation(cur), input)
                    }
                } else {
                    (Token::Paren(cur), input)
                }
            }
            '[' | ']' => {
                input = chars.as_str();
                if chars.next() == Some(cur) {
                    (Token::DoubleParen(cur), chars.as_str())
                } else {
                    (Token::Paren(cur), input)
                }
            }
            '0' ..= '9' => {
                let (number, rest) = consume_any(input, |c| (c>='0' && c<='9' || c=='.'));
                (Token::Number(number), rest)
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let (word, rest) = consume_any(input, |c| c.is_alphanumeric() || c=='_');
                (Token::Word(word), rest)
            }
            '"' => {
                let base = chars.as_str();
                let len = match chars.position(|c| c == '"') {
                    Some(pos) => pos,
                    None => return (Token::UnterminatedString, chars.as_str())
                };
                (Token::String(&base[..len]), chars.as_str())
            }
            '-' => {
                input = chars.as_str();
                if chars.next() == Some('>') {
                    (Token::Arrow, chars.as_str())
                } else {
                    (Token::Operation(cur), input)
                }
            }
            '+' | '*' | '/' | '%' | '^' => {
                (Token::Operation(cur), chars.as_str())
            }
            '!' => {
                if chars.next() == Some('=') {
                    (Token::LogicalOperation(cur), chars.as_str())
                } else {
                    (Token::Operation(cur), input)
                }
            }
            '=' | '&' | '|'  => {
                input = chars.as_str();
                if chars.next() == Some(cur) {
                    (Token::LogicalOperation(cur), chars.as_str())
                } else {
                    (Token::Operation(cur), input)
                }
            }
            '#' => {
                match chars.position(|c| c == '\n' || c == '\r') {
                    Some(_) => consume_token(chars.as_str()),
                    None => (Token::End, chars.as_str())
                }
            }
            _ => (Token::Unknown(cur), chars.as_str())
        }
    }
}

#[derive(Debug)]
pub enum Error<'a> {
    Unexpected(Token<'a>),
    BadInteger(&'a str, std::num::ParseIntError),
    BadFloat(&'a str, std::num::ParseFloatError),
    BadAccessor(&'a str),
    InvalidResolve(ResolveError),
    UnknownImport(&'a str),
    UnknownStorageClass(&'a str),
    UnknownDecoration(&'a str),
    UnknownBuiltin(&'a str),
    UnknownPipelineStage(&'a str),
    UnknownIdent(&'a str),
    UnknownType(&'a str),
    UnknownFunction(&'a str),
    MutabilityViolation(&'a str),
    Other,
}

#[derive(Clone)]
struct Lexer<'a> {
    input: &'a str,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer { input }
    }

    #[must_use]
    fn next(&mut self) -> Token<'a> {
        let (token, rest) = lex::consume_token(self.input);
        self.input = rest;
        token
    }

    #[must_use]
    fn peek(&mut self) -> Token<'a> {
        self.clone().next()
    }

    fn expect(&mut self, expected: Token<'a>) -> Result<(), Error<'a>> {
        let token = self.next();
        if token == expected {
            Ok(())
        } else {
            Err(Error::Unexpected(token))
        }
    }

    fn skip(&mut self, what: Token<'a>) -> bool {
        let (token, rest) = lex::consume_token(self.input);
        if token == what {
            self.input = rest;
            true
        } else {
            false
        }
    }

    fn next_ident(&mut self) -> Result<&'a str, Error<'a>> {
        match self.next() {
            Token::Word(word) => Ok(word),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn _next_float_literal(&mut self) -> Result<f32, Error<'a>> {
        match self.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadFloat(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn next_uint_literal(&mut self) -> Result<u32, Error<'a>> {
        match self.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadInteger(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn _next_sint_literal(&mut self) -> Result<i32, Error<'a>> {
        match self.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadInteger(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn next_scalar_generic(&mut self) -> Result<(crate::ScalarKind, u8), Error<'a>> {
        self.expect(Token::Paren('<'))?;
        let pair = match self.next() {
            Token::Word("f32") => (crate::ScalarKind::Float, 32),
            Token::Word("i32") => (crate::ScalarKind::Sint, 32),
            Token::Word("u32") => (crate::ScalarKind::Uint, 32),
            other => return Err(Error::Unexpected(other)),
        };
        self.expect(Token::Paren('>'))?;
        Ok(pair)
    }
}

trait StringValueLookup<'a> {
    type Value;
    fn lookup(&self, key: &'a str) -> Result<Self::Value, Error<'a>>;
}
impl<'a> StringValueLookup<'a> for FastHashMap<&'a str, Handle<crate::Expression>> {
    type Value = Handle<crate::Expression>;
    fn lookup(&self, key: &'a str) -> Result<Self::Value, Error<'a>> {
        self.get(key)
            .cloned()
            .ok_or(Error::UnknownIdent(key))
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
        }
    }

    fn resolve_type(&mut self, handle: Handle<crate::Expression>) -> Result<Handle<crate::Type>, Error<'a>> {
        self.typifier
            .resolve(handle, self.expressions, self.types, self.constants, self.global_vars, self.local_vars)
            .map_err(Error::InvalidResolve)
    }

    fn parse_binary_op(
        &mut self,
        lexer: &mut Lexer<'a>,
        classifier: impl Fn(Token<'a>) -> Option<crate::BinaryOperator>,
        mut parser: impl FnMut(&mut Lexer<'a>, ExpressionContext<'a, '_, '_>) -> Result<Handle<crate::Expression>, Error<'a>>,
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

#[derive(Debug)]
pub struct ParseError<'a> {
    pub error: Error<'a>,
    pub scopes: Vec<Scope>,
    pub pos: (usize, usize),
}

pub struct Parser {
    scopes: Vec<Scope>,
    lookup_type: FastHashMap<String, Handle<crate::Type>>,
    std_namespace: Option<String>,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            scopes: Vec::new(),
            lookup_type: FastHashMap::default(),
            std_namespace: None,
        }
    }

    fn get_storage_class(word: &str) -> Result<spirv::StorageClass, Error<'_>> {
        match word {
            "in" => Ok(spirv::StorageClass::Input),
            "out" => Ok(spirv::StorageClass::Output),
            "uniform" => Ok(spirv::StorageClass::Uniform),
            "storage_buffer" => Ok(spirv::StorageClass::StorageBuffer),
            _ => Err(Error::UnknownStorageClass(word)),
        }
    }

    fn get_built_in(word: &str) -> Result<spirv::BuiltIn, Error<'_>> {
        match word {
            "position" => Ok(spirv::BuiltIn::Position),
            "vertex_idx" => Ok(spirv::BuiltIn::VertexId),
            "global_invocation_id" => Ok(spirv::BuiltIn::GlobalInvocationId),
            _ => Err(Error::UnknownBuiltin(word)),
        }
    }

    fn get_execution_model(word: &str) -> Result<spirv::ExecutionModel, Error<'_>> {
        match word {
            "vertex" => Ok(spirv::ExecutionModel::Vertex),
            "fragment" => Ok(spirv::ExecutionModel::Fragment),
            "compute" => Ok(spirv::ExecutionModel::GLCompute),
            _ => Err(Error::UnknownPipelineStage(word)),
        }
    }

    fn get_constant_inner(word: &str) -> Result<(crate::ConstantInner, crate::ScalarKind), Error<'_>> {
        if word.contains('.') {
            word
                .parse()
                .map(|f|(crate::ConstantInner::Float(f), crate::ScalarKind::Float))
                .map_err(|err| Error::BadFloat(word, err))
        } else {
            word
                .parse()
                .map(|i|(crate::ConstantInner::Sint(i), crate::ScalarKind::Sint))
                .map_err(|err| Error::BadInteger(word, err))
        }
    }

    fn parse_const_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<crate::ConstantInner, Error<'a>> {
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
                let _ty = self.parse_type_decl(lexer, type_arena);
                lexer.expect(Token::Paren('('))?;
                while !lexer.skip(Token::Paren(')')) {
                    let _ = self.parse_const_expression(lexer, type_arena, const_arena)?;
                }
                unimplemented!()
            }
        };
        self.scopes.pop();
        Ok(inner)
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
                    ty: Typifier::deduce_type_handle(
                        crate::TypeInner::Scalar {
                            kind: crate::ScalarKind::Bool,
                            width: 1,
                        },
                        ctx.types,
                    )
                });
                crate::Expression::Constant(handle)
            }
            Token::Word("false") => {
                let handle = ctx.constants.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner: crate::ConstantInner::Bool(false),
                    ty: Typifier::deduce_type_handle(
                        crate::TypeInner::Scalar {
                            kind: crate::ScalarKind::Bool,
                            width: 1,
                        },
                        ctx.types,
                    )
                });
                crate::Expression::Constant(handle)
            }
            Token::Number(word) => {
                let (inner, kind) = Self::get_constant_inner(word)?;
                let handle = ctx.constants.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner,
                    ty: Typifier::deduce_type_handle(
                        crate::TypeInner::Scalar { kind, width: 32 },
                        ctx.types,
                    )
                });
                crate::Expression::Constant(handle)
            }
            Token::Word(word) => {
                if let Some(handle) = ctx.lookup_ident.get(word) {
                    self.scopes.pop();
                    return Ok(*handle);
                }
                if self.std_namespace.as_ref().map(|s| s.as_str()) == Some(word) {
                    lexer.expect(Token::DoubleColon)?;
                    let name = lexer.next_ident()?;
                    let mut arguments = Vec::new();
                    lexer.expect(Token::Paren('('))?;
                    while !lexer.skip(Token::Paren(')')) {
                        if !arguments.is_empty() {
                            lexer.expect(Token::Separator(','))?;
                        }
                        let arg = self.parse_general_expression(lexer, ctx.reborrow())?;
                        arguments.push(arg);
                    }
                    crate::Expression::Call {
                        name: name.to_owned(),
                        arguments,
                    }
                } else {
                    *lexer = backup;
                    let ty = self.parse_type_decl(lexer, ctx.types)?;
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
                    let type_handle = ctx.resolve_type(handle)?;
                    let base_type = &ctx.types[type_handle];
                    let expression = match base_type.inner {
                        crate::TypeInner::Struct { ref members } => {
                            let index = members
                                .iter()
                                .position(|m| m.name.as_ref().map(|s| s.as_str()) == Some(name))
                                .ok_or(Error::BadAccessor(name))? as u32;
                            crate::Expression::AccessIndex {
                                base: handle,
                                index,
                            }
                        }
                        crate::TypeInner::Vector { size, kind, width } |
                        crate::TypeInner::Matrix { columns: size, kind, width, .. } => {
                            const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];
                            if name.len() > 1 {
                                let mut components = Vec::with_capacity(name.len());
                                for ch in name.chars() {
                                    let expr = crate::Expression::AccessIndex {
                                        base: handle,
                                        index: MEMBERS[.. size as usize]
                                            .iter()
                                            .position(|&m| m == ch)
                                            .ok_or(Error::BadAccessor(name))? as u32,
                                    };
                                    components.push(ctx.expressions.append(expr));
                                }
                                let size = match name.len() {
                                    2 => crate::VectorSize::Bi,
                                    3 => crate::VectorSize::Tri,
                                    4 => crate::VectorSize::Quad,
                                    _ => return Err(Error::BadAccessor(name)),
                                };
                                let inner = if let crate::TypeInner::Matrix { rows, .. } = base_type.inner {
                                    crate::TypeInner::Matrix { columns: size, rows, kind, width }
                                } else {
                                    crate::TypeInner::Vector { size, kind, width }
                                };
                                crate::Expression::Compose {
                                    ty: Typifier::deduce_type_handle(inner, ctx.types),
                                    components,
                                }
                            } else {
                                let ch = name.chars().next().unwrap();
                                let index = MEMBERS[.. size as usize]
                                    .iter()
                                    .position(|&m| m == ch)
                                    .ok_or(Error::BadAccessor(name))? as u32;
                                crate::Expression::AccessIndex {
                                    base: handle,
                                    index,
                                }
                            }
                        }
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
        fn get_intrinsic(word: &str) -> Option<crate::IntrinsicFunction> {
            match word {
                "any" => Some(crate::IntrinsicFunction::Any),
                "all" => Some(crate::IntrinsicFunction::All),
                "is_nan" => Some(crate::IntrinsicFunction::IsNan),
                "is_inf" => Some(crate::IntrinsicFunction::IsInf),
                "is_normal" => Some(crate::IntrinsicFunction::IsNormal),
                _ => None,
            }
        }
        fn get_derivative(word: &str) -> Option<crate::DerivativeAxis> {
            match word {
                "dpdx" => Some(crate::DerivativeAxis::X),
                "dpdy" => Some(crate::DerivativeAxis::Y),
                "dwidth" => Some(crate::DerivativeAxis::Width),
                _ => None,
            }
        }

        self.scopes.push(Scope::SingularExpr);
        let backup = lexer.clone();
        let expression = match lexer.next() {
            Token::Operation('-') => {
                Some(crate::Expression::Unary {
                    op: crate::UnaryOperator::Negate,
                    expr: self.parse_singular_expression(lexer, ctx.reborrow())?,
                })
            }
            Token::Operation('!') => {
                Some(crate::Expression::Unary {
                    op: crate::UnaryOperator::Not,
                    expr: self.parse_singular_expression(lexer, ctx.reborrow())?,
                })
            }
            Token::Word(word) => {
                if let Some(fun) = get_intrinsic(word) {
                    lexer.expect(Token::Paren('('))?;
                    let argument = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::Intrinsic {
                        fun,
                        argument,
                    })
                } else if let Some(axis) = get_derivative(word) {
                    lexer.expect(Token::Paren('('))?;
                    let expr = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::Derivative {
                        axis,
                        expr,
                    })
                } else if word == "dot" {
                    lexer.expect(Token::Paren('('))?;
                    let a = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let b = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::DotProduct(a, b))
                } else if word == "outer_product" {
                    lexer.expect(Token::Paren('('))?;
                    let a = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let b = self.parse_primary_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(')'))?;
                    Some(crate::Expression::CrossProduct(a, b))
                } else {
                    None
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
                _ => None
            },
            // relational_expression
            |lexer, mut context| context.parse_binary_op(
                lexer,
                |token| match token {
                    Token::Paren('<') => Some(crate::BinaryOperator::Less),
                    Token::Paren('>') => Some(crate::BinaryOperator::Greater),
                    Token::LogicalOperation('<') => Some(crate::BinaryOperator::LessEqual),
                    Token::LogicalOperation('>') => Some(crate::BinaryOperator::GreaterEqual),
                    _ => None,
                },
                // shift_expression
                |lexer, mut context| context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::ShiftOperation('<') => Some(crate::BinaryOperator::ShiftLeftLogical),
                        Token::ShiftOperation('>') => Some(crate::BinaryOperator::ShiftRightLogical),
                        Token::ArithmeticShiftOperation('>') => Some(crate::BinaryOperator::ShiftRightArithmetic),
                        _ => None,
                    },
                    // additive_expression
                    |lexer, mut context| context.parse_binary_op(
                        lexer,
                        |token| match token {
                            Token::Operation('+') => Some(crate::BinaryOperator::Add),
                            Token::Operation('-') => Some(crate::BinaryOperator::Subtract),
                            _ => None,
                        },
                        // multiplicative_expression
                        |lexer, mut context| context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::Operation('*') => Some(crate::BinaryOperator::Multiply),
                                Token::Operation('/') => Some(crate::BinaryOperator::Divide),
                                Token::Operation('%') => Some(crate::BinaryOperator::Modulo),
                                _ => None,
                            },
                            |lexer, context| self.parse_singular_expression(lexer, context),
                        ),
                    ),
                ),
            ),
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
            |lexer, mut context| context.parse_binary_op(
                lexer,
                |token| match token {
                    Token::LogicalOperation('&') => Some(crate::BinaryOperator::LogicalAnd),
                    _ => None,
                },
                // inclusive_or_expression
                |lexer, mut context| context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::Operation('|') => Some(crate::BinaryOperator::InclusiveOr),
                        _ => None,
                    },
                    // exclusive_or_expression
                    |lexer, mut context| context.parse_binary_op(
                        lexer,
                        |token| match token {
                            Token::Operation('^') => Some(crate::BinaryOperator::ExclusiveOr),
                            _ => None,
                        },
                        // and_expression
                        |lexer, mut context| context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::Operation('&') => Some(crate::BinaryOperator::And),
                                _ => None,
                            },
                            |lexer, context| self.parse_equality_expression(lexer, context),
                        ),
                    ),
                ),
            ),
        )?;
        self.scopes.pop();
        Ok(handle)
    }

    fn parse_variable_ident_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
    ) -> Result<(&'a str, Handle<crate::Type>), Error<'a>> {
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, type_arena)?;
        Ok((name, ty))
    }

    fn parse_variable_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<(&'a str, Option<spirv::StorageClass>, Handle<crate::Type>), Error<'a>> {
        self.scopes.push(Scope::VariableDecl);
        let mut class = None;
        if lexer.skip(Token::Paren('<')) {
            let class_str = lexer.next_ident()?;
            class = Some(Self::get_storage_class(class_str)?);
            lexer.expect(Token::Paren('>'))?;
        }
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, type_arena)?;
        if lexer.skip(Token::Operation('=')) {
            let _inner = self.parse_const_expression(lexer, type_arena, const_arena)?;
            //TODO
        }
        lexer.expect(Token::Separator(';'))?;
        self.scopes.pop();
        Ok((name, class, ty))
    }

    fn parse_struct_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
    ) -> Result<Vec<crate::StructMember>, Error<'a>> {
        let mut members = Vec::new();
        lexer.expect(Token::Paren('{'))?;
        loop {
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
                            let _offset = lexer.next_uint_literal()?; //TODO
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
            lexer.expect(Token::Separator(':'))?;
            let ty = self.parse_type_decl(lexer, type_arena)?;
            lexer.expect(Token::Separator(';'))?;
            members.push(crate::StructMember {
                name: Some(name.to_owned()),
                binding: None,
                ty,
            });
        }
    }

    fn parse_type_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
    ) -> Result<Handle<crate::Type>, Error<'a>> {
        self.scopes.push(Scope::TypeDecl);
        let inner = match lexer.next() {
            Token::Word("f32") => {
                crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Float,
                    width: 32,
                }
            }
            Token::Word("i32") => {
                crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Sint,
                    width: 32,
                }
            }
            Token::Word("u32") => {
                crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Uint,
                    width: 32,
                }
            }
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
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            Token::Word("mat3x3") => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            Token::Word("mat4x4") => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Quad,
                    kind,
                    width,
                }
            }
            Token::Word("ptr") => {
                lexer.expect(Token::Paren('<'))?;
                let class = Self::get_storage_class(lexer.next_ident()?)?;
                lexer.expect(Token::Separator(','))?;
                let base = self.parse_type_decl(lexer, type_arena)?;
                lexer.expect(Token::Paren('>'))?;
                crate::TypeInner::Pointer { base, class }
            }
            Token::Word("array") => {
                lexer.expect(Token::Paren('<'))?;
                let base = self.parse_type_decl(lexer, type_arena)?;
                let size = match lexer.next() {
                    Token::Separator(',') => {
                        let value = lexer.next_uint_literal()?;
                        lexer.expect(Token::Paren('>'))?;
                        crate::ArraySize::Static(value)
                    }
                    Token::Separator('>') => crate::ArraySize::Dynamic,
                    other => return Err(Error::Unexpected(other)),
                };
                crate::TypeInner::Array { base, size }
            }
            Token::Word("struct") => {
                let members = self.parse_struct_body(lexer, type_arena)?;
                crate::TypeInner::Struct { members }
            }
            Token::Word(name) => {
                self.scopes.pop();
                return self.lookup_type
                    .get(name)
                    .cloned()
                    .ok_or(Error::UnknownType(name));
            }
            other => return Err(Error::Unexpected(other)),
        };
        self.scopes.pop();
        Ok(Typifier::deduce_type_handle(inner, type_arena))
    }

    fn parse_statement<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, '_>,
    ) -> Result<Option<crate::Statement>, Error<'a>> {
        match lexer.next() {
            Token::Separator(';') => Ok(Some(crate::Statement::Empty)),
            Token::Paren('}') => Ok(None),
            Token::Word(word) => {
                self.scopes.push(Scope::Statement);
                let statement = match word {
                    "var" => {
                        enum Init {
                            Empty,
                            Uniform(Handle<crate::Expression>),
                            Variable(Handle<crate::Expression>),
                        }
                        let (name, ty) = self.parse_variable_ident_decl(lexer, context.types)?;
                        let init = if lexer.skip(Token::Operation('=')) {
                            let value = self.parse_general_expression(lexer, context.as_expression())?;
                            if let crate::Expression::Constant(_) = context.expressions[value] {
                                Init::Uniform(value)
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
                                Init::Uniform(value) => Some(value),
                                _ => None,
                            }
                        });
                        let expr_id = context.expressions.append(crate::Expression::LocalVariable(var_id));
                        context.lookup_ident.insert(name, expr_id);
                        match init {
                            Init::Variable(value) => crate::Statement::Store { pointer: expr_id, value },
                            _ => crate::Statement::Empty,
                        }
                    }
                    "return" => {
                        let value = if lexer.peek() != Token::Separator(';') {
                            Some(self.parse_general_expression(lexer, context.as_expression())?)
                        } else {
                            None
                        };
                        lexer.expect(Token::Separator(';'))?;
                        crate::Statement::Return {
                            value
                        }
                    }
                    "if" => {
                        lexer.expect(Token::Paren('('))?;
                        let condition = self.parse_general_expression(lexer, context.as_expression())?;
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
                            match self.parse_statement(lexer, context.reborrow())? {
                                Some(s) => body.push(s),
                                None => break,
                            }
                        }
                        crate::Statement::Loop {
                            body,
                            continuing
                        }
                    }
                    "break" => crate::Statement::Break,
                    "continue" => crate::Statement::Continue,
                    ident => {
                        // assignment
                        let var_expr = context.lookup_ident.lookup(ident)?;
                        let left = self.parse_postfix(lexer, context.as_expression(), var_expr)?;
                        lexer.expect(Token::Operation('='))?;
                        let value = self.parse_general_expression(lexer, context.as_expression())?;
                        lexer.expect(Token::Separator(';'))?;
                        crate::Statement::Store {
                            pointer: left,
                            value,
                        }
                    }
                };
                self.scopes.pop();
                Ok(Some(statement))
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
        while let Some(s) = self.parse_statement(lexer, context.reborrow())? {
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
    ) -> Result<Handle<crate::Function>, Error<'a>> {
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
        let mut parameter_types = Vec::new();
        lexer.expect(Token::Paren('('))?;
        while !lexer.skip(Token::Paren(')')) {
            if !parameter_types.is_empty() {
                lexer.expect(Token::Separator(','))?;
            }
            let (param_name, param_type) = self.parse_variable_ident_decl(lexer, &mut module.types)?;
            let param_index = parameter_types.len() as u32;
            let expression_token = expressions.append(crate::Expression::FunctionParameter(param_index));
            lookup_ident.insert(param_name, expression_token);
            parameter_types.push(param_type);
        }
        // read return type
        lexer.expect(Token::Arrow)?;
        let return_type = if lexer.skip(Token::Word("void")) {
            None
        } else {
            Some(self.parse_type_decl(lexer, &mut module.types)?)
        };
        // read body
        let mut local_variables = Arena::new();
        let mut typifier = Typifier::new();
        let body = self.parse_block(lexer, StatementContext {
            lookup_ident: &mut lookup_ident,
            typifier: &mut typifier,
            variables: &mut local_variables,
            expressions: &mut expressions,
            types: &mut module.types,
            constants: &mut module.constants,
            global_vars: &module.global_variables,
        })?;
        // done
        let global_usage = crate::GlobalUse::scan(&expressions, &body, &module.global_variables);
        self.scopes.pop();

        let fun = crate::Function {
            name: Some(fun_name.to_owned()),
            control: spirv::FunctionControl::empty(),
            parameter_types,
            return_type,
            global_usage,
            local_variables,
            expressions,
            body,
        };
        Ok(module.functions.append(fun))
    }

    fn parse_global_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        module: &mut crate::Module,
        lookup_global_expression: &mut FastHashMap<&'a str, crate::Expression>,
    ) -> Result<bool, Error<'a>> {
        // read decorations
        let mut binding = None;
        if lexer.skip(Token::DoubleParen('[')) {
            let (mut bind_index, mut bind_set) = (None, None);
            self.scopes.push(Scope::Decoration);
            loop {
                match lexer.next_ident()? {
                    "location" => {
                        let loc = lexer.next_uint_literal()?;
                        binding = Some(crate::Binding::Location(loc));
                    }
                    "builtin" => {
                        let builtin = Self::get_built_in(lexer.next_ident()?)?;
                        binding = Some(crate::Binding::BuiltIn(builtin));
                    }
                    "binding" => {
                        bind_index = Some(lexer.next_uint_literal()?);
                    }
                    "set" => {
                        bind_set = Some(lexer.next_uint_literal()?);
                    }
                    other => return Err(Error::UnknownDecoration(other)),
                }
                match lexer.next() {
                    Token::DoubleParen(']') => {
                        break;
                    }
                    Token::Separator(',') => {
                    }
                    other => return Err(Error::Unexpected(other)),
                }
            }
            match (bind_set, bind_index) {
                (Some(set), Some(index)) if binding.is_none() => {
                    binding = Some(crate::Binding::Descriptor { set, binding: index });
                }
                _ if binding.is_none() => return Err(Error::Other),
                _ => {}
            }
            self.scopes.pop();
        }
        // read items
        match lexer.next() {
            Token::Separator(';') => {},
            Token::Word("import") => {
                self.scopes.push(Scope::ImportDecl);
                let path = match lexer.next() {
                    Token::String(path) => path,
                    other => return Err(Error::Unexpected(other)),
                };
                lexer.expect(Token::Word("as"))?;
                let namespace = lexer.next_ident()?;
                lexer.expect(Token::Separator(';'))?;
                match path {
                    "GLSL.std.450" => {
                        self.std_namespace = Some(namespace.to_owned())
                    }
                    _ => return Err(Error::UnknownImport(path)),
                }
                self.scopes.pop();
            }
            Token::Word("type") => {
                let name = lexer.next_ident()?;
                lexer.expect(Token::Operation('='))?;
                let ty = self.parse_type_decl(lexer, &mut module.types)?;
                self.lookup_type.insert(name.to_owned(), ty);
                lexer.expect(Token::Separator(';'))?;
            }
            Token::Word("const") => {
                let (name, ty) = self.parse_variable_ident_decl(lexer, &mut module.types)?;
                lexer.expect(Token::Operation('='))?;
                let inner = self.parse_const_expression(lexer, &mut module.types, &mut module.constants)?;
                lexer.expect(Token::Separator(';'))?;
                let const_handle = module.constants.append(crate::Constant {
                    name: Some(name.to_owned()),
                    specialization: None,
                    inner,
                    ty,
                });
                lookup_global_expression.insert(name, crate::Expression::Constant(const_handle));
            }
            Token::Word("var") => {
                let (name, class, ty) = self.parse_variable_decl(lexer, &mut module.types, &mut module.constants)?;
                let var_handle = module.global_variables.append(crate::GlobalVariable {
                    name: Some(name.to_owned()),
                    class: match class {
                        Some(c) => c,
                        None => match binding {
                            Some(crate::Binding::BuiltIn(builtin)) => match builtin {
                                spirv::BuiltIn::GlobalInvocationId => spirv::StorageClass::Input,
                                spirv::BuiltIn::Position => spirv::StorageClass::Output,
                                _ => unimplemented!(),
                            },
                            _ => spirv::StorageClass::Private,
                        },
                    },
                    binding: binding.take(),
                    ty,
                });
                lookup_global_expression.insert(name, crate::Expression::GlobalVariable(var_handle));
            }
            Token::Word("fn") => {
                self.parse_function_decl(lexer, module, &lookup_global_expression)?;
            }
            Token::Word("entry_point") => {
                let exec_model = Self::get_execution_model(lexer.next_ident()?)?;
                let export_name = if lexer.skip(Token::Word("as")) {
                    match lexer.next() {
                        Token::String(name) => Some(name),
                        other => return Err(Error::Unexpected(other))
                    }
                } else {
                    None
                };
                lexer.expect(Token::Operation('='))?;
                let fun_ident = lexer.next_ident()?;
                lexer.expect(Token::Separator(';'))?;
                let (fun_handle, _) = module.functions
                    .iter()
                    .find(|(_, fun)| fun.name.as_ref().map(|s| s.as_str()) == Some(fun_ident))
                    .ok_or(Error::UnknownFunction(fun_ident))?;
                module.entry_points.push(crate::EntryPoint {
                    exec_model,
                    name: export_name.unwrap_or(fun_ident).to_owned(),
                    function: fun_handle,
                });
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
                    let pos = source.len() - lexer.input.len();
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
                    assert_eq!(self.scopes, Vec::new());
                    return Ok(module);
                }
            }
        }
    }
}

pub fn parse_str(source: &str) -> Result<crate::Module, ParseError> {
    Parser::new().parse(source)
}
