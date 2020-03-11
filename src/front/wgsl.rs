use crate::{
    storage::{Storage, Token as Id},
    FastHashMap,
};

#[derive(Clone)]
struct Lexer<'a> {
    input: &'a str,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer {
            input,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Token<'a> {
    Separator(char),
    Paren(char),
    DoubleParen(char),
    Number(&'a str),
    String(&'a str),
    Word(&'a str),
    Operation(char),
    LogicalOperation(char),
    Arrow,
    Unknown(char),
    UnterminatedString,
    End,
}

impl<'a> Lexer<'a> {
    fn _consume_str(&mut self, what: &str) -> bool {
        if self.input.starts_with(what) {
            self.input = &self.input[what.len() ..];
            true
        } else {
            false
        }
    }

    fn consume_any(&mut self, what: impl Fn(char) -> bool) -> &'a str {
        let pos = self.input.find(|c| !what(c)).unwrap_or(self.input.len());
        let (left, right) = self.input.split_at(pos);
        debug_assert!(!left.is_empty(), "Leftover: '{}'...", &right[..10]);
        self.input = right;
        left
    }

    fn skip_whitespace(&mut self) {
        self.input = self.input.trim_start();
    }

    #[must_use]
    fn next(&mut self) -> Token<'a> {
        self.skip_whitespace();
        let mut chars = self.input.chars();
        let cur = match chars.next() {
            Some(c) => c,
            None => return Token::End,
        };
        let token = match cur {
            ':' | ';' | ',' | '.' => {
                self.input = chars.as_str();
                Token::Separator(cur)
            }
            '(' | ')' | '<' | '>' | '{' | '}' => {
                self.input = chars.as_str();
                Token::Paren(cur)
            }
            '[' | ']' => {
                self.input = chars.as_str();
                if chars.next() == Some(cur) {
                    self.input = chars.as_str();
                    Token::DoubleParen(cur)
                } else {
                    Token::Paren(cur)
                }
            }
            '0' ..= '9' => {
                let number = self.consume_any(|c| (c>='0' && c<='9' || c=='.'));
                Token::Number(number)
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let word = self.consume_any(|c| c.is_alphanumeric() || c=='_');
                Token::Word(word)
            }
            '"' => {
                let base = chars.as_str();
                let len = match chars.position(|c| c == '"') {
                    Some(pos) => pos,
                    None => return Token::UnterminatedString,
                };
                self.input = chars.as_str();
                Token::String(&base[..len])
            }
            '-' => {
                self.input = chars.as_str();
                if chars.next() == Some('>') {
                    self.input = chars.as_str();
                    Token::Arrow
                } else {
                    Token::Operation(cur)
                }
            }
            '+' | '*' | '/' | '^' => {
                self.input = chars.as_str();
                Token::Operation(cur)
            }
            '!' => {
                if chars.next() == Some('=') {
                    self.input = chars.as_str();
                    Token::LogicalOperation(cur)
                } else {
                    Token::Unknown(cur)
                }
            }
            '=' | '&' | '|'  => {
                if chars.next() == Some(cur) {
                    self.input = &self.input[2..];
                    Token::LogicalOperation(cur)
                } else {
                    self.input = &self.input[1..];
                    Token::Operation(cur)
                }
            }
            '#' => {
                match chars.position(|c| c == '\n' || c == '\r') {
                    Some(_) => {
                        self.input = chars.as_str();
                        self.next()
                    }
                    None => Token::End,
                }
            }
            _ => Token::Unknown(cur),
        };
        token
    }

    #[must_use]
    fn peek(&self) -> Token<'a> {
        self.clone().next()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scope {
    Decoration,
    ImportDecl,
    VariableDecl,
    TypeDecl,
    FunctionDecl,
    Statement,
    ConstantExpr,
    PrimaryExpr,
    GeneralExpr,
}

#[derive(Debug)]
pub enum Error<'a> {
    Unexpected(Token<'a>),
    BadInteger(&'a str, std::num::ParseIntError),
    BadFloat(&'a str, std::num::ParseFloatError),
    UnknownStorageClass(&'a str),
    UnknownBuiltin(&'a str),
    UnknownPipelineStage(&'a str),
    UnknownIdent(&'a str),
    UnknownFunction(&'a str),
    Other,
}

trait StringValueLookup<'a> {
    type Value;
    fn lookup(&self, key: &'a str) -> Result<Self::Value, Error<'a>>;
}
impl<'a> StringValueLookup<'a> for FastHashMap<&'a str, Id<crate::Expression>> {
    type Value = Id<crate::Expression>;
    fn lookup(&self, key: &'a str) -> Result<Self::Value, Error<'a>> {
        self.get(key)
            .cloned()
            .ok_or(Error::UnknownIdent(key))
    }
}

struct ExpressionContext<'input,'temp, 'out> {
    function: &'out mut crate::Function,
    lookup_ident: &'temp FastHashMap<&'input str, Id<crate::Expression>>,
    types: &'out mut Storage<crate::Type>,
    constants: &'out mut Storage<crate::Constant>,
}

impl<'a> ExpressionContext<'a, '_, '_> {
    fn reborrow(&mut self) -> ExpressionContext<'a, '_, '_> {
        ExpressionContext {
            function: self.function,
            lookup_ident: self.lookup_ident,
            types: self.types,
            constants: self.constants,
        }
    }

    fn parse_binary_op(
        &mut self,
        lexer: &mut Lexer<'a>,
        middle: Token<'a>,
        op: crate::BinaryOperator,
        mut parser: impl FnMut(&mut Lexer<'a>, ExpressionContext<'a, '_, '_>) -> Result<Id<crate::Expression>, Error<'a>>,
    ) -> Result<Id<crate::Expression>, Error<'a>> {
        let mut left = parser(lexer, self.reborrow())?;
        while lexer.peek() == middle {
            let _ = lexer.next();
            let expression = crate::Expression::Binary {
                op,
                left,
                right: parser(lexer, self.reborrow())?,
            };
            left = self.function.expressions.append(expression);
        }
        Ok(left)
    }
}

#[derive(Debug)]
pub struct ParseError<'a> {
    pub error: Error<'a>,
    pub scopes: Vec<Scope>,
    pub pos: (usize, usize),
}

pub struct Parser {
    scopes: Vec<Scope>,
    lookup_type: FastHashMap<String, Id<crate::Type>>,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            scopes: Vec::new(),
            lookup_type: FastHashMap::default(),
        }
    }

    fn expect<'a>(
        lexer: &mut Lexer<'a>,
        expected: Token<'a>,
    ) -> Result<(), Error<'a>> {
        let token = lexer.next();
        if token == expected {
            Ok(())
        } else {
            Err(Error::Unexpected(token))
        }
    }

    fn parse_ident<'a>(lexer: &mut Lexer<'a>) -> Result<&'a str, Error<'a>> {
        match lexer.next() {
            Token::Word(word) => Ok(word),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn _parse_float_literal<'a>(lexer: &mut Lexer<'a>) -> Result<f32, Error<'a>> {
        match lexer.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadFloat(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn parse_uint_literal<'a>(lexer: &mut Lexer<'a>) -> Result<u32, Error<'a>> {
        match lexer.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadInteger(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn _parse_sint_literal<'a>(lexer: &mut Lexer<'a>) -> Result<i32, Error<'a>> {
        match lexer.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadInteger(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn parse_scalar_generic<'a>(lexer: &mut Lexer<'a>) -> Result<(crate::ScalarKind, u8), Error<'a>> {
        Self::expect(lexer, Token::Paren('<'))?;
        let pair = match lexer.next() {
            Token::Word("f32") => (crate::ScalarKind::Float, 32),
            Token::Word("i32") => (crate::ScalarKind::Sint, 32),
            Token::Word("u32") => (crate::ScalarKind::Uint, 32),
            other => return Err(Error::Unexpected(other)),
        };
        Self::expect(lexer, Token::Paren('>'))?;
        Ok(pair)
    }

    fn get_storage_class<'a>(word: &'a str) -> Result<spirv::StorageClass, Error<'a>> {
        match word {
            "in" => Ok(spirv::StorageClass::Input),
            "out" => Ok(spirv::StorageClass::Output),
            _ => Err(Error::UnknownStorageClass(word)),
        }
    }

    fn get_built_in<'a>(word: &'a str) -> Result<spirv::BuiltIn, Error<'a>> {
        match word {
            "position" => Ok(spirv::BuiltIn::Position),
            "vertex_idx" => Ok(spirv::BuiltIn::VertexId),
            _ => Err(Error::UnknownBuiltin(word)),
        }
    }

    fn get_execution_model<'a>(word: &'a str) -> Result<spirv::ExecutionModel, Error<'a>> {
        match word {
            "vertex" => Ok(spirv::ExecutionModel::Vertex),
            "fragment" => Ok(spirv::ExecutionModel::Fragment),
            "compute" => Ok(spirv::ExecutionModel::GLCompute),
            _ => Err(Error::UnknownPipelineStage(word)),
        }
    }

    fn get_constant_inner<'a>(word: &'a str) -> Result<crate::ConstantInner, Error<'a>> {
        if word.contains('.') {
            word
                .parse()
                .map(crate::ConstantInner::Float)
                .map_err(|err| Error::BadFloat(word, err))
        } else {
            word
                .parse()
                .map(crate::ConstantInner::Sint)
                .map_err(|err| Error::BadInteger(word, err))
        }
    }

    fn parse_const_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_store: &mut Storage<crate::Type>,
        const_store: &mut Storage<crate::Constant>,
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
                Self::get_constant_inner(word)?
            }
            _ => {
                let _ty = self.parse_type_decl(lexer, type_store);
                Self::expect(lexer, Token::Paren('('))?;
                while lexer.peek() != Token::Paren(')') {
                    let _ = self.parse_const_expression(lexer, type_store, const_store)?;
                }
                let _ = lexer.next();
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
    ) -> Result<Id<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::PrimaryExpr);
        let backup = lexer.clone();
        let expression = match lexer.next() {
            Token::Paren('(') => {
                let expr = self.parse_general_expression(lexer, ctx)?;
                Self::expect(lexer, Token::Paren(')'))?;
                self.scopes.pop();
                return Ok(expr);
            }
            Token::Word("true") => {
                let id = ctx.constants.append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner: crate::ConstantInner::Bool(true),
                });
                crate::Expression::Constant(id)
            }
            Token::Word("false") => {
                let id = ctx.constants.append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner: crate::ConstantInner::Bool(false),
                });
                crate::Expression::Constant(id)
            }
            Token::Number(word) => {
                let id = ctx.constants.append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner: Self::get_constant_inner(word)?,
                });
                crate::Expression::Constant(id)
            }
            Token::Word(word) => {
                if let Some(id) = ctx.lookup_ident.get(word) {
                    self.scopes.pop();
                    return Ok(*id);
                }
                *lexer = backup;
                let ty = self.parse_type_decl(lexer, ctx.types)?;
                Self::expect(lexer, Token::Paren('('))?;
                let mut components = Vec::new();
                while lexer.peek() != Token::Paren(')') {
                    if !components.is_empty() {
                        Self::expect(lexer, Token::Separator(','))?;
                    }
                    let sub_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    components.push(sub_expr);
                }
                let _ = lexer.next();
                crate::Expression::Compose { ty, components }
            }
            other => return Err(Error::Unexpected(other)),
        };
        self.scopes.pop();
        Ok(ctx.function.expressions.append(expression))
    }

    fn parse_relational_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<Id<crate::Expression>, Error<'a>> {
        context.parse_binary_op(
            lexer,
            Token::Operation('+'),
            crate::BinaryOperator::Add,
            |lexer, mut context| context.parse_binary_op(
                lexer,
                Token::Operation('*'),
                crate::BinaryOperator::Multiply,
                |lexer, context| self.parse_primary_expression(lexer, context),
            ),
        )
    }

    fn parse_general_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<Id<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::GeneralExpr);
        let id = context.parse_binary_op(
            lexer,
            Token::LogicalOperation('|'),
            crate::BinaryOperator::LogicalOr,
            |lexer, mut context| context.parse_binary_op(
                lexer,
                Token::LogicalOperation('&'),
                crate::BinaryOperator::LogicalAnd,
                |lexer, mut context| context.parse_binary_op(
                    lexer,
                    Token::Operation('|'),
                    crate::BinaryOperator::InclusiveOr,
                    |lexer, mut context| context.parse_binary_op(
                        lexer,
                        Token::Operation('^'),
                        crate::BinaryOperator::ExclusiveOr,
                        |lexer, mut context| context.parse_binary_op(
                            lexer,
                            Token::Operation('&'),
                            crate::BinaryOperator::And,
                            |lexer, mut context| context.parse_binary_op(
                                lexer,
                                Token::LogicalOperation('='),
                                crate::BinaryOperator::Equals,
                                |lexer, context| self.parse_relational_expression(lexer, context),
                            ),
                        ),
                    ),
                ),
            ),
        )?;
        self.scopes.pop();
        Ok(id)
    }

    fn parse_variable_ident_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<(&'a str, Id<crate::Type>), Error<'a>> {
        let name = Self::parse_ident(lexer)?;
        Self::expect(lexer, Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, type_store)?;
        Ok((name, ty))
    }

    fn parse_variable_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_store: &mut Storage<crate::Type>,
        const_store: &mut Storage<crate::Constant>,
    ) -> Result<(&'a str, Option<spirv::StorageClass>, Id<crate::Type>), Error<'a>> {
        self.scopes.push(Scope::VariableDecl);
        let mut class = None;
        if let Token::Paren('<') = lexer.peek() {
            let _ = lexer.next();
            let class_str = Self::parse_ident(lexer)?;
            class = Some(Self::get_storage_class(class_str)?);
            Self::expect(lexer, Token::Paren('>'))?;
        }
        let name = Self::parse_ident(lexer)?;
        Self::expect(lexer, Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, type_store)?;
        if let Token::Operation('=') = lexer.peek() {
            let _ = lexer.next();
            let _inner = self.parse_const_expression(lexer, type_store, const_store)?;
            //TODO
        }
        Self::expect(lexer, Token::Separator(';'))?;
        self.scopes.pop();
        Ok((name, class, ty))
    }

    fn parse_struct_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<Vec<crate::StructMember>, Error<'a>> {
        let mut members = Vec::new();
        Self::expect(lexer, Token::Paren('{'))?;
        loop {
            if let Token::DoubleParen('[')  = lexer.peek() {
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
                            let _offset = Self::parse_uint_literal(lexer)?; //TODO
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
            Self::expect(lexer, Token::Separator(':'))?;
            let ty = self.parse_type_decl(lexer, type_store)?;
            Self::expect(lexer, Token::Separator(';'))?;
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
        type_store: &mut Storage<crate::Type>,
    ) -> Result<Id<crate::Type>, Error<'a>> {
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
                let (kind, width) = Self::parse_scalar_generic(lexer)?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            Token::Word("vec3") => {
                let (kind, width) = Self::parse_scalar_generic(lexer)?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            Token::Word("vec4") => {
                let (kind, width) = Self::parse_scalar_generic(lexer)?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Quad,
                    kind,
                    width,
                }
            }
            Token::Word("mat2x2") => {
                let (kind, width) = Self::parse_scalar_generic(lexer)?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            Token::Word("mat3x3") => {
                let (kind, width) = Self::parse_scalar_generic(lexer)?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            Token::Word("mat4x4") => {
                let (kind, width) = Self::parse_scalar_generic(lexer)?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Quad,
                    kind,
                    width,
                }
            }
            Token::Word("ptr") => {
                Self::expect(lexer, Token::Paren('<'))?;
                let class = Self::get_storage_class(Self::parse_ident(lexer)?)?;
                Self::expect(lexer, Token::Separator(','))?;
                let base = self.parse_type_decl(lexer, type_store)?;
                Self::expect(lexer, Token::Paren('>'))?;
                crate::TypeInner::Pointer { base, class }
            }
            Token::Word("array") => {
                Self::expect(lexer, Token::Paren('<'))?;
                let base = self.parse_type_decl(lexer, type_store)?;
                let size = match lexer.next() {
                    Token::Separator(',') => {
                        let value = Self::parse_uint_literal(lexer)?;
                        Self::expect(lexer, Token::Paren('>'))?;
                        crate::ArraySize::Static(value)
                    }
                    Token::Separator('>') => crate::ArraySize::Dynamic,
                    other => return Err(Error::Unexpected(other)),
                };
                crate::TypeInner::Array { base, size }
            }
            Token::Word("struct") => {
                let members = self.parse_struct_body(lexer, type_store)?;
                crate::TypeInner::Struct { members }
            }
            other => return Err(Error::Unexpected(other)),
        };
        self.scopes.pop();

        if let Some((token, _)) = type_store
            .iter()
            .find(|(_, ty)| ty.inner == inner)
        {
            return Ok(token);
        }
        Ok(type_store.append(crate::Type {
            name: None,
            inner,
        }))
    }

    fn parse_function_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        module: &mut crate::Module,
        lookup_global_expression: &FastHashMap<&'a str, crate::Expression>,
    ) -> Result<Id<crate::Function>, Error<'a>> {
        self.scopes.push(Scope::FunctionDecl);
        // read function name
        let mut lookup_ident = FastHashMap::default();
        let fun_name = Self::parse_ident(lexer)?;
        let mut fun = crate::Function {
            name: Some(fun_name.to_owned()),
            control: spirv::FunctionControl::empty(),
            parameter_types: Vec::new(),
            return_type: None,
            expressions: Storage::new(),
            body: Vec::new(),
        };
        // populare initial expressions
        for (&name, expression) in lookup_global_expression.iter() {
            let expr_id = fun.expressions.append(expression.clone());
            lookup_ident.insert(name, expr_id);
        }
        // read parameter list
        Self::expect(lexer, Token::Paren('('))?;
        while lexer.peek() != Token::Paren(')') {
            if !fun.parameter_types.is_empty() {
                Self::expect(lexer, Token::Separator(','))?;
            }
            let (param_name, param_type) = self.parse_variable_ident_decl(lexer, &mut module.types)?;
            let param_id = fun.parameter_types.len() as u32;
            let expression_token = fun.expressions.append(crate::Expression::FunctionParameter(param_id));
            lookup_ident.insert(param_name, expression_token);
            fun.parameter_types.push(param_type);
        }
        let _ = lexer.next();
        // read return type
        Self::expect(lexer, Token::Arrow)?;
        if let Token::Word("void") = lexer.peek() {
            let _ = lexer.next();
        } else {
            fun.return_type = Some(self.parse_type_decl(lexer, &mut module.types)?);
        };
        // read body
        Self::expect(lexer, Token::Paren('{'))?;
        loop {
            let context = ExpressionContext {
                function: &mut fun,
                lookup_ident: &lookup_ident,
                types: &mut module.types,
                constants: &mut module.constants,
            };
            match lexer.next() {
                Token::Separator(';') => {},
                Token::Paren('}') => break,
                Token::Word(word) => {
                    self.scopes.push(Scope::Statement);
                    let statement = match word {
                        "var" => {
                            let (name, ty) = self.parse_variable_ident_decl(lexer, context.types)?;
                            let value = if let Token::Operation('=') = lexer.peek() {
                                let _ = lexer.next();
                                Some(self.parse_general_expression(lexer, context)?)
                            } else {
                                None
                            };
                            Self::expect(lexer, Token::Separator(';'))?;
                            crate::Statement::VariableDeclaration {
                                name: name.to_owned(),
                                ty,
                                value,
                            }
                        }
                        "return" => {
                            let value = if context.function.return_type.is_some() {
                                Some(self.parse_general_expression(lexer, context)?)
                            } else {
                                None
                            };
                            Self::expect(lexer, Token::Separator(';'))?;
                            crate::Statement::Return {
                                value
                            }
                        }
                        ident => {
                            // assignment
                            let left = lookup_ident.lookup(ident)?;
                            Self::expect(lexer, Token::Operation('='))?;
                            let value = self.parse_general_expression(lexer, context)?;
                            Self::expect(lexer, Token::Separator(';'))?;
                            crate::Statement::Store {
                                pointer: left,
                                value,
                            }
                        }
                    };
                    self.scopes.pop();
                    fun.body.push(statement);
                }
                other => return Err(Error::Unexpected(other)),
            }
        };
        // done
        self.scopes.pop();
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
        if let Token::DoubleParen('[') = lexer.peek() {
            self.scopes.push(Scope::Decoration);
            let _ = lexer.next();
            let mut ready = true;
            loop {
                match lexer.next() {
                    Token::DoubleParen(']') => {
                        break;
                    }
                    Token::Separator(',') if !ready => {
                        ready = true;
                    }
                    Token::Word("location") if ready => {
                        let location = Self::parse_uint_literal(lexer)?;
                        binding = Some(crate::Binding::Location(location));
                        ready = false;
                    }
                    Token::Word("builtin") if ready => {
                        let builtin = Self::get_built_in(Self::parse_ident(lexer)?)?;
                        binding = Some(crate::Binding::BuiltIn(builtin));
                        ready = false;
                    }
                    other => return Err(Error::Unexpected(other)),
                }
            }
            self.scopes.pop();
        }
        // read items
        match lexer.next() {
            Token::Separator(';') => {},
            Token::Word("import") => {
                self.scopes.push(Scope::ImportDecl);
                let _path = lexer.next();
                //consume(words, "as", Scope::ImportDecl)?;
                let _namespace = lexer.next();
                Self::expect(lexer, Token::Separator(';'))?;
                self.scopes.pop();
            }
            Token::Word("type") => {
                let name = Self::parse_ident(lexer)?;
                Self::expect(lexer, Token::Operation('='))?;
                let ty = self.parse_type_decl(lexer, &mut module.types)?;
                self.lookup_type.insert(name.to_owned(), ty);
                Self::expect(lexer, Token::Separator(';'))?;
            }
            Token::Word("const") => {
                let (name, _ty) = self.parse_variable_ident_decl(lexer, &mut module.types)?;
                Self::expect(lexer, Token::Operation('='))?;
                let inner = self.parse_const_expression(lexer, &mut module.types, &mut module.constants)?;
                Self::expect(lexer, Token::Separator(';'))?;
                let const_id = module.constants.append(crate::Constant {
                    name: Some(name.to_owned()),
                    specialization: None,
                    inner,
                });
                lookup_global_expression.insert(name, crate::Expression::Constant(const_id));
            }
            Token::Word("var") => {
                let (name, class, ty) = self.parse_variable_decl(lexer, &mut module.types, &mut module.constants)?;
                let var_id = module.global_variables.append(crate::GlobalVariable {
                    name: Some(name.to_owned()),
                    class: class.unwrap_or(spirv::StorageClass::Private),
                    binding: binding.take(),
                    ty,
                });
                lookup_global_expression.insert(name, crate::Expression::GlobalVariable(var_id));
            }
            Token::Word("fn") => {
                self.parse_function_decl(lexer, module, &lookup_global_expression)?;
            }
            Token::Word("entry_point") => {
                let exec_model = Self::get_execution_model(Self::parse_ident(lexer)?)?;
                let export_name = if let Token::Word("as") = lexer.peek() {
                    let _ = lexer.next();
                    match lexer.next() {
                        Token::String(name) => Some(name),
                        other => return Err(Error::Unexpected(other))
                    }
                } else {
                    None
                };
                Self::expect(lexer, Token::Operation('='))?;
                let fun_ident = Self::parse_ident(lexer)?;
                Self::expect(lexer, Token::Separator(';'))?;
                let function = module.functions
                    .iter()
                    .find(|(_, fun)| fun.name.as_ref().map(|s| s.as_str()) == Some(fun_ident))
                    .map(|(id, _)| id)
                    .ok_or(Error::UnknownFunction(fun_ident))?;
                module.entry_points.push(crate::EntryPoint {
                    exec_model,
                    name: export_name.unwrap_or(fun_ident).to_owned(),
                    inputs: Vec::new(), //TODO
                    outputs: Vec::new(), //TODO
                    function,
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
        let mut module = crate::Module::generate_empty();
        let mut lexer = Lexer::new(source);
        let mut lookup_global_expression = FastHashMap::default();
        loop {
            match self.parse_global_decl(&mut lexer, &mut module, &mut lookup_global_expression) {
                Err(error) => {
                    let pos = source.len() - lexer.input.len();
                    let (mut rows, mut cols) = (0, 0);
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
