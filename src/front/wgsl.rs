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
    Number(&'a str),
    Word(&'a str),
    Operation { op: char, boolean: bool },
    End,
}

impl<'a> Lexer<'a> {
    fn consume_str(&mut self, what: &str) -> bool {
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

    fn next(&mut self) -> Token<'a> {
        self.skip_whitespace();
        let mut chars = self.input.chars();
        let cur = match chars.next() {
            Some(c) => c,
            None => return Token::End,
        };
        match cur {
            ':' | ';' | ',' | '.' => {
                self.input = chars.as_str();
                Token::Separator(cur)
            }
            '(' | ')' | '[' | ']' | '<' | '>' => {
                self.input = chars.as_str();
                Token::Paren(cur)
            }
            '0' ..= '9' => {
                let number = self.consume_any(|c| (c>='0' && c<='9' || c=='.'));
                Token::Number(number)
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let word = self.consume_any(|c| c.is_alphanumeric() || c=='_');
                Token::Word(word)
            }
            '=' | '+'| '-' | '*' | '/' | '&' | '|' | '^' => {
                let next = chars.next();
                let boolean = (cur == '=' || cur == '&' || cur == '|') && next == Some(cur);
                self.input = &self.input[if boolean { 2 } else { 1 } ..];
                Token::Operation { op: cur, boolean }
            }
            '#' => {
                match chars.position(|c| c == '\n' || c == '\r') {
                    Some(_) => {
                        self.input = chars.as_str();
                        self.next()
                    }
                    None => return Token::End,
                }
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scope {
    Decoration,
    ImportDecl,
    VariableDecl,
    TypeDecl,
    ConstantExpr,
}

#[derive(Debug)]
pub enum Error<'a> {
    Unexpected(Token<'a>),
    BadInteger(&'a str, std::num::ParseIntError),
    BadFloat(&'a str, std::num::ParseFloatError),
    UnknownStorageClass(&'a str),
    UnknownBuiltin(&'a str),
    Other,
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

    fn parse_ident<'a>(&mut self, lexer: &mut Lexer<'a>) -> Result<&'a str, Error<'a>> {
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

    fn parse_const_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_store: &mut Storage<crate::Type>,
        const_store: &mut Storage<crate::Constant>,
    ) -> Result<crate::ConstantInner, Error<'a>> {
        self.scopes.push(Scope::ConstantExpr);
        let backup = lexer.clone();
        let inner = match lexer.next() {
            Token::Word("true") => crate::ConstantInner::Bool(true),
            Token::Word("false") => crate::ConstantInner::Bool(false),
            Token::Number(word) => {
                if word.contains('.') {
                    let value = word
                        .parse()
                        .map_err(|err| Error::BadFloat(word, err))?;
                    crate::ConstantInner::Float(value)
                } else {
                    let value = word
                        .parse()
                        .map_err(|err| Error::BadInteger(word, err))?;
                    crate::ConstantInner::Sint(value)
                }
            }
            _ => {
                *lexer = backup;
                let _ty = self.parse_type_decl(lexer, type_store);
                Self::expect(lexer, Token::Paren('('))?;
                loop {
                    let backup = lexer.clone();
                    match lexer.next() {
                        Token::Paren(')') => break,
                        _ => {
                            *lexer = backup;
                            let _ = self.parse_const_expression(lexer, type_store, const_store)?;
                        }
                    }
                }
                unimplemented!()
            }
        };
        self.scopes.pop();
        Ok(inner)
    }

    fn parse_variable_ident_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<(&'a str, Id<crate::Type>), Error<'a>> {
        let name = match lexer.next() {
            Token::Word(word) => word,
            other => return Err(Error::Unexpected(other)),
        };
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
        let mut token = lexer.next();
        if let Token::Paren('<') = token {
            class = Some(match lexer.next() {
                Token::Word(word) => Self::get_storage_class(word)?,
                other => return Err(Error::Unexpected(other)),
            });
            Self::expect(lexer, Token::Paren('>'))?;
            token = lexer.next();
        }
        let name = match token {
            Token::Word(word) => word,
            other => return Err(Error::Unexpected(other)),
        };
        Self::expect(lexer, Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, type_store)?;
        match lexer.next() {
            Token::Separator(';') => {}
            Token::Operation{ op: '=', boolean: false } => {
                let _inner = self.parse_const_expression(lexer, type_store, const_store)?;
                //TODO
                Self::expect(lexer, Token::Separator(';'))?;
            }
            other => return Err(Error::Unexpected(other)),
        }
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
            let mut token = lexer.next();
            if let Token::Paren('[')  = token {
                self.scopes.push(Scope::Decoration);
                if !lexer.consume_str("[") {
                    return Err(Error::Other);
                }
                let mut ready = true;
                loop {
                    match lexer.next() {
                        Token::Paren(']') => {
                            if !lexer.consume_str("]") {
                                return Err(Error::Other);
                            }
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
                token = lexer.next();
                self.scopes.pop();
            }
            let name = match token {
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
                let class = match lexer.next() {
                    Token::Word(word) => Self::get_storage_class(word)?,
                    other => return Err(Error::Unexpected(other)),
                };
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

    fn parse_global_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        module: &mut crate::Module,
    ) -> Result<bool, Error<'a>> {
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
                let name = self.parse_ident(lexer)?;
                Self::expect(lexer, Token::Operation { op: '=', boolean: false })?;
                let ty = self.parse_type_decl(lexer, &mut module.types)?;
                self.lookup_type.insert(name.to_owned(), ty);
                Self::expect(lexer, Token::Separator(';'))?;
            }
            Token::Word("const") => {
                let (name, _ty) = self.parse_variable_ident_decl(lexer, &mut module.types)?;
                Self::expect(lexer, Token::Operation { op: '=', boolean: false })?;
                let inner = self.parse_const_expression(lexer, &mut module.types, &mut module.constants)?;
                module.constants.append(crate::Constant {
                    name: Some(name.to_owned()),
                    specialization: None,
                    inner,
                });
                Self::expect(lexer, Token::Separator(';'))?;
            }
            Token::Paren('[') => {
                self.scopes.push(Scope::Decoration);
                if !lexer.consume_str("[") {
                    return Err(Error::Other);
                }
                let mut ready = true;
                let mut binding = None;
                loop {
                    match lexer.next() {
                        Token::Paren(']') => {
                            if !lexer.consume_str("]") {
                                return Err(Error::Other);
                            }
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
                            let builtin = match lexer.next() {
                                Token::Word(word) => Self::get_built_in(word)?,
                                other => return Err(Error::Unexpected(other)),
                            };
                            binding = Some(crate::Binding::BuiltIn(builtin));
                            ready = false;
                        }
                        other => return Err(Error::Unexpected(other)),
                    }
                }
                self.scopes.pop();
                Self::expect(lexer, Token::Word("var"))?;
                let (name, class, ty) = self.parse_variable_decl(lexer, &mut module.types, &mut module.constants)?;
                module.global_variables.append(crate::GlobalVariable {
                    name: Some(name.to_owned()),
                    class: class.unwrap_or(spirv::StorageClass::Private),
                    binding,
                    ty,
                });
            }
            Token::Word("var") => {
                let (name, class, ty) = self.parse_variable_decl(lexer, &mut module.types, &mut module.constants)?;
                module.global_variables.append(crate::GlobalVariable {
                    name: Some(name.to_owned()),
                    class: class.unwrap_or(spirv::StorageClass::Private),
                    binding: None,
                    ty,
                });
            }
            Token::Word("fn") => {
                unimplemented!()
            }
            Token::Word("entry_point") => {
                unimplemented!()
            }
            Token::End => return Ok(false),
            token => return Err(Error::Unexpected(token)),
        }
        Ok(true)
    }

    pub fn parse<'a>(&mut self, source: &'a str) -> Result<crate::Module, ParseError<'a>> {
        let mut module = crate::Module::generate_empty();
        let mut lexer = Lexer::new(source);
        loop {
            match self.parse_global_decl(&mut lexer, &mut module) {
                Err(error) => {
                    let pos = source.rfind(lexer.input).unwrap();
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
