use crate::{
    storage::{Storage, Token},
    FastHashMap,
};


#[derive(Parser)]
#[grammar = "../grammars/wgsl.pest"]
struct Tokenizer;

#[derive(Debug)]
pub enum Error {
    Pest(pest::error::Error<Rule>),
    BadInt(std::num::ParseIntError),
    BadStorageClass(String),
    BadBool(String),
    UnknownType(String),
}
impl From<pest::error::Error<Rule>> for Error {
    fn from(error: pest::error::Error<Rule>) -> Self {
        Error::Pest(error)
    }
}
impl From<std::num::ParseIntError> for Error {
    fn from(error: std::num::ParseIntError) -> Self {
        Error::BadInt(error)
    }
}

pub struct Parser {
    lookup_type: FastHashMap<String, Token<crate::Type>>,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            lookup_type: FastHashMap::default(),
        }
    }

    fn parse_uint_literal(pair: pest::iterators::Pair<Rule>) -> Result<u32, Error> {
        Ok(pair.as_str().parse()?)
    }

    fn parse_int_literal(pair: pest::iterators::Pair<Rule>) -> Result<i32, Error> {
        let istr = pair.as_str();
        let (sign, istr) = match &istr[..1] {
            "_" => (-1, &istr[1..]),
            _ => (1, &istr[..]),
        };
        let integer: i32 = istr.parse()?;
        Ok(sign * integer)
    }

    fn parse_decoration_list(variable_decoration_list: pest::iterators::Pair<Rule>) -> Result<Option<crate::Binding>, Error> {
        assert_eq!(variable_decoration_list.as_rule(), Rule::variable_decoration_list);
        for variable_decoration in variable_decoration_list.into_inner() {
            assert_eq!(variable_decoration.as_rule(), Rule::variable_decoration);
            let mut inner = variable_decoration.into_inner();
            let first = inner.next().unwrap();
            match first.as_rule() {
                Rule::int_literal => {
                    let location = Self::parse_uint_literal(first)?;
                    return Ok(Some(crate::Binding::Location(location)));
                }
                unknown => panic!("Unexpected decoration: {:?}", unknown),
            }
        }
        unimplemented!()
    }

    fn parse_storage_class(storage_class: pest::iterators::Pair<Rule>) -> Result<spirv::StorageClass, Error> {
        match storage_class.as_str() {
            "in" => Ok(spirv::StorageClass::Input),
            "out" => Ok(spirv::StorageClass::Output),
            other => Err(Error::BadStorageClass(other.to_owned())),
        }
    }

    fn parse_type_decl(
        &self,
        type_decl: pest::iterators::Pair<Rule>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<Token<crate::Type>, Error> {
        assert_eq!(type_decl.as_rule(), Rule::type_decl);
        let mut type_decl_pairs = type_decl.into_inner();
        let type_kind = type_decl_pairs.next().unwrap();
        let inner = match type_kind.as_rule() {
            Rule::scalar_type => {
                let kind = match type_kind.as_str() {
                    "f32" => crate::ScalarKind::Float,
                    "i32" => crate::ScalarKind::Sint,
                    "u32" => crate::ScalarKind::Uint,
                    other => panic!("Unexpected scalar kind {:?}", other),
                };
                crate::TypeInner::Scalar { kind, width: 32 }
            }
            Rule::type_array_kind => {
                let base = self.parse_type_decl(type_decl_pairs.next().unwrap(), type_store)?;
                let size = match type_decl_pairs.next() {
                    Some(pair) => crate::ArraySize::Static(Self::parse_uint_literal(pair)?),
                    None => crate::ArraySize::Dynamic,
                };
                crate::TypeInner::Array { base, size }
            }
            Rule::type_vec_kind => {
                let size = match type_kind.as_str() {
                    "vec2" => crate::VectorSize::Bi,
                    "vec3" => crate::VectorSize::Tri,
                    "vec4" => crate::VectorSize::Quad,
                    other => panic!("Unexpected vec kind {:?}", other),
                };
                crate::TypeInner::Vector { size, kind: crate::ScalarKind::Float, width: 32 }
            }
            Rule::ident => {
                return self.lookup_type
                    .get(type_kind.as_str())
                    .cloned()
                    .ok_or(Error::UnknownType(type_kind.as_str().to_owned()));
            }
            other => panic!("Unexpected type {:?}", other),
        };
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

    fn parse_struct_decl(
        &self,
        struct_body_decl: pest::iterators::Pair<Rule>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<crate::TypeInner, Error> {
        assert_eq!(struct_body_decl.as_rule(), Rule::struct_body_decl);
        let mut members = Vec::new();
        for member_decl in struct_body_decl.into_inner() {
            assert_eq!(member_decl.as_rule(), Rule::struct_member);
            let mut member_decl_pairs = member_decl.into_inner();
            let mut body = member_decl_pairs.next().unwrap();
            let binding = None;
            if body.as_rule() == Rule::struct_member_decoration_decl {
                body = member_decl_pairs.next().unwrap(); //TODO: parse properly
            }
            assert_eq!(body.as_rule(), Rule::variable_ident_decl);
            let mut variable_ident_decl_pairs = body.into_inner();
            let member_name = variable_ident_decl_pairs.next().unwrap().as_str().to_owned();
            let ty = self.parse_type_decl(variable_ident_decl_pairs.next().unwrap(), type_store)?;
            members.push(crate::StructMember {
                name: Some(member_name),
                binding,
                ty,
            });
        }
        Ok(crate::TypeInner::Struct { members })
    }

    fn parse_const_literal(
        const_literal: pest::iterators::Pair<Rule>,
        const_store: &mut Storage<crate::Constant>,
    ) -> Result<Token<crate::Constant>, Error> {
        let inner = match const_literal.as_rule() {
            Rule::int_literal => {
                let value = Self::parse_int_literal(const_literal)?;
                crate::ConstantInner::Sint(value as i64)
            }
            Rule::bool_literal => {
                let value = match const_literal.as_str() {
                    "true" => true,
                    "false" => false,
                    other => return Err(Error::BadBool(other.to_owned())),
                };
                crate::ConstantInner::Bool(value)
            }
            ref other => panic!("Unknown const literal {:?}", other),
        };
        Ok(const_store.append(crate::Constant {
            name: None,
            specialization: None,
            inner,
        }))
    }

    fn parse_primary_expression(
        &self,
        primary_expression: pest::iterators::Pair<Rule>,
        function: &mut crate::Function,
        const_store: &mut Storage<crate::Constant>,
    ) -> Result<Token<crate::Expression>, Error> {
        let expression = match primary_expression.as_rule() {
            Rule::const_literal => {
                let const_literal = primary_expression.into_inner().next().unwrap();
                let token = Self::parse_const_literal(const_literal, const_store)?;
                crate::Expression::Constant(token)
            }
            ref other => panic!("Unknown expression {:?}", other),
        };
        Ok(function.expressions.append(expression))
    }

    fn parse_function_decl(
        &self,
        function_decl: pest::iterators::Pair<Rule>,
        module: &mut crate::Module,
    ) -> Result<Token<crate::Function>, Error> {
        enum Ident {
            Parameter(u8),
        }
        let mut lookup_symbols = FastHashMap::default();
        assert_eq!(function_decl.as_rule(), Rule::function_decl);
        let mut function_decl_pairs = function_decl.into_inner();

        let function_header = function_decl_pairs.next().unwrap();
        assert_eq!(function_header.as_rule(), Rule::function_header);
        let mut function_header_pairs = function_header.into_inner();
        let fun_name = function_header_pairs.next().unwrap().as_str().to_owned();
        let mut fun = crate::Function {
            name: Some(fun_name),
            control: spirv::FunctionControl::empty(),
            parameter_types: Vec::new(),
            return_type: None,
            expressions: Storage::new(),
            body: Vec::new(),
        };
        let param_list = function_header_pairs.next().unwrap();
        assert_eq!(param_list.as_rule(), Rule::param_list);
        for (i, variable_ident_decl) in param_list.into_inner().enumerate() {
            assert_eq!(variable_ident_decl.as_rule(), Rule::variable_ident_decl);
            let mut variable_ident_decl_pairs = variable_ident_decl.into_inner();
            let param_name = variable_ident_decl_pairs.next().unwrap().as_str().to_owned();
            lookup_symbols.insert(param_name, Ident::Parameter(i as u8));
            let param_type_decl = variable_ident_decl_pairs.next().unwrap();
            let ty = self.parse_type_decl(param_type_decl, &mut module.types)?;
            fun.parameter_types.push(ty);
        }
        let function_type_decl = function_header_pairs.next().unwrap();
        if function_type_decl.as_rule() == Rule::type_decl {
            let ty = self.parse_type_decl(function_type_decl, &mut module.types)?;
            fun.return_type = Some(ty);
        }

        let function_body = function_decl_pairs.next().unwrap();
        assert_eq!(function_body.as_rule(), Rule::body_stmt);
        for statement in function_body.into_inner() {
            assert_eq!(statement.as_rule(), Rule::statement);
            let mut statement_pairs = statement.into_inner();
            let first_statement = match statement_pairs.next() {
                Some(st) => st,
                None => continue,
            };
            let stmt = match first_statement.as_rule() {
                Rule::return_statement => {
                    let mut return_pairs = first_statement.into_inner();
                    let value = match return_pairs.next() {
                        Some(st) => Some(self.parse_primary_expression(st, &mut fun, &mut module.constants)?),
                        None => None,
                    };
                    crate::Statement::Return { value }
                }
                ref other => panic!("Unknown statement {:?}", other),
            };
            fun.body.push(stmt);
        }
        Ok(module.functions.append(fun))
    }

    pub fn parse(&mut self, source: &str) -> Result<crate::Module, Error> {
        use pest::Parser as _;
        let pairs = Tokenizer::parse(Rule::translation_unit, source)?;
        let mut module = crate::Module::generate_empty();
        for global_decl_maybe in pairs {
            match global_decl_maybe.as_rule() {
                Rule::global_decl => {
                    let global_decl = global_decl_maybe.into_inner().next().unwrap();
                    match global_decl.as_rule() {
                        Rule::import_decl => {
                            let mut import_decl = global_decl.into_inner();
                            let path = import_decl.next().unwrap().as_str();
                            log::warn!("Ignoring import {:?}", path);
                        }
                        Rule::global_variable_decl => {
                            let mut global_decl_pairs = global_decl.into_inner();
                            let binding = Self::parse_decoration_list(global_decl_pairs.next().unwrap())?;
                            let var_decl = global_decl_pairs.next().unwrap();
                            assert_eq!(var_decl.as_rule(), Rule::variable_decl);
                            let mut var_decl_pairs = var_decl.into_inner();
                            let mut body = var_decl_pairs.next().unwrap();
                            let class = if body.as_rule() == Rule::storage_class {
                                let class = Self::parse_storage_class(body)?;
                                body = var_decl_pairs.next().unwrap();
                                class
                            } else {
                                spirv::StorageClass::Private
                            };
                            assert_eq!(body.as_rule(), Rule::variable_ident_decl);
                            let mut var_ident_decl_pairs = body.into_inner();
                            let name = var_ident_decl_pairs.next().unwrap().as_str().to_owned();
                            let ty = self.parse_type_decl(var_ident_decl_pairs.next().unwrap(), &mut module.types)?;
                            module.global_variables.append(crate::GlobalVariable {
                                name: Some(name),
                                class,
                                binding,
                                ty,
                            });
                        }
                        Rule::type_alias => {
                            let mut type_alias_pairs = global_decl.into_inner();
                            let name = type_alias_pairs.next().unwrap().as_str().to_owned();
                            let something_decl = type_alias_pairs.next().unwrap();
                            match something_decl.as_rule() {
                                Rule::type_decl => {
                                    let token = self.parse_type_decl(something_decl, &mut module.types)?;
                                    self.lookup_type.insert(name, token);
                                }
                                Rule::struct_decl => {
                                    let mut struct_decl_pairs = something_decl.into_inner();
                                    let mut body = struct_decl_pairs.next().unwrap();
                                    while body.as_rule() == Rule::struct_decoration_decl {
                                        body = struct_decl_pairs.next().unwrap(); //skip
                                    }
                                    let inner = self.parse_struct_decl(body, &mut module.types)?;
                                    module.types.append(crate::Type {
                                        name: Some(name),
                                        inner,
                                    });
                                }
                                other => panic!("Unexpected type alias rule {:?}", other),
                            };
                        }
                        Rule::function_decl => {
                            self.parse_function_decl(global_decl, &mut module)?;
                        }
                        unknown => panic!("Unexpected global decl: {:?}", unknown),
                    }
                }
                Rule::EOI => break,
                unknown => panic!("Unexpected: {:?}", unknown),
            }
        }
        Ok(module)
    }
}

pub fn parse_str(source: &str) -> Result<crate::Module, Error> {
    Parser::new().parse(source)
}
