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

    fn _parse_int_literal(pair: pest::iterators::Pair<Rule>) -> Result<i32, Error> {
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
