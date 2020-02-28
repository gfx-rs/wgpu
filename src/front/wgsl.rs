#[derive(Parser)]
#[grammar = "../grammars/wgsl.pest"]
struct Tokenizer;

#[derive(Debug)]
pub enum Error {
    Pest(pest::error::Error<Rule>),
    BadInt(std::num::ParseIntError),
    BadStorageClass(String),
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
}

impl Parser {
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

    fn parse_type_decl(type_decl: pest::iterators::Pair<Rule>) -> Result<crate::Type, Error> {
        assert_eq!(type_decl.as_rule(), Rule::type_decl);
        let inner = type_decl.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::scalar_type => match inner.as_str() {
                "f32" => Ok(crate::Type::Scalar { kind: crate::ScalarKind::Float, width: 32 }),
                "i32" => Ok(crate::Type::Scalar { kind: crate::ScalarKind::Sint, width: 32 }),
                "u32" => Ok(crate::Type::Scalar { kind: crate::ScalarKind::Uint, width: 32 }),
                other => panic!("Unexpected scalar {:?}", other),
            },
            Rule::ident => unimplemented!(),
            other => panic!("Unexpected type {:?}", other),
        }
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
                            let ty = Self::parse_type_decl(var_ident_decl_pairs.next().unwrap())?;
                            module.global_variables.append(crate::GlobalVariable {
                                name: Some(name),
                                class,
                                binding,
                                ty,
                            });
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
    Parser{}.parse(source)
}
