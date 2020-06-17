use super::{
    super::{Error, ErrorKind},
    Literal, Token, TokenMetadata,
};
use crate::{BinaryOperator, FastHashMap};
use std::{iter::Peekable, vec::IntoIter};

#[derive(Debug)]
enum PreprocessorIfNode {
    Literal(Literal),
    Unary {
        op: UnaryOp,
        tgt: Box<PreprocessorIfNode>,
    },
    Binary {
        left: Box<PreprocessorIfNode>,
        op: BinaryOperator,
        right: Box<PreprocessorIfNode>,
    },
}

impl Literal {
    pub fn as_isize(&self) -> isize {
        match self {
            Literal::Double(double) => *double as isize,
            Literal::Float(float) => *float as isize,
            Literal::Uint(uint) => *uint as isize,
            Literal::Sint(sint) => *sint,
            Literal::Bool(val) => *val as isize,
        }
    }

    pub fn as_bool(&self) -> Result<bool, Error> {
        Ok(match self {
            Literal::Double(_) | Literal::Float(_) => panic!(),
            Literal::Uint(uint) => {
                if *uint == 0 {
                    false
                } else {
                    true
                }
            }
            Literal::Sint(sint) => {
                if *sint == 0 {
                    false
                } else {
                    true
                }
            }
            Literal::Bool(val) => *val,
        })
    }
}

#[derive(Debug, Copy, Clone)]
pub enum UnaryOp {
    Positive,
    Negative,
    BitWiseNot,
    LogicalNot,
}

macro_rules! get_macro {
    ($name:expr, $token:expr, $line_offset:expr,$macros:expr) => {
        match $name.as_str() {
            "__LINE__" => Some(vec![TokenMetadata {
                token: Token::Integral(($token.line as i32 + $line_offset + 1) as usize),
                line: 0,
                chars: 0..1,
            }]),
            "__FILE__" => Some(vec![TokenMetadata {
                token: Token::Integral(0),
                line: 0,
                chars: 0..1,
            }]),
            "__VERSION__" => Some(vec![TokenMetadata {
                token: Token::Integral(460),
                line: 0,
                chars: 0..1,
            }]), /* TODO */
            other => $macros.get(other).cloned().map(|mut tokens| {
                let mut start = tokens[0].chars.start;
                let mut offset = 0;

                for token in tokens.iter_mut() {
                    token.line = $token.line;

                    let length = token.chars.end - token.chars.start;

                    offset += token.chars.start - start;
                    start = token.chars.start;

                    token.chars.start = $token.chars.start + offset;

                    token.chars.end = length + $token.chars.start + offset;
                }
                tokens
            }),
        }
    };
}

pub fn preprocess(
    lexer: &mut Peekable<IntoIter<TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<Vec<TokenMetadata>, Error> {
    let mut tokens = Vec::new();
    let mut line_offset = 0i32;

    let mut offset = (0, 0);

    loop {
        let token = match lexer.next() {
            Some(t) => t,
            None => break,
        };

        match token.token {
            Token::Preprocessor => {
                let preprocessor_op_token = if token.line
                    == lexer
                        .peek()
                        .ok_or(Error {
                            kind: ErrorKind::EOF,
                        })?
                        .line
                {
                    lexer.next().ok_or(Error {
                        kind: ErrorKind::EOF,
                    })?
                } else {
                    continue;
                };

                let preprocessor_op = if let Token::Word(name) = preprocessor_op_token.token {
                    name
                } else {
                    return Err(Error {
                        kind: ErrorKind::UnexpectedToken {
                            expected: vec![Token::Word(String::new())],
                            got: preprocessor_op_token,
                        },
                    });
                };

                match preprocessor_op.as_str() {
                    "define" => {
                        let macro_name_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let macro_name = if let Token::Word(name) = macro_name_token.token {
                            name
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: macro_name_token,
                                },
                            });
                        };

                        if macro_name.starts_with("GL_") {
                            return Err(Error {
                                kind: ErrorKind::ReservedMacro,
                            });
                        }

                        let mut macro_tokens = Vec::new();

                        while Some(token.line) == lexer.peek().map(|t| t.line) {
                            let macro_token = lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?;

                            match macro_token.token {
                                Token::Word(ref word) => {
                                    match get_macro!(word, &token, line_offset, macros) {
                                        Some(stream) => macro_tokens.append(&mut stream.clone()),
                                        None => macro_tokens.push(macro_token),
                                    }
                                }
                                _ => macro_tokens.push(macro_token),
                            }
                        }

                        macros.insert(macro_name, macro_tokens);
                    }
                    "undef" => {
                        let macro_name_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let macro_name = if let Token::Word(name) = macro_name_token.token {
                            name
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: macro_name_token,
                                },
                            });
                        };

                        macros.remove(&macro_name);
                    }
                    "if" => {
                        let mut expr = Vec::new();

                        while lexer
                            .peek()
                            .ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                            .line
                            == token.line
                        {
                            let expr_token = lexer.next().unwrap();

                            match expr_token.token {
                                Token::Word(ref macro_name) => expr.append(
                                    &mut get_macro!(macro_name, expr_token, line_offset, macros)
                                        .unwrap(),
                                ),
                                _ => expr.push(expr_token),
                            }
                        }

                        let condition = evaluate_preprocessor_if(expr, macros)?;

                        let mut body_tokens =
                            parse_preprocessor_if(lexer, macros, condition, line_offset, offset)?;

                        tokens.append(&mut body_tokens);
                    }
                    "ifdef" | "ifndef" => {
                        let macro_name_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().unwrap()
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let macro_name = if let Token::Word(name) = macro_name_token.token {
                            name
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: macro_name_token,
                                },
                            });
                        };

                        // There shouldn't be any more tokens on this line so we throw a error
                        if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            return Err(Error {
                                kind: ErrorKind::ExpectedEOL {
                                    got: lexer.next().unwrap(),
                                },
                            });
                        }

                        let mut body_tokens = parse_preprocessor_if(
                            lexer,
                            macros,
                            match preprocessor_op.as_str() {
                                "ifdef" => macros.get(&macro_name).is_some(),
                                "ifndef" => macros.get(&macro_name).is_none(),
                                _ => unreachable!(),
                            },
                            line_offset,
                            offset,
                        )?;

                        tokens.append(&mut body_tokens);
                    }
                    "else" | "elif" | "endif" => {
                        return Err(Error {
                            kind: ErrorKind::UnboundedIfCloserOrVariant { token },
                        })
                    }
                    "error" => {
                        let mut error_token = lexer.next();

                        let first_byte = error_token
                            .as_ref()
                            .ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                            .chars
                            .start;

                        let mut error_message = String::new();

                        while error_token.as_ref().map(|t| t.line) == Some(token.line) {
                            let error_msg_token = error_token.as_ref().unwrap();

                            let spacing = error_msg_token.chars.start
                                - first_byte
                                - error_message.chars().count();

                            error_message.push_str(&" ".repeat(spacing));
                            error_message.push_str(error_msg_token.token.to_string().as_str());

                            error_token = lexer.next()
                        }

                        panic!(error_message)
                    }
                    "pragma" => {
                        let pragma_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let pragma = if let Token::Word(name) = pragma_token.token {
                            name
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: pragma_token,
                                },
                            });
                        };

                        match pragma.as_str() {
                            "optimize" => {
                                let open_paren_token = if token.line
                                    == lexer
                                        .peek()
                                        .ok_or(Error {
                                            kind: ErrorKind::EOF,
                                        })?
                                        .line
                                {
                                    lexer.next().ok_or(Error {
                                        kind: ErrorKind::EOF,
                                    })?
                                } else {
                                    return Err(Error {
                                        kind: ErrorKind::EOL,
                                    });
                                };

                                if Token::Paren('(') != open_paren_token.token {
                                    return Err(Error {
                                        kind: ErrorKind::UnexpectedToken {
                                            expected: vec![Token::Paren('(')],
                                            got: open_paren_token,
                                        },
                                    });
                                };

                                let status_token = if token.line
                                    == lexer
                                        .peek()
                                        .ok_or(Error {
                                            kind: ErrorKind::EOF,
                                        })?
                                        .line
                                {
                                    lexer.next().ok_or(Error {
                                        kind: ErrorKind::EOF,
                                    })?
                                } else {
                                    return Err(Error {
                                        kind: ErrorKind::EOL,
                                    });
                                };

                                let _ = if let Token::Word(name) = status_token.token {
                                    name
                                } else {
                                    return Err(Error {
                                        kind: ErrorKind::UnexpectedToken {
                                            expected: vec![Token::Word(String::new())],
                                            got: status_token,
                                        },
                                    });
                                };

                                let close_paren_token = if token.line
                                    == lexer
                                        .peek()
                                        .ok_or(Error {
                                            kind: ErrorKind::EOF,
                                        })?
                                        .line
                                {
                                    lexer.next().ok_or(Error {
                                        kind: ErrorKind::EOF,
                                    })?
                                } else {
                                    return Err(Error {
                                        kind: ErrorKind::EOL,
                                    });
                                };

                                if Token::Paren(')') != close_paren_token.token {
                                    return Err(Error {
                                        kind: ErrorKind::UnexpectedToken {
                                            expected: vec![Token::Paren(')')],
                                            got: close_paren_token,
                                        },
                                    });
                                };
                            }
                            "debug" => {
                                let open_paren_token = if token.line
                                    == lexer
                                        .peek()
                                        .ok_or(Error {
                                            kind: ErrorKind::EOF,
                                        })?
                                        .line
                                {
                                    lexer.next().ok_or(Error {
                                        kind: ErrorKind::EOF,
                                    })?
                                } else {
                                    return Err(Error {
                                        kind: ErrorKind::EOL,
                                    });
                                };

                                if Token::Paren('(') != open_paren_token.token {
                                    return Err(Error {
                                        kind: ErrorKind::UnexpectedToken {
                                            expected: vec![Token::Paren('(')],
                                            got: open_paren_token,
                                        },
                                    });
                                };

                                let status_token = if token.line
                                    == lexer
                                        .peek()
                                        .ok_or(Error {
                                            kind: ErrorKind::EOF,
                                        })?
                                        .line
                                {
                                    lexer.next().ok_or(Error {
                                        kind: ErrorKind::EOF,
                                    })?
                                } else {
                                    return Err(Error {
                                        kind: ErrorKind::EOL,
                                    });
                                };

                                let _ = if let Token::Word(name) = status_token.token {
                                    name
                                } else {
                                    return Err(Error {
                                        kind: ErrorKind::UnexpectedToken {
                                            expected: vec![Token::Word(String::new())],
                                            got: status_token,
                                        },
                                    });
                                };

                                let close_paren_token = if token.line
                                    == lexer
                                        .peek()
                                        .ok_or(Error {
                                            kind: ErrorKind::EOF,
                                        })?
                                        .line
                                {
                                    lexer.next().ok_or(Error {
                                        kind: ErrorKind::EOF,
                                    })?
                                } else {
                                    return Err(Error {
                                        kind: ErrorKind::EOL,
                                    });
                                };

                                if Token::Paren(')') != close_paren_token.token {
                                    return Err(Error {
                                        kind: ErrorKind::UnexpectedToken {
                                            expected: vec![Token::Paren(')')],
                                            got: close_paren_token,
                                        },
                                    });
                                };
                            }
                            _ => {
                                return Err(Error {
                                    kind: ErrorKind::UnknownPragma { pragma },
                                })
                            }
                        }
                    }
                    "extension" => {
                        let extension_name_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let extension_name = if let Token::Word(word) = extension_name_token.token {
                            word
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: extension_name_token,
                                },
                            });
                        };

                        let separator_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        if separator_token.token != Token::DoubleColon {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::DoubleColon],
                                    got: separator_token,
                                },
                            });
                        }

                        let behavior_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let behavior = if let Token::Word(word) = behavior_token.token {
                            word
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: behavior_token,
                                },
                            });
                        };

                        match extension_name.as_str() {
                            "all" => match behavior.as_str() {
                                "require" | "enable" => {
                                    return Err(Error {
                                        kind: ErrorKind::AllExtensionsEnabled,
                                    })
                                }
                                "warn" | "disable" => {}
                                _ => {
                                    return Err(Error {
                                        kind: ErrorKind::ExtensionUnknownBehavior { behavior },
                                    })
                                }
                            },
                            _ => match behavior.as_str() {
                                "require" => {
                                    return Err(Error {
                                        kind: ErrorKind::ExtensionNotSupported {
                                            extension: extension_name,
                                        },
                                    })
                                }
                                "enable" | "warn" | "disable" => log::warn!(
                                    "Unsupported extensions was enabled: {}",
                                    extension_name
                                ),
                                _ => {
                                    return Err(Error {
                                        kind: ErrorKind::ExtensionUnknownBehavior { behavior },
                                    })
                                }
                            },
                        }
                    }
                    "version" => {
                        let version_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let version = if let Token::Integral(int) = version_token.token {
                            int
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Integral(0)],
                                    got: version_token,
                                },
                            });
                        };

                        match version {
                            450 | 460 => {}
                            _ => {
                                return Err(Error {
                                    kind: ErrorKind::UnsupportedVersion { version },
                                })
                            }
                        };

                        let profile_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let profile = if let Token::Word(word) = profile_token.token {
                            word
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: profile_token,
                                },
                            });
                        };

                        match profile.as_str() {
                            "core" => macros.insert(
                                String::from("GL_core_profile"),
                                vec![TokenMetadata {
                                    token: Token::Integral(1),
                                    line: 0,
                                    chars: 0..1,
                                }],
                            ),
                            "compatibility" | "es" => {
                                return Err(Error {
                                    kind: ErrorKind::UnsupportedProfile { profile },
                                })
                            }
                            _ => {
                                return Err(Error {
                                    kind: ErrorKind::UnknownProfile { profile },
                                })
                            }
                        };
                    }
                    "line" => {
                        let line_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        let line = if let Token::Integral(int) = line_token.token {
                            int
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Integral(0)],
                                    got: line_token,
                                },
                            });
                        };

                        let source_string_token = if token.line
                            == lexer
                                .peek()
                                .ok_or(Error {
                                    kind: ErrorKind::EOF,
                                })?
                                .line
                        {
                            lexer.next().ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                        } else {
                            return Err(Error {
                                kind: ErrorKind::EOL,
                            });
                        };

                        if let Token::Word(_) = source_string_token.token {
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: source_string_token,
                                },
                            });
                        }

                        line_offset = line as i32 - token.line as i32;
                    }
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::UnknownPreprocessorDirective {
                                directive: preprocessor_op,
                            },
                        })
                    }
                }

                if lexer.peek().map(|t| t.line) == Some(token.line) {
                    return Err(Error {
                        kind: ErrorKind::ExpectedEOL {
                            got: lexer.next().unwrap(),
                        },
                    });
                }
            }
            Token::End => {
                let mut token = token;

                if offset.0 == token.line {
                    token.chars.start = (token.chars.start as isize + offset.1) as usize;
                    token.chars.end = (token.chars.end as isize + offset.1) as usize;
                }

                tokens.push(token);
                break;
            }
            Token::Word(ref word) => match get_macro!(word, &token, line_offset, macros) {
                Some(mut stream) => {
                    for macro_token in stream.iter_mut() {
                        if offset.0 == token.line {
                            macro_token.chars.start =
                                (macro_token.chars.start as isize + offset.1) as usize;
                            macro_token.chars.end =
                                (macro_token.chars.end as isize + offset.1) as usize;
                        }
                    }

                    offset.0 = stream.last().unwrap().line;
                    offset.1 = stream.last().unwrap().chars.end as isize - token.chars.end as isize;

                    tokens.append(&mut stream)
                }
                None => {
                    let mut token = token;

                    if offset.0 == token.line {
                        token.chars.start = (token.chars.start as isize + offset.1) as usize;
                        token.chars.end = (token.chars.end as isize + offset.1) as usize;
                    }

                    tokens.push(token)
                }
            },
            _ => {
                let mut token = token;

                if offset.0 == token.line {
                    token.chars.start = (token.chars.start as isize + offset.1) as usize;
                    token.chars.end = (token.chars.end as isize + offset.1) as usize;
                }

                tokens.push(token)
            }
        }
    }

    Ok(tokens)
}

fn parse_preprocessor_if(
    lexer: &mut Peekable<IntoIter<TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
    mut condition: bool,
    line_offset: i32,
    offset: (usize, isize),
) -> Result<Vec<TokenMetadata>, Error> {
    let mut body = Vec::new();
    let mut else_block = false;

    loop {
        let macro_token = lexer.peek().ok_or(Error {
            kind: ErrorKind::EOF,
        })?;

        if let Token::Preprocessor = macro_token.token {
            let macro_token = lexer.next().unwrap();

            let directive_token = if macro_token.line
                == lexer
                    .peek()
                    .ok_or(Error {
                        kind: ErrorKind::EOF,
                    })?
                    .line
            {
                lexer.next().unwrap()
            } else {
                return Err(Error {
                    kind: ErrorKind::EOL,
                });
            };

            let directive = if let Token::Word(name) = directive_token.token {
                name
            } else {
                return Err(Error {
                    kind: ErrorKind::UnexpectedToken {
                        expected: vec![Token::Word(String::new())],
                        got: macro_token,
                    },
                });
            };

            match directive.as_str() {
                "if" => {
                    let mut expr = Vec::new();

                    while lexer
                        .peek()
                        .ok_or(Error {
                            kind: ErrorKind::EOF,
                        })?
                        .line
                        == macro_token.line
                    {
                        let expr_token = lexer.next().unwrap();

                        match expr_token.token {
                            Token::Word(ref macro_name) => expr.append(
                                &mut get_macro!(macro_name, expr_token, line_offset, macros)
                                    .unwrap(),
                            ),
                            _ => expr.push(expr_token),
                        }
                    }

                    let condition = evaluate_preprocessor_if(expr, macros)?;

                    let mut body_tokens =
                        parse_preprocessor_if(lexer, macros, condition, line_offset, offset)?;

                    body.append(&mut body_tokens);
                }
                "elif" => {
                    let mut expr = Vec::new();

                    while lexer
                        .peek()
                        .ok_or(Error {
                            kind: ErrorKind::EOF,
                        })?
                        .line
                        == macro_token.line
                    {
                        let expr_token = lexer.next().unwrap();

                        match expr_token.token {
                            Token::Word(ref macro_name) => expr.append(
                                &mut get_macro!(macro_name, expr_token, line_offset, macros)
                                    .unwrap(),
                            ),
                            _ => expr.push(expr_token),
                        }
                    }

                    if !condition {
                        condition = evaluate_preprocessor_if(expr, macros)?;
                    }
                }
                "ifdef" | "ifndef" => {
                    let macro_name_token = if macro_token.line
                        == lexer
                            .peek()
                            .ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                            .line
                    {
                        lexer.next().unwrap()
                    } else {
                        return Err(Error {
                            kind: ErrorKind::EOL,
                        });
                    };

                    let macro_name = if let Token::Word(name) = macro_name_token.token {
                        name
                    } else {
                        return Err(Error {
                            kind: ErrorKind::UnexpectedToken {
                                expected: vec![Token::Word(String::new())],
                                got: macro_name_token,
                            },
                        });
                    };

                    // There shouldn't be any more tokens on this line so we throw a error
                    if macro_token.line
                        == lexer
                            .peek()
                            .ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                            .line
                    {
                        return Err(Error {
                            kind: ErrorKind::ExpectedEOL {
                                got: lexer.next().unwrap(),
                            },
                        });
                    }

                    let mut body_tokens = parse_preprocessor_if(
                        lexer,
                        macros,
                        match directive.as_str() {
                            "ifdef" => macros.get(&macro_name).is_some(),
                            "ifndef" => macros.get(&macro_name).is_none(),
                            _ => unreachable!(),
                        },
                        line_offset,
                        offset,
                    )?;

                    body.append(&mut body_tokens);
                }
                "else" => {
                    // There shouldn't be any more tokens on this line so we throw a error
                    if directive_token.line
                        == lexer
                            .peek()
                            .ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                            .line
                    {
                        return Err(Error {
                            kind: ErrorKind::ExpectedEOL {
                                got: lexer.next().unwrap(),
                            },
                        });
                    }

                    if else_block {
                        return Err(Error {
                            kind: ErrorKind::UnexpectedWord {
                                expected: vec!["endif"],
                                got: directive,
                            },
                        });
                    }

                    else_block = true;
                    condition = !condition;
                }
                "endif" => {
                    // There shouldn't be any more tokens on this line so we throw a error
                    if directive_token.line
                        == lexer
                            .peek()
                            .ok_or(Error {
                                kind: ErrorKind::EOF,
                            })?
                            .line
                    {
                        if lexer.peek().unwrap().token != Token::End {
                            return Err(Error {
                                kind: ErrorKind::ExpectedEOL {
                                    got: lexer.next().unwrap(),
                                },
                            });
                        } else {
                            body.push(lexer.next().unwrap());
                        }
                    }

                    break;
                }
                _ => {}
            }
        }

        if condition {
            body.push(lexer.next().unwrap());
        } else {
            lexer.next().unwrap();
        }
    }

    let body_tokens = preprocess(&mut body.into_iter().peekable(), macros)?;

    Ok(body_tokens)
}

fn evaluate_preprocessor_if(
    expr: Vec<TokenMetadata>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<bool, Error> {
    let tree = logical_or_parser(&mut expr.into_iter().peekable(), macros)?;
    log::trace!("{:#?}", tree);
    evaluate_node(tree)?.as_bool()
}

fn evaluate_node(node: PreprocessorIfNode) -> Result<Literal, Error> {
    Ok(match node {
        PreprocessorIfNode::Literal(literal) => literal,
        PreprocessorIfNode::Unary { op, tgt } => {
            let literal = evaluate_node(*tgt)?;

            match op {
                UnaryOp::Positive => literal,
                UnaryOp::Negative => Literal::Sint(-literal.as_isize()),
                UnaryOp::BitWiseNot => Literal::Sint(!literal.as_isize()),
                UnaryOp::LogicalNot => Literal::Sint((!literal.as_bool()?) as isize),
            }
        }
        PreprocessorIfNode::Binary { left, op, right } => {
            let left = evaluate_node(*left)?;
            let right = evaluate_node(*right)?;

            match op {
                BinaryOperator::Multiply => Literal::Sint(left.as_isize() * right.as_isize()),
                BinaryOperator::Divide => Literal::Sint(left.as_isize() / right.as_isize()),
                BinaryOperator::Modulo => Literal::Sint(left.as_isize() % right.as_isize()),
                BinaryOperator::Add => Literal::Sint(left.as_isize() + right.as_isize()),
                BinaryOperator::Subtract => Literal::Sint(left.as_isize() - right.as_isize()),

                BinaryOperator::ShiftLeftLogical => {
                    Literal::Sint(left.as_isize() << right.as_isize())
                }
                BinaryOperator::ShiftRightArithmetic => {
                    Literal::Sint(left.as_isize() << right.as_isize())
                }

                BinaryOperator::Greater => {
                    Literal::Sint((left.as_isize() > right.as_isize()) as isize)
                }
                BinaryOperator::Less => {
                    Literal::Sint((left.as_isize() < right.as_isize()) as isize)
                }
                BinaryOperator::GreaterEqual => {
                    Literal::Sint((left.as_isize() >= right.as_isize()) as isize)
                }
                BinaryOperator::LessEqual => {
                    Literal::Sint((left.as_isize() <= right.as_isize()) as isize)
                }

                BinaryOperator::Equal => {
                    Literal::Sint((left.as_isize() == right.as_isize()) as isize)
                }
                BinaryOperator::NotEqual => {
                    Literal::Sint((left.as_isize() != right.as_isize()) as isize)
                }

                BinaryOperator::And => Literal::Sint(left.as_isize() & right.as_isize()),
                BinaryOperator::ExclusiveOr => Literal::Sint(left.as_isize() ^ right.as_isize()),
                BinaryOperator::InclusiveOr => Literal::Sint(left.as_isize() | right.as_isize()),

                BinaryOperator::LogicalOr => {
                    Literal::Sint((left.as_bool()? || right.as_bool()?) as isize)
                }
                BinaryOperator::LogicalAnd => {
                    Literal::Sint((left.as_bool()? && right.as_bool()?) as isize)
                }
                _ => unreachable!(),
            }
        }
    })
}

pub(self) fn logical_or_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = logical_and_parser(expr, macros)?;

    let mut node = left;

    while expr.peek().map(|t| &t.token) == Some(&Token::LogicalOperation('|')) {
        let _ = expr.next().unwrap();

        let right = logical_and_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op: BinaryOperator::LogicalOr,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn logical_and_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = bitwise_or_parser(expr, macros)?;

    let mut node = left;

    while expr.peek().map(|t| &t.token) == Some(&Token::LogicalOperation('&')) {
        let _ = expr.next().unwrap();

        let right = bitwise_or_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op: BinaryOperator::LogicalAnd,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn bitwise_or_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = bitwise_xor_parser(expr, macros)?;

    let mut node = left;

    while expr.peek().map(|t| &t.token) == Some(&Token::Operation('|')) {
        let _ = expr.next().unwrap();

        let right = bitwise_xor_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op: BinaryOperator::InclusiveOr,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn bitwise_xor_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = bitwise_and_parser(expr, macros)?;

    let mut node = left;

    while expr.peek().map(|t| &t.token) == Some(&Token::Operation('^')) {
        let _ = expr.next().unwrap();

        let right = bitwise_and_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op: BinaryOperator::ExclusiveOr,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn bitwise_and_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = equality_parser(expr, macros)?;

    let mut node = left;

    while expr.peek().map(|t| &t.token) == Some(&Token::Operation('&')) {
        let _ = expr.next().unwrap();

        let right = equality_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op: BinaryOperator::And,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn equality_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = relational_parser(expr, macros)?;

    let mut node = left;

    loop {
        let equality_token = match expr.peek() {
            Some(t) => t,
            None => break,
        };

        let op = match equality_token.token {
            Token::LogicalOperation('=') => BinaryOperator::Equal,
            Token::LogicalOperation('!') => BinaryOperator::NotEqual,
            _ => break,
        };

        let _ = expr.next().unwrap();

        let right = relational_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn relational_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = shift_parser(expr, macros)?;

    let mut node = left;

    loop {
        let relational_token = match expr.peek() {
            Some(t) => t,
            None => break,
        };

        let op = match relational_token.token {
            Token::LogicalOperation('<') => BinaryOperator::LessEqual,
            Token::LogicalOperation('>') => BinaryOperator::GreaterEqual,
            Token::Operation('<') => BinaryOperator::Less,
            Token::Operation('>') => BinaryOperator::Greater,
            _ => break,
        };

        let _ = expr.next().unwrap();

        let right = shift_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn shift_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = additive_parser(expr, macros)?;

    let mut node = left;

    loop {
        let shift_token = match expr.peek() {
            Some(t) => t,
            None => break,
        };

        let op = match shift_token.token {
            Token::ShiftOperation('<') => BinaryOperator::ShiftLeftLogical,
            Token::ShiftOperation('>') => BinaryOperator::ShiftRightArithmetic,
            _ => break,
        };

        let _ = expr.next().unwrap();

        let right = additive_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn additive_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = multiplicative_parser(expr, macros)?;

    let mut node = left;

    loop {
        let additive_token = match expr.peek() {
            Some(t) => t,
            None => break,
        };

        let op = match additive_token.token {
            Token::Operation('+') => BinaryOperator::Add,
            Token::Operation('-') => BinaryOperator::Subtract,
            _ => break,
        };

        let _ = expr.next().unwrap();

        let right = multiplicative_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn multiplicative_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let left = unary_parser(expr, macros)?;

    let mut node = left;

    loop {
        let multiplicative_token = match expr.peek() {
            Some(t) => t,
            None => break,
        };

        let op = match multiplicative_token.token {
            Token::Operation('*') => BinaryOperator::Multiply,
            Token::Operation('/') => BinaryOperator::Divide,
            Token::Operation('%') => BinaryOperator::Modulo,
            _ => break,
        };

        let _ = expr.next().unwrap();

        let right = unary_parser(expr, macros)?;

        node = PreprocessorIfNode::Binary {
            left: Box::new(node),
            op,
            right: Box::new(right),
        }
    }

    Ok(node)
}

fn unary_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let unary_or_atom_token = expr.peek().ok_or(Error {
        kind: ErrorKind::EOF,
    })?;

    Ok(match unary_or_atom_token.token {
        Token::Operation(op) => {
            let unary_token = expr.next().unwrap();

            PreprocessorIfNode::Unary {
                op: match op {
                    '+' => UnaryOp::Positive,
                    '-' => UnaryOp::Negative,
                    '!' => UnaryOp::BitWiseNot,
                    '~' => UnaryOp::LogicalNot,
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::UnexpectedToken {
                                expected: vec![
                                    Token::Operation('+'),
                                    Token::Operation('-'),
                                    Token::Operation('!'),
                                    Token::Operation('~'),
                                ],
                                got: unary_token,
                            },
                        })
                    }
                },
                tgt: Box::new(atom_parser(expr, macros)?),
            }
        }
        _ => atom_parser(expr, macros)?,
    })
}

fn atom_parser(
    expr: &mut Peekable<impl Iterator<Item = TokenMetadata>>,
    macros: &mut FastHashMap<String, Vec<TokenMetadata>>,
) -> Result<PreprocessorIfNode, Error> {
    let atom = expr.next().ok_or(Error {
        kind: ErrorKind::EOF,
    })?;

    Ok(match atom.token {
        Token::Double(_) | Token::Float(_) => {
            return Err(Error {
                kind: ErrorKind::NonIntegralType { token: atom },
            })
        }
        Token::Integral(int) => PreprocessorIfNode::Literal(Literal::Uint(int)),
        Token::Word(word) => PreprocessorIfNode::Literal(match word.as_str() {
            "defined" => {
                let macro_name_or_paren_token = expr.next().ok_or(Error {
                    kind: ErrorKind::EOF,
                })?;

                match macro_name_or_paren_token.token {
                    Token::Paren('(') => {
                        let macro_name_token = expr.next().ok_or(Error {
                            kind: ErrorKind::EOF,
                        })?;

                        let node = if let Token::Word(macro_name) = macro_name_token.token {
                            Literal::Sint(macros.get(&macro_name).is_some() as isize)
                        } else {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Word(String::new())],
                                    got: macro_name_token,
                                },
                            });
                        };

                        let close_paren_token = expr.next().ok_or(Error {
                            kind: ErrorKind::EOF,
                        })?;

                        if Token::Paren(')') != close_paren_token.token {
                            return Err(Error {
                                kind: ErrorKind::UnexpectedToken {
                                    expected: vec![Token::Paren(')')],
                                    got: close_paren_token,
                                },
                            });
                        }

                        node
                    }
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::UnexpectedToken {
                                expected: vec![Token::Word(String::new())],
                                got: macro_name_or_paren_token,
                            },
                        })
                    }
                }
            }
            _ => {
                return logical_or_parser(
                    &mut macros
                        .get_mut(&word)
                        .cloned()
                        .unwrap()
                        .into_iter()
                        .peekable(),
                    macros,
                )
            }
        }),
        Token::Paren('(') => {
            let node = logical_or_parser(expr, macros)?;

            let close_paren = expr.next().ok_or(Error {
                kind: ErrorKind::EOF,
            })?;

            if close_paren.token != Token::Paren(')') {
                return Err(Error {
                    kind: ErrorKind::UnexpectedToken {
                        expected: vec![Token::Paren(')')],
                        got: close_paren,
                    },
                });
            }

            node
        }
        _ => {
            return Err(Error {
                kind: ErrorKind::UnexpectedToken {
                    expected: vec![
                        Token::Word(String::new()),
                        Token::Paren('('),
                        Token::Integral(0),
                    ],
                    got: atom,
                },
            })
        }
    })
}
