use crate::{
    front::glsl::{
        ast::{QualifierKey, QualifierValue, StorageQualifier, StructLayout, TypeQualifiers},
        error::ExpectedToken,
        parser::ParsingContext,
        token::{Token, TokenValue},
        Error, ErrorKind, Parser, Result,
    },
    AddressSpace, ArraySize, Handle, Span, Type, TypeInner,
};

impl<'source> ParsingContext<'source> {
    /// Parses an optional array_specifier returning wether or not it's present
    /// and modifying the type handle if it exists
    pub fn parse_array_specifier(
        &mut self,
        parser: &mut Parser,
        span: &mut Span,
        ty: &mut Handle<Type>,
    ) -> Result<()> {
        while self.parse_array_specifier_single(parser, span, ty)? {}
        Ok(())
    }

    /// Implementation of [`Self::parse_array_specifier`] for a single array_specifier
    fn parse_array_specifier_single(
        &mut self,
        parser: &mut Parser,
        span: &mut Span,
        ty: &mut Handle<Type>,
    ) -> Result<bool> {
        if self.bump_if(parser, TokenValue::LeftBracket).is_some() {
            let size =
                if let Some(Token { meta, .. }) = self.bump_if(parser, TokenValue::RightBracket) {
                    span.subsume(meta);
                    ArraySize::Dynamic
                } else {
                    let (value, constant_span) = self.parse_uint_constant(parser)?;
                    let constant = parser.module.constants.fetch_or_append(
                        crate::Constant {
                            name: None,
                            specialization: None,
                            inner: crate::ConstantInner::Scalar {
                                width: 4,
                                value: crate::ScalarValue::Uint(value as u64),
                            },
                        },
                        constant_span,
                    );
                    let end_span = self.expect(parser, TokenValue::RightBracket)?.meta;
                    span.subsume(end_span);
                    ArraySize::Constant(constant)
                };

            parser
                .layouter
                .update(&parser.module.types, &parser.module.constants)
                .unwrap();
            let stride = parser.layouter[*ty].to_stride();
            *ty = parser.module.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Array {
                        base: *ty,
                        size,
                        stride,
                    },
                },
                *span,
            );

            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn parse_type(&mut self, parser: &mut Parser) -> Result<(Option<Handle<Type>>, Span)> {
        let token = self.bump(parser)?;
        let mut handle = match token.value {
            TokenValue::Void => return Ok((None, token.meta)),
            TokenValue::TypeName(ty) => parser.module.types.insert(ty, token.meta),
            TokenValue::Struct => {
                let mut meta = token.meta;
                let ty_name = self.expect_ident(parser)?.0;
                self.expect(parser, TokenValue::LeftBrace)?;
                let mut members = Vec::new();
                let span =
                    self.parse_struct_declaration_list(parser, &mut members, StructLayout::Std140)?;
                let end_meta = self.expect(parser, TokenValue::RightBrace)?.meta;
                meta.subsume(end_meta);
                let ty = parser.module.types.insert(
                    Type {
                        name: Some(ty_name.clone()),
                        inner: TypeInner::Struct { members, span },
                    },
                    meta,
                );
                parser.lookup_type.insert(ty_name, ty);
                ty
            }
            TokenValue::Identifier(ident) => match parser.lookup_type.get(&ident) {
                Some(ty) => *ty,
                None => {
                    return Err(Error {
                        kind: ErrorKind::UnknownType(ident),
                        meta: token.meta,
                    })
                }
            },
            _ => {
                return Err(Error {
                    kind: ErrorKind::InvalidToken(
                        token.value,
                        vec![
                            TokenValue::Void.into(),
                            TokenValue::Struct.into(),
                            ExpectedToken::TypeName,
                        ],
                    ),
                    meta: token.meta,
                });
            }
        };

        let mut span = token.meta;
        self.parse_array_specifier(parser, &mut span, &mut handle)?;
        Ok((Some(handle), span))
    }

    pub fn parse_type_non_void(&mut self, parser: &mut Parser) -> Result<(Handle<Type>, Span)> {
        let (maybe_ty, meta) = self.parse_type(parser)?;
        let ty = maybe_ty.ok_or_else(|| Error {
            kind: ErrorKind::SemanticError("Type can't be void".into()),
            meta,
        })?;

        Ok((ty, meta))
    }

    pub fn peek_type_qualifier(&mut self, parser: &mut Parser) -> bool {
        self.peek(parser).map_or(false, |t| match t.value {
            TokenValue::Invariant
            | TokenValue::Interpolation(_)
            | TokenValue::Sampling(_)
            | TokenValue::PrecisionQualifier(_)
            | TokenValue::Const
            | TokenValue::In
            | TokenValue::Out
            | TokenValue::Uniform
            | TokenValue::Shared
            | TokenValue::Buffer
            | TokenValue::Restrict
            | TokenValue::MemoryQualifier(_)
            | TokenValue::Layout => true,
            _ => false,
        })
    }

    pub fn parse_type_qualifiers<'a>(&mut self, parser: &mut Parser) -> Result<TypeQualifiers<'a>> {
        let mut qualifiers = TypeQualifiers::default();

        while self.peek_type_qualifier(parser) {
            let token = self.bump(parser)?;

            // Handle layout qualifiers outside the match since this can push multiple values
            if token.value == TokenValue::Layout {
                self.parse_layout_qualifier_id_list(parser, &mut qualifiers)?;
                continue;
            }

            qualifiers.span.subsume(token.meta);

            match token.value {
                TokenValue::Invariant => {
                    if qualifiers.invariant.is_some() {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Cannot use more than one invariant qualifier per declaration"
                                    .into(),
                            ),
                            meta: token.meta,
                        })
                    }

                    qualifiers.invariant = Some(token.meta);
                }
                TokenValue::Interpolation(i) => {
                    if qualifiers.interpolation.is_some() {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Cannot use more than one interpolation qualifier per declaration"
                                    .into(),
                            ),
                            meta: token.meta,
                        })
                    }

                    qualifiers.interpolation = Some((i, token.meta));
                }
                TokenValue::Const
                | TokenValue::In
                | TokenValue::Out
                | TokenValue::Uniform
                | TokenValue::Shared
                | TokenValue::Buffer => {
                    let storage = match token.value {
                        TokenValue::Const => StorageQualifier::Const,
                        TokenValue::In => StorageQualifier::Input,
                        TokenValue::Out => StorageQualifier::Output,
                        TokenValue::Uniform => {
                            StorageQualifier::AddressSpace(AddressSpace::Uniform)
                        }
                        TokenValue::Shared => {
                            StorageQualifier::AddressSpace(AddressSpace::WorkGroup)
                        }
                        TokenValue::Buffer => {
                            StorageQualifier::AddressSpace(AddressSpace::Storage {
                                access: crate::StorageAccess::all(),
                            })
                        }
                        _ => unreachable!(),
                    };

                    if StorageQualifier::AddressSpace(AddressSpace::Function)
                        != qualifiers.storage.0
                    {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Cannot use more than one storage qualifier per declaration".into(),
                            ),
                            meta: token.meta,
                        });
                    }

                    qualifiers.storage = (storage, token.meta);
                }
                TokenValue::Sampling(s) => {
                    if qualifiers.sampling.is_some() {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Cannot use more than one sampling qualifier per declaration"
                                    .into(),
                            ),
                            meta: token.meta,
                        })
                    }

                    qualifiers.sampling = Some((s, token.meta));
                }
                TokenValue::PrecisionQualifier(p) => {
                    if qualifiers.precision.is_some() {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Cannot use more than one precision qualifier per declaration"
                                    .into(),
                            ),
                            meta: token.meta,
                        })
                    }

                    qualifiers.precision = Some((p, token.meta));
                }
                TokenValue::MemoryQualifier(access) => {
                    let storage_access = qualifiers
                        .storage_access
                        .get_or_insert((crate::StorageAccess::all(), Span::default()));
                    if !storage_access.0.contains(!access) {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "The same memory qualifier can only be used once".into(),
                            ),
                            meta: token.meta,
                        })
                    }

                    storage_access.0 &= access;
                    storage_access.1.subsume(token.meta);
                }
                TokenValue::Restrict => continue,
                _ => unreachable!(),
            };
        }

        Ok(qualifiers)
    }

    pub fn parse_layout_qualifier_id_list(
        &mut self,
        parser: &mut Parser,
        qualifiers: &mut TypeQualifiers,
    ) -> Result<()> {
        self.expect(parser, TokenValue::LeftParen)?;
        loop {
            self.parse_layout_qualifier_id(parser, &mut qualifiers.layout_qualifiers)?;

            if self.bump_if(parser, TokenValue::Comma).is_some() {
                continue;
            }

            break;
        }
        let token = self.expect(parser, TokenValue::RightParen)?;
        qualifiers.span.subsume(token.meta);

        Ok(())
    }

    pub fn parse_layout_qualifier_id(
        &mut self,
        parser: &mut Parser,
        qualifiers: &mut crate::FastHashMap<QualifierKey, (QualifierValue, Span)>,
    ) -> Result<()> {
        // layout_qualifier_id:
        //     IDENTIFIER
        //     IDENTIFIER EQUAL constant_expression
        //     SHARED
        let mut token = self.bump(parser)?;
        match token.value {
            TokenValue::Identifier(name) => {
                let (key, value) = match name.as_str() {
                    "std140" => (
                        QualifierKey::Layout,
                        QualifierValue::Layout(StructLayout::Std140),
                    ),
                    "std430" => (
                        QualifierKey::Layout,
                        QualifierValue::Layout(StructLayout::Std430),
                    ),
                    word => {
                        if let Some(format) = map_image_format(word) {
                            (QualifierKey::Format, QualifierValue::Format(format))
                        } else {
                            let key = QualifierKey::String(name.into());
                            let value = if self.bump_if(parser, TokenValue::Assign).is_some() {
                                let (value, end_meta) = match self.parse_uint_constant(parser) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        parser.errors.push(e);
                                        (0, Span::default())
                                    }
                                };
                                token.meta.subsume(end_meta);

                                QualifierValue::Uint(value)
                            } else {
                                QualifierValue::None
                            };

                            (key, value)
                        }
                    }
                };

                qualifiers.insert(key, (value, token.meta));
            }
            _ => parser.errors.push(Error {
                kind: ErrorKind::InvalidToken(token.value, vec![ExpectedToken::Identifier]),
                meta: token.meta,
            }),
        }

        Ok(())
    }

    pub fn peek_type_name(&mut self, parser: &mut Parser) -> bool {
        self.peek(parser).map_or(false, |t| match t.value {
            TokenValue::TypeName(_) | TokenValue::Void => true,
            TokenValue::Struct => true,
            TokenValue::Identifier(ref ident) => parser.lookup_type.contains_key(ident),
            _ => false,
        })
    }
}

fn map_image_format(word: &str) -> Option<crate::StorageFormat> {
    use crate::StorageFormat as Sf;

    let format = match word {
        // float-image-format-qualifier:
        "rgba32f" => Sf::Rgba32Float,
        "rgba16f" => Sf::Rgba16Float,
        "rg32f" => Sf::Rg32Float,
        "rg16f" => Sf::Rg16Float,
        "r11f_g11f_b10f" => Sf::Rg11b10Float,
        "r32f" => Sf::R32Float,
        "r16f" => Sf::R16Float,
        "rgba16" => Sf::Rgba16Float,
        "rgb10_a2" => Sf::Rgb10a2Unorm,
        "rgba8" => Sf::Rgba8Unorm,
        "rg16" => Sf::Rg16Float,
        "rg8" => Sf::Rg8Unorm,
        "r16" => Sf::R16Float,
        "r8" => Sf::R8Unorm,
        "rgba8_snorm" => Sf::Rgba8Snorm,
        "rg8_snorm" => Sf::Rg8Snorm,
        "r8_snorm" => Sf::R8Snorm,
        // int-image-format-qualifier:
        "rgba32i" => Sf::Rgba32Sint,
        "rgba16i" => Sf::Rgba16Sint,
        "rgba8i" => Sf::Rgba8Sint,
        "rg32i" => Sf::Rg32Sint,
        "rg16i" => Sf::Rg16Sint,
        "rg8i" => Sf::Rg8Sint,
        "r32i" => Sf::R32Sint,
        "r16i" => Sf::R16Sint,
        "r8i" => Sf::R8Sint,
        // uint-image-format-qualifier:
        "rgba32ui" => Sf::Rgba32Uint,
        "rgba16ui" => Sf::Rgba16Uint,
        "rgba8ui" => Sf::Rgba8Uint,
        "rg32ui" => Sf::Rg32Uint,
        "rg16ui" => Sf::Rg16Uint,
        "rg8ui" => Sf::Rg8Uint,
        "r32ui" => Sf::R32Uint,
        "r16ui" => Sf::R16Uint,
        "r8ui" => Sf::R8Uint,
        // TODO: These next ones seem incorrect to me
        // "rgba16_snorm" => Sf::Rgba16Float,
        // "rg16_snorm" => Sf::Rg16Float,
        // "r16_snorm" => Sf::R16Float,
        // "rgb10_a2ui" => Sf::Rgb10a2Unorm,
        _ => return None,
    };

    Some(format)
}
