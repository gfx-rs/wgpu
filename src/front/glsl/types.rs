use crate::{ImageClass, ImageDimension, ScalarKind, Type, TypeInner, VectorSize};

pub fn parse_type(type_name: &str) -> Option<Type> {
    match type_name {
        "bool" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Bool,
                width: crate::BOOL_WIDTH,
            },
        }),
        "float" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Float,
                width: 4,
            },
        }),
        "double" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Float,
                width: 8,
            },
        }),
        "int" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Sint,
                width: 4,
            },
        }),
        "uint" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
        }),
        "sampler" | "samplerShadow" => Some(Type {
            name: None,
            inner: TypeInner::Sampler {
                comparison: type_name == "samplerShadow",
            },
        }),
        word => {
            fn kind_width_parse(ty: &str) -> Option<(ScalarKind, u8)> {
                Some(match ty {
                    "" => (ScalarKind::Float, 4),
                    "b" => (ScalarKind::Bool, crate::BOOL_WIDTH),
                    "i" => (ScalarKind::Sint, 4),
                    "u" => (ScalarKind::Uint, 4),
                    "d" => (ScalarKind::Float, 8),
                    _ => return None,
                })
            }

            fn size_parse(n: &str) -> Option<VectorSize> {
                Some(match n {
                    "2" => VectorSize::Bi,
                    "3" => VectorSize::Tri,
                    "4" => VectorSize::Quad,
                    _ => return None,
                })
            }

            let vec_parse = |word: &str| {
                let mut iter = word.split("vec");

                let kind = iter.next()?;
                let size = iter.next()?;
                let (kind, width) = kind_width_parse(kind)?;
                let size = size_parse(size)?;

                Some(Type {
                    name: None,
                    inner: TypeInner::Vector { size, kind, width },
                })
            };

            let mat_parse = |word: &str| {
                let mut iter = word.split("mat");

                let kind = iter.next()?;
                let size = iter.next()?;
                let (_, width) = kind_width_parse(kind)?;

                let (columns, rows) = if let Some(size) = size_parse(size) {
                    (size, size)
                } else {
                    let mut iter = size.split('x');
                    match (iter.next()?, iter.next()?, iter.next()) {
                        (col, row, None) => (size_parse(col)?, size_parse(row)?),
                        _ => return None,
                    }
                };

                Some(Type {
                    name: None,
                    inner: TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                })
            };

            let texture_parse = |word: &str| {
                let mut iter = word.split("texture");

                let texture_kind = |ty| {
                    Some(match ty {
                        "" => ScalarKind::Float,
                        "i" => ScalarKind::Sint,
                        "u" => ScalarKind::Uint,
                        _ => return None,
                    })
                };

                let kind = iter.next()?;
                let size = iter.next()?;
                let kind = texture_kind(kind)?;

                let sampled = |multi| ImageClass::Sampled { kind, multi };

                let (dim, arrayed, class) = match size {
                    "1D" => (ImageDimension::D1, false, sampled(false)),
                    "1DArray" => (ImageDimension::D1, false, sampled(false)),
                    "2D" => (ImageDimension::D2, false, sampled(false)),
                    "2DArray" => (ImageDimension::D2, false, sampled(false)),
                    "2DMS" => (ImageDimension::D2, true, sampled(true)),
                    "2DMSArray" => (ImageDimension::D2, true, sampled(true)),
                    "3D" => (ImageDimension::D3, false, sampled(false)),
                    "Cube" => (ImageDimension::Cube, false, sampled(false)),
                    "CubeArray" => (ImageDimension::D2, false, sampled(false)),
                    _ => return None,
                };

                Some(Type {
                    name: None,
                    inner: TypeInner::Image {
                        dim,
                        arrayed,
                        class,
                    },
                })
            };

            vec_parse(word)
                .or_else(|| mat_parse(word))
                .or_else(|| texture_parse(word))
        }
    }
}
