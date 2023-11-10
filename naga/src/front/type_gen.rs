/*!
Type generators.
*/

use crate::{arena::Handle, span::Span};

impl crate::Module {
    /// Populate this module's [`SpecialTypes::ray_desc`] type.
    ///
    /// [`SpecialTypes::ray_desc`] is the type of the [`descriptor`] operand of
    /// an [`Initialize`] [`RayQuery`] statement. In WGSL, it is a struct type
    /// referred to as `RayDesc`.
    ///
    /// Backends consume values of this type to drive platform APIs, so if you
    /// change any its fields, you must update the backends to match. Look for
    /// backend code dealing with [`RayQueryFunction::Initialize`].
    ///
    /// [`SpecialTypes::ray_desc`]: crate::SpecialTypes::ray_desc
    /// [`descriptor`]: crate::RayQueryFunction::Initialize::descriptor
    /// [`Initialize`]: crate::RayQueryFunction::Initialize
    /// [`RayQuery`]: crate::Statement::RayQuery
    /// [`RayQueryFunction::Initialize`]: crate::RayQueryFunction::Initialize
    pub fn generate_ray_desc_type(&mut self) -> Handle<crate::Type> {
        if let Some(handle) = self.special_types.ray_desc {
            return handle;
        }

        let ty_flag = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar(crate::Scalar::U32),
            },
            Span::UNDEFINED,
        );
        let ty_scalar = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar(crate::Scalar::F32),
            },
            Span::UNDEFINED,
        );
        let ty_vector = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Vector {
                    size: crate::VectorSize::Tri,
                    scalar: crate::Scalar::F32,
                },
            },
            Span::UNDEFINED,
        );

        let handle = self.types.insert(
            crate::Type {
                name: Some("RayDesc".to_string()),
                inner: crate::TypeInner::Struct {
                    members: vec![
                        crate::StructMember {
                            name: Some("flags".to_string()),
                            ty: ty_flag,
                            binding: None,
                            offset: 0,
                        },
                        crate::StructMember {
                            name: Some("cull_mask".to_string()),
                            ty: ty_flag,
                            binding: None,
                            offset: 4,
                        },
                        crate::StructMember {
                            name: Some("tmin".to_string()),
                            ty: ty_scalar,
                            binding: None,
                            offset: 8,
                        },
                        crate::StructMember {
                            name: Some("tmax".to_string()),
                            ty: ty_scalar,
                            binding: None,
                            offset: 12,
                        },
                        crate::StructMember {
                            name: Some("origin".to_string()),
                            ty: ty_vector,
                            binding: None,
                            offset: 16,
                        },
                        crate::StructMember {
                            name: Some("dir".to_string()),
                            ty: ty_vector,
                            binding: None,
                            offset: 32,
                        },
                    ],
                    span: 48,
                },
            },
            Span::UNDEFINED,
        );

        self.special_types.ray_desc = Some(handle);
        handle
    }

    /// Populate this module's [`SpecialTypes::ray_intersection`] type.
    ///
    /// [`SpecialTypes::ray_intersection`] is the type of a
    /// `RayQueryGetIntersection` expression. In WGSL, it is a struct type
    /// referred to as `RayIntersection`.
    ///
    /// Backends construct values of this type based on platform APIs, so if you
    /// change any its fields, you must update the backends to match. Look for
    /// the backend's handling for [`Expression::RayQueryGetIntersection`].
    ///
    /// [`SpecialTypes::ray_intersection`]: crate::SpecialTypes::ray_intersection
    /// [`Expression::RayQueryGetIntersection`]: crate::Expression::RayQueryGetIntersection
    pub fn generate_ray_intersection_type(&mut self) -> Handle<crate::Type> {
        if let Some(handle) = self.special_types.ray_intersection {
            return handle;
        }

        let ty_flag = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar(crate::Scalar::U32),
            },
            Span::UNDEFINED,
        );
        let ty_scalar = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar(crate::Scalar::F32),
            },
            Span::UNDEFINED,
        );
        let ty_barycentrics = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Vector {
                    size: crate::VectorSize::Bi,
                    scalar: crate::Scalar::F32,
                },
            },
            Span::UNDEFINED,
        );
        let ty_bool = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar(crate::Scalar::BOOL),
            },
            Span::UNDEFINED,
        );
        let ty_transform = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Tri,
                    width: 4,
                },
            },
            Span::UNDEFINED,
        );

        let handle = self.types.insert(
            crate::Type {
                name: Some("RayIntersection".to_string()),
                inner: crate::TypeInner::Struct {
                    members: vec![
                        crate::StructMember {
                            name: Some("kind".to_string()),
                            ty: ty_flag,
                            binding: None,
                            offset: 0,
                        },
                        crate::StructMember {
                            name: Some("t".to_string()),
                            ty: ty_scalar,
                            binding: None,
                            offset: 4,
                        },
                        crate::StructMember {
                            name: Some("instance_custom_index".to_string()),
                            ty: ty_flag,
                            binding: None,
                            offset: 8,
                        },
                        crate::StructMember {
                            name: Some("instance_id".to_string()),
                            ty: ty_flag,
                            binding: None,
                            offset: 12,
                        },
                        crate::StructMember {
                            name: Some("sbt_record_offset".to_string()),
                            ty: ty_flag,
                            binding: None,
                            offset: 16,
                        },
                        crate::StructMember {
                            name: Some("geometry_index".to_string()),
                            ty: ty_flag,
                            binding: None,
                            offset: 20,
                        },
                        crate::StructMember {
                            name: Some("primitive_index".to_string()),
                            ty: ty_flag,
                            binding: None,
                            offset: 24,
                        },
                        crate::StructMember {
                            name: Some("barycentrics".to_string()),
                            ty: ty_barycentrics,
                            binding: None,
                            offset: 28,
                        },
                        crate::StructMember {
                            name: Some("front_face".to_string()),
                            ty: ty_bool,
                            binding: None,
                            offset: 36,
                        },
                        crate::StructMember {
                            name: Some("object_to_world".to_string()),
                            ty: ty_transform,
                            binding: None,
                            offset: 48,
                        },
                        crate::StructMember {
                            name: Some("world_to_object".to_string()),
                            ty: ty_transform,
                            binding: None,
                            offset: 112,
                        },
                    ],
                    span: 176,
                },
            },
            Span::UNDEFINED,
        );

        self.special_types.ray_intersection = Some(handle);
        handle
    }

    /// Populate this module's [`SpecialTypes::predeclared_types`] type and return the handle.
    ///
    /// [`SpecialTypes::predeclared_types`]: crate::SpecialTypes::predeclared_types
    pub fn generate_predeclared_type(
        &mut self,
        special_type: crate::PredeclaredType,
    ) -> Handle<crate::Type> {
        use std::fmt::Write;

        if let Some(value) = self.special_types.predeclared_types.get(&special_type) {
            return *value;
        }

        let ty = match special_type {
            crate::PredeclaredType::AtomicCompareExchangeWeakResult(scalar) => {
                let bool_ty = self.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar(crate::Scalar::BOOL),
                    },
                    Span::UNDEFINED,
                );
                let scalar_ty = self.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar(scalar),
                    },
                    Span::UNDEFINED,
                );

                crate::Type {
                    name: Some(format!(
                        "__atomic_compare_exchange_result<{:?},{}>",
                        scalar.kind, scalar.width,
                    )),
                    inner: crate::TypeInner::Struct {
                        members: vec![
                            crate::StructMember {
                                name: Some("old_value".to_string()),
                                ty: scalar_ty,
                                binding: None,
                                offset: 0,
                            },
                            crate::StructMember {
                                name: Some("exchanged".to_string()),
                                ty: bool_ty,
                                binding: None,
                                offset: 4,
                            },
                        ],
                        span: 8,
                    },
                }
            }
            crate::PredeclaredType::ModfResult { size, width } => {
                let float_ty = self.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar(crate::Scalar::float(width)),
                    },
                    Span::UNDEFINED,
                );

                let (member_ty, second_offset) = if let Some(size) = size {
                    let vec_ty = self.types.insert(
                        crate::Type {
                            name: None,
                            inner: crate::TypeInner::Vector {
                                size,
                                scalar: crate::Scalar::float(width),
                            },
                        },
                        Span::UNDEFINED,
                    );
                    (vec_ty, size as u32 * width as u32)
                } else {
                    (float_ty, width as u32)
                };

                let mut type_name = "__modf_result_".to_string();
                if let Some(size) = size {
                    let _ = write!(type_name, "vec{}_", size as u8);
                }
                let _ = write!(type_name, "f{}", width * 8);

                crate::Type {
                    name: Some(type_name),
                    inner: crate::TypeInner::Struct {
                        members: vec![
                            crate::StructMember {
                                name: Some("fract".to_string()),
                                ty: member_ty,
                                binding: None,
                                offset: 0,
                            },
                            crate::StructMember {
                                name: Some("whole".to_string()),
                                ty: member_ty,
                                binding: None,
                                offset: second_offset,
                            },
                        ],
                        span: second_offset * 2,
                    },
                }
            }
            crate::PredeclaredType::FrexpResult { size, width } => {
                let float_ty = self.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar(crate::Scalar::float(width)),
                    },
                    Span::UNDEFINED,
                );

                let int_ty = self.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar(crate::Scalar {
                            kind: crate::ScalarKind::Sint,
                            width,
                        }),
                    },
                    Span::UNDEFINED,
                );

                let (fract_member_ty, exp_member_ty, second_offset) = if let Some(size) = size {
                    let vec_float_ty = self.types.insert(
                        crate::Type {
                            name: None,
                            inner: crate::TypeInner::Vector {
                                size,
                                scalar: crate::Scalar::float(width),
                            },
                        },
                        Span::UNDEFINED,
                    );
                    let vec_int_ty = self.types.insert(
                        crate::Type {
                            name: None,
                            inner: crate::TypeInner::Vector {
                                size,
                                scalar: crate::Scalar {
                                    kind: crate::ScalarKind::Sint,
                                    width,
                                },
                            },
                        },
                        Span::UNDEFINED,
                    );
                    (vec_float_ty, vec_int_ty, size as u32 * width as u32)
                } else {
                    (float_ty, int_ty, width as u32)
                };

                let mut type_name = "__frexp_result_".to_string();
                if let Some(size) = size {
                    let _ = write!(type_name, "vec{}_", size as u8);
                }
                let _ = write!(type_name, "f{}", width * 8);

                crate::Type {
                    name: Some(type_name),
                    inner: crate::TypeInner::Struct {
                        members: vec![
                            crate::StructMember {
                                name: Some("fract".to_string()),
                                ty: fract_member_ty,
                                binding: None,
                                offset: 0,
                            },
                            crate::StructMember {
                                name: Some("exp".to_string()),
                                ty: exp_member_ty,
                                binding: None,
                                offset: second_offset,
                            },
                        ],
                        span: second_offset * 2,
                    },
                }
            }
        };

        let handle = self.types.insert(ty, Span::UNDEFINED);
        self.special_types
            .predeclared_types
            .insert(special_type, handle);
        handle
    }
}
