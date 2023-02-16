/*!
Type generators.
*/

use crate::{arena::Handle, span::Span};

impl crate::Module {
    pub(super) fn generate_ray_desc_type(&mut self) -> Handle<crate::Type> {
        if let Some(handle) = self.special_types.ray_desc {
            return handle;
        }

        let width = 4;
        let ty_flag = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar {
                    width,
                    kind: crate::ScalarKind::Uint,
                },
            },
            Span::UNDEFINED,
        );
        let ty_scalar = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar {
                    width,
                    kind: crate::ScalarKind::Float,
                },
            },
            Span::UNDEFINED,
        );
        let ty_vector = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Vector {
                    size: crate::VectorSize::Tri,
                    kind: crate::ScalarKind::Float,
                    width,
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

    pub(super) fn generate_ray_intersection_type(&mut self) -> Handle<crate::Type> {
        if let Some(handle) = self.special_types.ray_intersection {
            return handle;
        }

        let width = 4;
        let ty_flag = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar {
                    width,
                    kind: crate::ScalarKind::Uint,
                },
            },
            Span::UNDEFINED,
        );
        let ty_scalar = self.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Scalar {
                    width,
                    kind: crate::ScalarKind::Float,
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
                        //TODO: the rest
                    ],
                    span: 8,
                },
            },
            Span::UNDEFINED,
        );

        self.special_types.ray_intersection = Some(handle);
        handle
    }
}
