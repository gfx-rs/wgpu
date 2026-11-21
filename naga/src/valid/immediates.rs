use crate::valid::{ImmediateSlots, ImmediateSlotsOverflowError, ImmediateUsage};

/// Construct an [`ImmediateUsage`] from a naga type.
pub trait ImmediateUsageExt {
    fn from_type(
        ty: &crate::TypeInner,
        types: &crate::UniqueArena<crate::Type>,
        gctx: crate::proc::GlobalCtx,
    ) -> ImmediateUsage;
}

impl ImmediateUsageExt for ImmediateUsage {
    fn from_type(
        ty: &crate::TypeInner,
        types: &crate::UniqueArena<crate::Type>,
        gctx: crate::proc::GlobalCtx,
    ) -> ImmediateUsage {
        let size = ty.size(gctx);
        ImmediateSlots::from_type(ty, types, gctx)
            .map(|slots| ImmediateUsage::Valid { slots, size })
            .unwrap_or(ImmediateUsage::Invalid { size })
    }
}

/// Construct an [`ImmediateSlots`] from a naga type.
pub trait ImmediateSlotsExt: Sized {
    /// Compute the slots occupied by a type,
    /// excluding padding between matrix columns or struct members.
    fn from_type(
        ty: &crate::TypeInner,
        types: &crate::UniqueArena<crate::Type>,
        gctx: crate::proc::GlobalCtx,
    ) -> Result<Self, ImmediateSlotsOverflowError>;
}

impl ImmediateSlotsExt for ImmediateSlots {
    fn from_type(
        ty: &crate::TypeInner,
        types: &crate::UniqueArena<crate::Type>,
        gctx: crate::proc::GlobalCtx,
    ) -> Result<Self, ImmediateSlotsOverflowError> {
        fn from_type_recursive(
            ty: &crate::TypeInner,
            offset: u32,
            types: &crate::UniqueArena<crate::Type>,
            gctx: crate::proc::GlobalCtx,
        ) -> Result<ImmediateSlots, ImmediateSlotsOverflowError> {
            // <https://www.w3.org/TR/WGSL/#accessible-bytes>
            match *ty {
                crate::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                } => {
                    let mut slots = ImmediateSlots::default();
                    let stride = crate::proc::Alignment::from(rows) * u32::from(scalar.width);
                    for col in 0..u32::from(columns) {
                        slots |= ImmediateSlots::from_range(
                            offset + col * stride,
                            u32::from(rows) * u32::from(scalar.width),
                        )?;
                    }
                    Ok(slots)
                }
                crate::TypeInner::Struct { ref members, .. } => {
                    let mut slots = ImmediateSlots::default();
                    for member in members {
                        let member_ty = &types[member.ty].inner;
                        slots |=
                            from_type_recursive(member_ty, offset + member.offset, types, gctx)?;
                    }
                    Ok(slots)
                }
                _ => ImmediateSlots::from_range(offset, ty.size(gctx)),
            }
        }
        from_type_recursive(ty, 0, types, gctx)
    }
}

#[cfg(test)]
mod tests {
    use crate::valid::ImmediateSlotsOverflowError;

    use super::{ImmediateSlots, ImmediateSlotsExt as _};

    #[test]
    fn range_single() {
        assert_eq!(
            ImmediateSlots::from_range(0, 4).unwrap(),
            ImmediateSlots::from_raw(0b1)
        );
        assert_eq!(
            ImmediateSlots::from_range(4, 4).unwrap(),
            ImmediateSlots::from_raw(0b10)
        );
        assert_eq!(
            ImmediateSlots::from_range(8, 4).unwrap(),
            ImmediateSlots::from_raw(0b100)
        );
    }

    #[test]
    fn range_vec4() {
        assert_eq!(
            ImmediateSlots::from_range(0, 16).unwrap(),
            ImmediateSlots::from_raw(0b1111)
        );
        assert_eq!(
            ImmediateSlots::from_range(16, 16).unwrap(),
            ImmediateSlots::from_raw(0b1111_0000)
        );
    }

    #[test]
    fn range_full_256() {
        assert_eq!(
            ImmediateSlots::from_range(0, 256).unwrap(),
            ImmediateSlots::from_raw(u64::MAX)
        );
    }

    #[test]
    fn range_overflow() {
        assert_eq!(
            ImmediateSlots::from_range(0, 257),
            Err(ImmediateSlotsOverflowError(257))
        );
    }

    #[test]
    fn from_type_overflow() {
        let module = crate::front::wgsl::parse_str(
            "struct S { \
            e64: mat4x4<f32>, \
            e128: mat4x4<f32>, \
            e192: mat4x4<f32>, \
            e256: mat4x4<f32>, \
            e260: f32\
            }",
        )
        .unwrap();
        let struct_ty = (module.types.iter().map(|ty| ty.1))
            .find(|ty| ty.name.as_deref() == Some("S"))
            .unwrap();
        let slots = ImmediateSlots::from_type(&struct_ty.inner, &module.types, module.to_ctx());
        assert_eq!(slots, Err(ImmediateSlotsOverflowError(260)));
    }

    #[test]
    fn from_type_excludes_struct_padding() {
        let module = crate::front::wgsl::parse_str("struct S { a: f32, b: vec4<f32> }").unwrap();
        let struct_ty = (module.types.iter().map(|ty| ty.1))
            .find(|ty| ty.name.as_deref() == Some("S"))
            .unwrap();
        let slots =
            ImmediateSlots::from_type(&struct_ty.inner, &module.types, module.to_ctx()).unwrap();
        assert_eq!(slots, ImmediateSlots::from_raw(0b1111_0001));
    }

    #[test]
    fn from_type_excludes_matrix_padding() {
        let module = crate::front::wgsl::parse_str("struct S { mat: mat3x3<f32> }").unwrap();
        let struct_ty = (module.types.iter().map(|ty| ty.1))
            .find(|ty| ty.name.as_deref() == Some("S"))
            .unwrap();
        let slots =
            ImmediateSlots::from_type(&struct_ty.inner, &module.types, module.to_ctx()).unwrap();
        assert_eq!(slots, ImmediateSlots::from_raw(0b0111_0111_0111));

        let module =
            crate::front::wgsl::parse_str("struct S { f: f32, mat: mat2x2<f32> }").unwrap();
        let struct_ty = (module.types.iter().map(|ty| ty.1))
            .find(|ty| ty.name.as_deref() == Some("S"))
            .unwrap();
        let slots =
            ImmediateSlots::from_type(&struct_ty.inner, &module.types, module.to_ctx()).unwrap();
        assert_eq!(slots, ImmediateSlots::from_raw(0b11_11_01));
    }

    #[test]
    fn range_unaligned() {
        assert_eq!(
            ImmediateSlots::from_range(0, 3).unwrap(),
            ImmediateSlots::from_raw(0b1)
        );
        assert_eq!(
            ImmediateSlots::from_range(0, 5).unwrap(),
            ImmediateSlots::from_raw(0b11)
        );
    }

    #[test]
    fn contains() {
        let required = ImmediateSlots::from_raw(0b1111_0001);
        let mut set = ImmediateSlots::default();
        assert!(!set.contains(required));
        set |= ImmediateSlots::from_range(0, 4).unwrap();
        assert!(!set.contains(required));
        set |= ImmediateSlots::from_range(16, 16).unwrap();
        assert!(set.contains(required));
    }

    #[test]
    fn difference() {
        let required = ImmediateSlots::from_raw(0b1111_0001);
        let set = ImmediateSlots::from_range(0, 4).unwrap();
        assert_eq!(
            required.difference(set),
            ImmediateSlots::from_raw(0b1111_0000)
        );
    }
}
