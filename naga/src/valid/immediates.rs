use core::{
    fmt::{self, Debug},
    ops,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("Immediate size {0} overflows the bitmask")]
pub struct ImmediateSlotsOverflowError(pub u64);

/// A bitmask, tracking which 4-byte slots have been written via `set_immediates`.
/// Bit N corresponds to bytes [N*4 .. N*4+4).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct ImmediateSlots(u64);

#[derive(Clone, Copy, Debug)]
pub enum ImmediateUsage {
    Valid { slots: ImmediateSlots, size: u32 },
    // Used when a shader's immediate data type exceeds the maximum of 256 bytes and cannot
    // be represented in the `ImmediateSlots` bitmask.
    Invalid { size: u32 },
}

impl Default for ImmediateUsage {
    fn default() -> Self {
        ImmediateUsage::Valid {
            slots: ImmediateSlots::default(),
            size: 0,
        }
    }
}

impl ImmediateUsage {
    pub const fn size(&self) -> u32 {
        match *self {
            ImmediateUsage::Valid { size, .. } => size,
            ImmediateUsage::Invalid { size } => size,
        }
    }

    pub fn from_type(
        ty: &crate::TypeInner,
        types: &crate::UniqueArena<crate::Type>,
        gctx: crate::proc::GlobalCtx,
    ) -> Self {
        let size = ty.size(gctx);
        ImmediateSlots::from_type(ty, types, gctx)
            .map(|slots| Self::Valid { slots, size })
            .unwrap_or(Self::Invalid { size })
    }

    pub fn merge(&self, other: &ImmediateUsage) -> Self {
        let size = self.size().max(other.size());
        match (*self, *other) {
            (
                ImmediateUsage::Valid { slots, .. },
                ImmediateUsage::Valid {
                    slots: other_slots, ..
                },
            ) => Self::Valid {
                slots: slots | other_slots,
                size,
            },
            _ => Self::Invalid { size },
        }
    }
}

impl ImmediateSlots {
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Compute the bitmask for a byte range [offset .. offset + size_bytes).
    pub const fn from_range(
        offset: u32,
        size_bytes: u32,
    ) -> Result<Self, ImmediateSlotsOverflowError> {
        let Some(end) = offset.checked_add(size_bytes) else {
            return Err(ImmediateSlotsOverflowError(
                offset as u64 + size_bytes as u64,
            ));
        };
        if end > u64::BITS * 4 {
            return Err(ImmediateSlotsOverflowError(end as u64));
        }
        if size_bytes == 0 {
            return Ok(Self(0));
        }
        let lo = offset / 4;
        let hi = (offset + size_bytes).div_ceil(4);
        Ok(Self(u64::MAX << lo & u64::MAX >> (64 - hi)))
    }

    /// Compute the slots occupied by a type,
    /// excluding padding between matrix columns or struct members.
    pub fn from_type(
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

    /// Returns true if `self` contains all bits in `other`.
    pub const fn contains(self, other: Self) -> bool {
        other.0 & !self.0 == 0
    }

    /// Returns the bits in `self` that are not set in `other`.
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl ops::BitOrAssign for ImmediateSlots {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl ops::BitOr for ImmediateSlots {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl fmt::Display for ImmediateSlots {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return write!(f, "(none)");
        }
        let mut first = true;
        let mut bit = 0u32;
        while bit < 64 {
            if self.0 & (1u64 << bit) != 0 {
                let start = bit * 4;
                while bit < 64 && self.0 & (1u64 << bit) != 0 {
                    bit += 1;
                }
                let end = bit * 4;
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{start}..{end}")?;
                first = false;
            } else {
                bit += 1;
            }
        }
        Ok(())
    }
}

impl Debug for ImmediateSlots {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use crate::valid::ImmediateSlotsOverflowError;

    use super::ImmediateSlots;

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
