use core::{fmt, ops};

/// A bitmask, tracking which 4-byte slots have been written via `set_immediates`.
/// Bit N corresponds to bytes [N*4 .. N*4+4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct ImmediateSlots(u64);

impl ImmediateSlots {
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Compute the bitmask for a byte range [offset .. offset + size_bytes).
    pub const fn from_range(offset: u32, size_bytes: u32) -> Self {
        if size_bytes == 0 {
            return Self(0);
        }
        let lo = offset / 4;
        let hi = (offset + size_bytes).div_ceil(4);
        Self(u64::MAX << lo & u64::MAX >> (64 - hi))
    }

    /// Compute the slots occupied by a type at a given byte offset,
    /// excluding padding between struct members.
    pub fn from_type(
        ty: &crate::TypeInner,
        offset: u32,
        types: &crate::UniqueArena<crate::Type>,
        gctx: crate::proc::GlobalCtx,
    ) -> Self {
        match *ty {
            crate::TypeInner::Struct { ref members, .. } => {
                let mut slots = Self::default();
                for member in members {
                    let member_ty = &types[member.ty].inner;
                    slots |= Self::from_type(member_ty, offset + member.offset, types, gctx);
                }
                slots
            }
            _ => Self::from_range(offset, ty.size(gctx)),
        }
    }

    /// Returns true if `self` contains all bits in `other`.
    pub const fn contains(self, other: Self) -> bool {
        other.0 & !self.0 == 0
    }

    /// Returns the bits in `self` that are not set in `other`.
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Returns the byte size of the `var<immediate>` type in a module.
    /// Zero if the module has no `var<immediate>`.
    pub fn size_for_module(module: &crate::Module) -> u32 {
        module
            .global_variables
            .iter()
            .find(|&(_, var)| var.space == crate::AddressSpace::Immediate)
            .map(|(_, var)| module.types[var.ty].inner.size(module.to_ctx()))
            .unwrap_or(0)
    }

    /// Compute the immediate slot bitmask for a pointer expression that
    /// refers to (part of) an immediate global variable.
    ///
    /// `global` is the handle of the immediate global variable that this
    /// pointer derives from (obtained from `assignable_global`).
    pub(crate) fn for_pointer(
        pointer: crate::arena::Handle<crate::Expression>,
        global: crate::arena::Handle<crate::GlobalVariable>,
        expression_arena: &crate::Arena<crate::Expression>,
        global_vars: &crate::Arena<crate::GlobalVariable>,
        types: &crate::UniqueArena<crate::Type>,
    ) -> Self {
        use crate::Expression as E;
        use crate::TypeInner;

        let gctx = crate::proc::GlobalCtx {
            types,
            constants: &crate::Arena::new(),
            overrides: &crate::Arena::new(),
            global_expressions: &crate::Arena::new(),
        };

        let global_ty = &types[global_vars[global].ty].inner;

        match expression_arena[pointer] {
            E::GlobalVariable(_) => Self::from_type(global_ty, 0, types, gctx),
            E::AccessIndex { base, index } => {
                if let E::GlobalVariable(_) = expression_arena[base] {
                    if let TypeInner::Struct { ref members, .. } = *global_ty {
                        let member = &members[index as usize];
                        let member_ty = &types[member.ty].inner;
                        return Self::from_type(member_ty, member.offset, types, gctx);
                    }
                }
                Self::from_type(global_ty, 0, types, gctx)
            }
            _ => Self::from_type(global_ty, 0, types, gctx),
        }
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

#[cfg(test)]
mod tests {
    use super::ImmediateSlots;

    #[test]
    fn range_single() {
        assert_eq!(
            ImmediateSlots::from_range(0, 4),
            ImmediateSlots::from_raw(0b1)
        );
        assert_eq!(
            ImmediateSlots::from_range(4, 4),
            ImmediateSlots::from_raw(0b10)
        );
        assert_eq!(
            ImmediateSlots::from_range(8, 4),
            ImmediateSlots::from_raw(0b100)
        );
    }

    #[test]
    fn range_vec4() {
        assert_eq!(
            ImmediateSlots::from_range(0, 16),
            ImmediateSlots::from_raw(0b1111)
        );
        assert_eq!(
            ImmediateSlots::from_range(16, 16),
            ImmediateSlots::from_raw(0b1111_0000)
        );
    }

    #[test]
    fn range_full_256() {
        assert_eq!(
            ImmediateSlots::from_range(0, 256),
            ImmediateSlots::from_raw(u64::MAX)
        );
    }

    #[test]
    fn from_type_excludes_struct_padding() {
        let module = crate::front::wgsl::parse_str("struct S { a: f32, b: vec4<f32> }").unwrap();
        let struct_ty = (module.types.iter().map(|ty| ty.1))
            .find(|ty| ty.name.as_deref() == Some("S"))
            .unwrap();
        let slots = ImmediateSlots::from_type(&struct_ty.inner, 0, &module.types, module.to_ctx());
        assert_eq!(slots, ImmediateSlots::from_raw(0b1111_0001));
    }

    #[test]
    fn range_unaligned() {
        assert_eq!(
            ImmediateSlots::from_range(0, 3),
            ImmediateSlots::from_raw(0b1)
        );
        assert_eq!(
            ImmediateSlots::from_range(0, 5),
            ImmediateSlots::from_raw(0b11)
        );
    }

    #[test]
    fn contains() {
        let required = ImmediateSlots::from_raw(0b1111_0001);
        let mut set = ImmediateSlots::default();
        assert!(!set.contains(required));
        set |= ImmediateSlots::from_range(0, 4);
        assert!(!set.contains(required));
        set |= ImmediateSlots::from_range(16, 16);
        assert!(set.contains(required));
    }

    #[test]
    fn difference() {
        let required = ImmediateSlots::from_raw(0b1111_0001);
        let set = ImmediateSlots::from_range(0, 4);
        assert_eq!(
            required.difference(set),
            ImmediateSlots::from_raw(0b1111_0000)
        );
    }
}
