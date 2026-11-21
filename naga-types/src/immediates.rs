//! Tracking of which bytes of immediate data a shader uses.

use core::{
    fmt::{self, Debug},
    ops,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmediateSlotsOverflowError(pub u64);

impl fmt::Display for ImmediateSlotsOverflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Immediate size {} overflows the bitmask", self.0)
    }
}

impl core::error::Error for ImmediateSlotsOverflowError {}

/// A bitmask, tracking which 4-byte slots have been written via `set_immediates`.
/// Bit N corresponds to bytes [N*4 .. N*4+4).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ImmediateSlots(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
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
