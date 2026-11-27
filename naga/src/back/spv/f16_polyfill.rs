/*!
This module provides functionality for polyfilling 16-bit (`f16`, `i16` and `u16`)
input/output variables when the `StorageInputOutput16` capability is not available
or disabled.

It works by:

1. Declaring 16-bit I/O variables as their 32-bit counterparts (`f32`, `i32` or `u32`) in SPIR-V
2. Converting between the 16-bit and 32-bit types at runtime using `OpFConvert`,
   `OpSConvert` or `OpUConvert`
3. Maintaining mappings to track which variables need conversion
*/

use crate::back::spv::{Instruction, LocalType, NumericType, Word};
use alloc::vec::Vec;

/// Manages 16-bit I/O polyfill state and operations.
#[derive(Default)]
pub(in crate::back::spv) struct F16IoPolyfill {
    use_native: bool,
    /// For each polyfilled I/O variable, the id of the 32-bit type it is declared
    /// with, and the instruction that converts between that type and the 16-bit one.
    io_var_to_f32_type: crate::FastHashMap<Word, (Word, spirv::Op)>,
}

impl F16IoPolyfill {
    pub fn new(use_storage_input_output_16: bool) -> Self {
        Self {
            use_native: use_storage_input_output_16,
            io_var_to_f32_type: crate::FastHashMap::default(),
        }
    }

    pub const fn needs_polyfill(&self, ty_inner: &crate::TypeInner) -> bool {
        !self.use_native && Self::scalar_16_bit(ty_inner).is_some()
    }

    /// Returns the scalar of a 16-bit numeric scalar or vector type, or `None`
    /// for any other type.
    const fn scalar_16_bit(ty_inner: &crate::TypeInner) -> Option<crate::Scalar> {
        use crate::{ScalarKind as Sk, TypeInner};

        match *ty_inner {
            TypeInner::Scalar(scalar) | TypeInner::Vector { scalar, .. }
                if scalar.width == 2 && matches!(scalar.kind, Sk::Float | Sk::Sint | Sk::Uint) =>
            {
                Some(scalar)
            }
            _ => None,
        }
    }

    pub fn register_io_var(&mut self, variable_id: Word, f32_type_id: Word, convert_op: spirv::Op) {
        self.io_var_to_f32_type
            .insert(variable_id, (f32_type_id, convert_op));
    }

    /// Returns the id of the 32-bit type of a polyfilled I/O variable, and the
    /// instruction that converts between it and the 16-bit type.
    pub fn get_f32_io_type(&self, variable_id: Word) -> Option<(Word, spirv::Op)> {
        self.io_var_to_f32_type.get(&variable_id).copied()
    }

    /// Returns the 32-bit type used in place of a 16-bit I/O type, and the
    /// instruction that converts between the two.
    pub fn create_polyfill_type(ty_inner: &crate::TypeInner) -> Option<(LocalType, spirv::Op)> {
        use crate::{ScalarKind as Sk, TypeInner};

        let scalar = Self::scalar_16_bit(ty_inner)?;
        let (wide_scalar, convert_op) = match scalar.kind {
            Sk::Float => (crate::Scalar::F32, spirv::Op::FConvert),
            Sk::Sint => (crate::Scalar::I32, spirv::Op::SConvert),
            Sk::Uint => (crate::Scalar::U32, spirv::Op::UConvert),
            Sk::Bool | Sk::AbstractInt | Sk::AbstractFloat => return None,
        };
        let numeric = match *ty_inner {
            TypeInner::Vector { size, .. } => NumericType::Vector {
                size,
                scalar: wide_scalar,
            },
            _ => NumericType::Scalar(wide_scalar),
        };
        Some((LocalType::Numeric(numeric), convert_op))
    }

    /// Emits the conversion of `value_id` to `result_type_id`, between a 16-bit
    /// type and its 32-bit counterpart, in either direction.
    pub fn emit_conversion(
        convert_op: spirv::Op,
        value_id: Word,
        result_type_id: Word,
        converted_id: Word,
        body: &mut Vec<Instruction>,
    ) {
        body.push(Instruction::unary(
            convert_op,
            result_type_id,
            converted_id,
            value_id,
        ));
    }
}

impl crate::back::spv::reclaimable::Reclaimable for F16IoPolyfill {
    fn reclaim(mut self) -> Self {
        self.io_var_to_f32_type = self.io_var_to_f32_type.reclaim();
        self
    }
}
