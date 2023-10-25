/*!
Bounds-checking for SPIR-V output.
*/

use super::{
    helpers::global_needs_wrapper, selection::Selection, Block, BlockContext, Error, IdGenerator,
    Instruction, Word,
};
use crate::{arena::Handle, proc::BoundsCheckPolicy};

/// The results of performing a bounds check.
///
/// On success, `write_bounds_check` returns a value of this type.
pub(super) enum BoundsCheckResult {
    /// The index is statically known and in bounds, with the given value.
    KnownInBounds(u32),

    /// The given instruction computes the index to be used.
    Computed(Word),

    /// The given instruction computes a boolean condition which is true
    /// if the index is in bounds.
    Conditional(Word),
}

/// A value that we either know at translation time, or need to compute at runtime.
pub(super) enum MaybeKnown<T> {
    /// The value is known at shader translation time.
    Known(T),

    /// The value is computed by the instruction with the given id.
    Computed(Word),
}

impl<'w> BlockContext<'w> {
    /// Emit code to compute the length of a run-time array.
    ///
    /// Given `array`, an expression referring a runtime-sized array, return the
    /// instruction id for the array's length.
    pub(super) fn write_runtime_array_length(
        &mut self,
        array: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        // Naga IR permits runtime-sized arrays as global variables or as the
        // final member of a struct that is a global variable. SPIR-V permits
        // only the latter, so this back end wraps bare runtime-sized arrays
        // in a made-up struct; see `helpers::global_needs_wrapper` and its uses.
        // This code must handle both cases.
        let (structure_id, last_member_index) = match self.ir_function.expressions[array] {
            crate::Expression::AccessIndex { base, index } => {
                match self.ir_function.expressions[base] {
                    crate::Expression::GlobalVariable(handle) => (
                        self.writer.global_variables[handle.index()].access_id,
                        index,
                    ),
                    _ => return Err(Error::Validation("array length expression")),
                }
            }
            crate::Expression::GlobalVariable(handle) => {
                let global = &self.ir_module.global_variables[handle];
                if !global_needs_wrapper(self.ir_module, global) {
                    return Err(Error::Validation("array length expression"));
                }

                (self.writer.global_variables[handle.index()].var_id, 0)
            }
            _ => return Err(Error::Validation("array length expression")),
        };

        let length_id = self.gen_id();
        block.body.push(Instruction::array_length(
            self.writer.get_uint_type_id(),
            length_id,
            structure_id,
            last_member_index,
        ));

        Ok(length_id)
    }

    /// Compute the length of a subscriptable value.
    ///
    /// Given `sequence`, an expression referring to some indexable type, return
    /// its length. The result may either be computed by SPIR-V instructions, or
    /// known at shader translation time.
    ///
    /// `sequence` may be a `Vector`, `Matrix`, or `Array`, a `Pointer` to any
    /// of those, or a `ValuePointer`. An array may be fixed-size, dynamically
    /// sized, or use a specializable constant as its length.
    fn write_sequence_length(
        &mut self,
        sequence: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<MaybeKnown<u32>, Error> {
        let sequence_ty = self.fun_info[sequence].ty.inner_with(&self.ir_module.types);
        match sequence_ty.indexable_length(self.ir_module) {
            Ok(crate::proc::IndexableLength::Known(known_length)) => {
                Ok(MaybeKnown::Known(known_length))
            }
            Ok(crate::proc::IndexableLength::Dynamic) => {
                let length_id = self.write_runtime_array_length(sequence, block)?;
                Ok(MaybeKnown::Computed(length_id))
            }
            Err(err) => {
                log::error!("Sequence length for {:?} failed: {}", sequence, err);
                Err(Error::Validation("indexable length"))
            }
        }
    }

    /// Compute the maximum valid index of a subscriptable value.
    ///
    /// Given `sequence`, an expression referring to some indexable type, return
    /// its maximum valid index - one less than its length. The result may
    /// either be computed, or known at shader translation time.
    ///
    /// `sequence` may be a `Vector`, `Matrix`, or `Array`, a `Pointer` to any
    /// of those, or a `ValuePointer`. An array may be fixed-size, dynamically
    /// sized, or use a specializable constant as its length.
    fn write_sequence_max_index(
        &mut self,
        sequence: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<MaybeKnown<u32>, Error> {
        match self.write_sequence_length(sequence, block)? {
            MaybeKnown::Known(known_length) => {
                // We should have thrown out all attempts to subscript zero-length
                // sequences during validation, so the following subtraction should never
                // underflow.
                assert!(known_length > 0);
                // Compute the max index from the length now.
                Ok(MaybeKnown::Known(known_length - 1))
            }
            MaybeKnown::Computed(length_id) => {
                // Emit code to compute the max index from the length.
                let const_one_id = self.get_index_constant(1);
                let max_index_id = self.gen_id();
                block.body.push(Instruction::binary(
                    spirv::Op::ISub,
                    self.writer.get_uint_type_id(),
                    max_index_id,
                    length_id,
                    const_one_id,
                ));
                Ok(MaybeKnown::Computed(max_index_id))
            }
        }
    }

    /// Restrict an index to be in range for a vector, matrix, or array.
    ///
    /// This is used to implement `BoundsCheckPolicy::Restrict`. An in-bounds
    /// index is left unchanged. An out-of-bounds index is replaced with some
    /// arbitrary in-bounds index. Note,this is not necessarily clamping; for
    /// example, negative indices might be changed to refer to the last element
    /// of the sequence, not the first, as clamping would do.
    ///
    /// Either return the restricted index value, if known, or add instructions
    /// to `block` to compute it, and return the id of the result. See the
    /// documentation for `BoundsCheckResult` for details.
    ///
    /// The `sequence` expression may be a `Vector`, `Matrix`, or `Array`, a
    /// `Pointer` to any of those, or a `ValuePointer`. An array may be
    /// fixed-size, dynamically sized, or use a specializable constant as its
    /// length.
    pub(super) fn write_restricted_index(
        &mut self,
        sequence: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        let index_id = self.cached[index];

        // Get the sequence's maximum valid index. Return early if we've already
        // done the bounds check.
        let max_index_id = match self.write_sequence_max_index(sequence, block)? {
            MaybeKnown::Known(known_max_index) => {
                if let Ok(known_index) = self
                    .ir_module
                    .to_ctx()
                    .eval_expr_to_u32_from(index, &self.ir_function.expressions)
                {
                    // Both the index and length are known at compile time.
                    //
                    // In strict WGSL compliance mode, out-of-bounds indices cannot be
                    // reported at shader translation time, and must be replaced with
                    // in-bounds indices at run time. So we cannot assume that
                    // validation ensured the index was in bounds. Restrict now.
                    let restricted = std::cmp::min(known_index, known_max_index);
                    return Ok(BoundsCheckResult::KnownInBounds(restricted));
                }

                self.get_index_constant(known_max_index)
            }
            MaybeKnown::Computed(max_index_id) => max_index_id,
        };

        // One or the other of the index or length is dynamic, so emit code for
        // BoundsCheckPolicy::Restrict.
        let restricted_index_id = self.gen_id();
        block.body.push(Instruction::ext_inst(
            self.writer.gl450_ext_inst_id,
            spirv::GLOp::UMin,
            self.writer.get_uint_type_id(),
            restricted_index_id,
            &[index_id, max_index_id],
        ));
        Ok(BoundsCheckResult::Computed(restricted_index_id))
    }

    /// Write an index bounds comparison to `block`, if needed.
    ///
    /// If we're able to determine statically that `index` is in bounds for
    /// `sequence`, return `KnownInBounds(value)`, where `value` is the actual
    /// value of the index. (In principle, one could know that the index is in
    /// bounds without knowing its specific value, but in our simple-minded
    /// situation, we always know it.)
    ///
    /// If instead we must generate code to perform the comparison at run time,
    /// return `Conditional(comparison_id)`, where `comparison_id` is an
    /// instruction producing a boolean value that is true if `index` is in
    /// bounds for `sequence`.
    ///
    /// The `sequence` expression may be a `Vector`, `Matrix`, or `Array`, a
    /// `Pointer` to any of those, or a `ValuePointer`. An array may be
    /// fixed-size, dynamically sized, or use a specializable constant as its
    /// length.
    fn write_index_comparison(
        &mut self,
        sequence: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        let index_id = self.cached[index];

        // Get the sequence's length. Return early if we've already done the
        // bounds check.
        let length_id = match self.write_sequence_length(sequence, block)? {
            MaybeKnown::Known(known_length) => {
                if let Ok(known_index) = self
                    .ir_module
                    .to_ctx()
                    .eval_expr_to_u32_from(index, &self.ir_function.expressions)
                {
                    // Both the index and length are known at compile time.
                    //
                    // It would be nice to assume that, since we are using the
                    // `ReadZeroSkipWrite` policy, we are not in strict WGSL
                    // compliance mode, and thus we can count on the validator to have
                    // rejected any programs with known out-of-bounds indices, and
                    // thus just return `KnownInBounds` here without actually
                    // checking.
                    //
                    // But it's also reasonable to expect that bounds check policies
                    // and error reporting policies should be able to vary
                    // independently without introducing security holes. So, we should
                    // support the case where bad indices do not cause validation
                    // errors, and are handled via `ReadZeroSkipWrite`.
                    //
                    // In theory, when `known_index` is bad, we could return a new
                    // `KnownOutOfBounds` variant here. But it's simpler just to fall
                    // through and let the bounds check take place. The shader is
                    // broken anyway, so it doesn't make sense to invest in emitting
                    // the ideal code for it.
                    if known_index < known_length {
                        return Ok(BoundsCheckResult::KnownInBounds(known_index));
                    }
                }

                self.get_index_constant(known_length)
            }
            MaybeKnown::Computed(length_id) => length_id,
        };

        // Compare the index against the length.
        let condition_id = self.gen_id();
        block.body.push(Instruction::binary(
            spirv::Op::ULessThan,
            self.writer.get_bool_type_id(),
            condition_id,
            index_id,
            length_id,
        ));

        // Indicate that we did generate the check.
        Ok(BoundsCheckResult::Conditional(condition_id))
    }

    /// Emit a conditional load for `BoundsCheckPolicy::ReadZeroSkipWrite`.
    ///
    /// Generate code to load a value of `result_type` if `condition` is true,
    /// and generate a null value of that type if it is false. Call `emit_load`
    /// to emit the instructions to perform the load. Return the id of the
    /// merged value of the two branches.
    pub(super) fn write_conditional_indexed_load<F>(
        &mut self,
        result_type: Word,
        condition: Word,
        block: &mut Block,
        emit_load: F,
    ) -> Word
    where
        F: FnOnce(&mut IdGenerator, &mut Block) -> Word,
    {
        // For the out-of-bounds case, we produce a zero value.
        let null_id = self.writer.get_constant_null(result_type);

        let mut selection = Selection::start(block, result_type);

        // As it turns out, we don't actually need a full 'if-then-else'
        // structure for this: SPIR-V constants are declared up front, so the
        // 'else' block would have no instructions. Instead we emit something
        // like this:
        //
        //     result = zero;
        //     if in_bounds {
        //         result = do the load;
        //     }
        //     use result;

        // Continue only if the index was in bounds. Otherwise, branch to the
        // merge block.
        selection.if_true(self, condition, null_id);

        // The in-bounds path. Perform the access and the load.
        let loaded_value = emit_load(&mut self.writer.id_gen, selection.block());

        selection.finish(self, loaded_value)
    }

    /// Emit code for bounds checks for an array, vector, or matrix access.
    ///
    /// This implements either `index_bounds_check_policy` or
    /// `buffer_bounds_check_policy`, depending on the address space of the
    /// pointer being accessed.
    ///
    /// Return a `BoundsCheckResult` indicating how the index should be
    /// consumed. See that type's documentation for details.
    pub(super) fn write_bounds_check(
        &mut self,
        base: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        let policy = self.writer.bounds_check_policies.choose_policy(
            base,
            &self.ir_module.types,
            self.fun_info,
        );

        Ok(match policy {
            BoundsCheckPolicy::Restrict => self.write_restricted_index(base, index, block)?,
            BoundsCheckPolicy::ReadZeroSkipWrite => {
                self.write_index_comparison(base, index, block)?
            }
            BoundsCheckPolicy::Unchecked => BoundsCheckResult::Computed(self.cached[index]),
        })
    }

    /// Emit code to subscript a vector by value with a computed index.
    ///
    /// Return the id of the element value.
    pub(super) fn write_vector_access(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        base: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        let result_type_id = self.get_expression_type_id(&self.fun_info[expr_handle].ty);

        let base_id = self.cached[base];
        let index_id = self.cached[index];

        let result_id = match self.write_bounds_check(base, index, block)? {
            BoundsCheckResult::KnownInBounds(known_index) => {
                let result_id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    result_type_id,
                    result_id,
                    base_id,
                    &[known_index],
                ));
                result_id
            }
            BoundsCheckResult::Computed(computed_index_id) => {
                let result_id = self.gen_id();
                block.body.push(Instruction::vector_extract_dynamic(
                    result_type_id,
                    result_id,
                    base_id,
                    computed_index_id,
                ));
                result_id
            }
            BoundsCheckResult::Conditional(comparison_id) => {
                // Run-time bounds checks were required. Emit
                // conditional load.
                self.write_conditional_indexed_load(
                    result_type_id,
                    comparison_id,
                    block,
                    |id_gen, block| {
                        // The in-bounds path. Generate the access.
                        let element_id = id_gen.next();
                        block.body.push(Instruction::vector_extract_dynamic(
                            result_type_id,
                            element_id,
                            base_id,
                            index_id,
                        ));
                        element_id
                    },
                )
            }
        };

        Ok(result_id)
    }
}
