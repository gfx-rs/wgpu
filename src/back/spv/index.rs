//! Bounds-checking for SPIR-V output.

use super::{Block, BlockContext, Error, IdGenerator, Instruction, Word};
use crate::{arena::Handle, back::IndexBoundsCheckPolicy};

/// The results of emitting code for a left-hand-side expression.
///
/// On success, `write_expression_pointer` returns one of these.
pub(super) enum ExpressionPointer {
    /// The pointer to the expression's value is available, as the value of the
    /// expression with the given id.
    Ready { pointer_id: Word },

    /// The access expression must be conditional on the value of `condition`, a boolean
    /// expression that is true if all indices are in bounds. If `condition` is true, then
    /// `access` is an `OpAccessChain` instruction that will compute a pointer to the
    /// expression's value. If `condition` is false, then executing `access` would be
    /// undefined behavior.
    Conditional {
        condition: Word,
        access: Instruction,
    },
}

/// The results of performing a bounds check.
///
/// On success, `write_bounds_check` returns a value of this type.
pub enum BoundsCheckResult {
    /// The index is statically known and in bounds, with the given value.
    KnownInBounds(u32),

    /// The given instruction computes the index to be used.
    Computed(Word),

    /// The given instruction computes a boolean condition which is true
    /// if the index is in bounds.
    Conditional(Word),
}

/// A value that we either know at translation time, or need to compute at runtime.
pub enum MaybeKnown<T> {
    /// The value is known at shader translation time.
    Known(T),

    /// The value is computed by the instruction with the given id.
    Computed(Word),
}

impl<'w> BlockContext<'w> {
    /// Emit code to compute the length of a run-time array.
    ///
    /// Given `array`, an expression referring to the final member of a struct,
    /// where the member in question is a runtime-sized array, return the
    /// instruction id for the array's length.
    pub(super) fn write_runtime_array_length(
        &mut self,
        array: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        // Look into the expression to find the value and type of the struct
        // holding the dynamically-sized array.
        let (structure_id, last_member_index) = match self.ir_function.expressions[array] {
            crate::Expression::AccessIndex { base, index } => {
                match self.ir_function.expressions[base] {
                    crate::Expression::GlobalVariable(handle) => {
                        (self.writer.global_variables[handle.index()].id, index)
                    }
                    crate::Expression::FunctionArgument(index) => {
                        let parameter_id = self.function.parameter_id(index);
                        (parameter_id, index)
                    }
                    _ => return Err(Error::Validation("array length expression")),
                }
            }
            _ => return Err(Error::Validation("array length expression")),
        };

        let length_id = self.gen_id();
        block.body.push(Instruction::array_length(
            self.writer.get_uint_type_id()?,
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
        match sequence_ty.indexable_length(self.ir_module)? {
            crate::proc::IndexableLength::Known(known_length) => {
                Ok(MaybeKnown::Known(known_length))
            }
            crate::proc::IndexableLength::Dynamic => {
                let length_id = self.write_runtime_array_length(sequence, block)?;
                Ok(MaybeKnown::Computed(length_id))
            }
            crate::proc::IndexableLength::Specializable(constant) => {
                let length_id = self.writer.constant_ids[constant.index()];
                Ok(MaybeKnown::Computed(length_id))
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
                let const_one_id = self.get_index_constant(1)?;
                let max_index_id = self.gen_id();
                block.body.push(Instruction::binary(
                    spirv::Op::ISub,
                    self.writer.get_uint_type_id()?,
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
    /// This is used to implement `IndexBoundsCheckPolicy::Restrict`. An
    /// in-bounds index is left unchanged. An out-of-bounds index is replaced
    /// with some arbitrary in-bounds index. Note,this is not necessarily
    /// clamping; for example, negative indices might be changed to refer to the
    /// last element of the sequence, not the first, as clamping would do.
    ///
    /// Either return the restricted index value, if known, or add instructions
    /// to `block` to compute it, and return the id of the result. See the
    /// documentation for `BoundsCheckResult` for details.
    ///
    /// The `sequence` expression may be a `Vector`, `Matrix`, or `Array`, a
    /// `Pointer` to any of those, or a `ValuePointer`. An array may be
    /// fixed-size, dynamically sized, or use a specializable constant as its
    /// length.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_restricted_index(
        &mut self,
        sequence: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        let index_id = self.writer.cached[index];

        // Get the sequence's maximum valid index. Return early if we've already
        // done the bounds check.
        let max_index_id = match self.write_sequence_max_index(sequence, block)? {
            MaybeKnown::Known(known_max_index) => {
                if let crate::Expression::Constant(index_k) = self.ir_function.expressions[index] {
                    if let Some(known_index) = self.ir_module.constants[index_k].to_array_length() {
                        // Both the index and length are known at compile time.
                        //
                        // In strict WGSL compliance mode, out-of-bounds indices cannot be
                        // reported at shader translation time, and must be replaced with
                        // in-bounds indices at run time. So we cannot assume that
                        // validation ensured the index was in bounds. Restrict now.
                        let restricted = std::cmp::min(known_index, known_max_index);
                        return Ok(BoundsCheckResult::KnownInBounds(restricted));
                    }
                }

                self.get_index_constant(known_max_index)?
            }
            MaybeKnown::Computed(max_index_id) => max_index_id,
        };

        // One or the other of the index or length is dynamic, so emit code for
        // IndexBoundsCheckPolicy::Restrict.
        let restricted_index_id = self.gen_id();
        block.body.push(Instruction::ext_inst(
            self.writer.gl450_ext_inst_id,
            spirv::GLOp::UMin,
            self.writer.get_uint_type_id()?,
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
    #[allow(clippy::too_many_arguments)]
    fn write_index_comparison(
        &mut self,
        sequence: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        let index_id = self.writer.cached[index];

        // Get the sequence's length. Return early if we've already done the
        // bounds check.
        let length_id = match self.write_sequence_length(sequence, block)? {
            MaybeKnown::Known(known_length) => {
                if let crate::Expression::Constant(index_k) = self.ir_function.expressions[index] {
                    if let Some(known_index) = self.ir_module.constants[index_k].to_array_length() {
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
                }

                self.get_index_constant(known_length)?
            }
            MaybeKnown::Computed(length_id) => length_id,
        };

        // Compare the index against the length.
        let condition_id = self.gen_id();
        block.body.push(Instruction::binary(
            spirv::Op::ULessThan,
            self.writer.get_bool_type_id()?,
            condition_id,
            index_id,
            length_id,
        ));

        // Indicate that we did generate the check.
        Ok(BoundsCheckResult::Conditional(condition_id))
    }

    /// Emit a conditional load for `IndexBoundsCheckPolicy::ReadZeroSkipWrite`.
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
        let header_block = block.label_id;
        let merge_block = self.gen_id();
        let in_bounds_block = self.gen_id();

        // Branch based on whether the index was in bounds.
        //
        // As it turns out, our out-of-bounds branch block would contain no
        // instructions: it just produces a constant zero, whose instruction is
        // in the module's declarations section at the front. In this case,
        // SPIR-V lets us omit the empty 'else' block, and branch directly to
        // the merge block. The phi instruction in the merge block can cite the
        // header block as its CFG predecessor.
        block.body.push(Instruction::selection_merge(
            merge_block,
            spirv::SelectionControl::NONE,
        ));
        self.function.consume(
            std::mem::replace(block, Block::new(in_bounds_block)),
            Instruction::branch_conditional(condition, in_bounds_block, merge_block),
        );

        // The in-bounds path. Perform the access and the load.
        let value_id = emit_load(&mut self.writer.id_gen, block);

        // Finish the in-bounds block and start the merge block. This
        // is the block we'll leave current on return.
        self.function.consume(
            std::mem::replace(block, Block::new(merge_block)),
            Instruction::branch(merge_block),
        );

        // For the out-of-bounds case, produce a zero value.
        let null_id = self.writer.write_constant_null(result_type);

        // Merge the results from the two paths.
        let result_id = self.gen_id();
        block.body.push(Instruction::phi(
            result_type,
            result_id,
            &[(value_id, in_bounds_block), (null_id, header_block)],
        ));

        result_id
    }

    /// Emit code for bounds checks, per self.index_bounds_check_policy.
    ///
    /// Return a `BoundsCheckResult` indicating how the index should be
    /// consumed. See that type's documentation for details.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_bounds_check(
        &mut self,
        base: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<BoundsCheckResult, Error> {
        Ok(match self.writer.index_bounds_check_policy {
            IndexBoundsCheckPolicy::Restrict => self.write_restricted_index(base, index, block)?,
            IndexBoundsCheckPolicy::ReadZeroSkipWrite => {
                self.write_index_comparison(base, index, block)?
            }
            IndexBoundsCheckPolicy::UndefinedBehavior => {
                BoundsCheckResult::Computed(self.writer.cached[index])
            }
        })
    }

    /// Emit code to subscript a vector by value with a computed index.
    ///
    /// Return the id of the element value.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_vector_access(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        base: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        let result_type_id = self.get_expression_type_id(&self.fun_info[expr_handle].ty)?;

        let base_id = self.writer.cached[base];
        let index_id = self.writer.cached[index];

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
