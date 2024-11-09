/*!
Implementations for `BlockContext` methods.
*/

use super::{
    index::BoundsCheckResult, selection::Selection, Block, BlockContext, Dimension, Error,
    Instruction, LocalType, LookupType, NumericType, ResultMember, Writer, WriterFlags,
};
use crate::{arena::Handle, proc::index::GuardedIndex, Statement};
use spirv::Word;

fn get_dimension(type_inner: &crate::TypeInner) -> Dimension {
    match *type_inner {
        crate::TypeInner::Scalar(_) => Dimension::Scalar,
        crate::TypeInner::Vector { .. } => Dimension::Vector,
        crate::TypeInner::Matrix { .. } => Dimension::Matrix,
        _ => unreachable!(),
    }
}

/// How to derive the type of `OpAccessChain` instructions from Naga IR.
///
/// Most of the time, we compile Naga IR to SPIR-V instructions whose result
/// types are simply the direct SPIR-V analog of the Naga IR's. But in some
/// cases, the Naga IR and SPIR-V types need to diverge.
///
/// This enum specifies how [`BlockContext::write_access_chain`] should
/// choose a SPIR-V result type for the `OpAccessChain` it generates, based on
/// the type of the given Naga IR [`Expression`] it's generating code for.
///
/// [`Expression`]: crate::Expression
enum AccessTypeAdjustment {
    /// No adjustment needed: the SPIR-V type should be the direct
    /// analog of the Naga IR expression type.
    ///
    /// For most access chains, this is the right thing: the Naga IR access
    /// expression produces a [`Pointer`] to the element / component, and the
    /// SPIR-V `OpAccessChain` instruction does the same.
    ///
    /// [`Pointer`]: crate::TypeInner::Pointer
    None,

    /// The SPIR-V type should be an `OpPointer` to the direct analog of the
    /// Naga IR expression's type.
    ///
    /// This is necessary for indexing binding arrays in the [`Handle`] address
    /// space:
    ///
    /// - In Naga IR, referencing a binding array [`GlobalVariable`] in the
    ///   [`Handle`] address space produces a value of type [`BindingArray`],
    ///   not a pointer to such. And [`Access`] and [`AccessIndex`] expressions
    ///   operate on handle binding arrays by value, and produce handle values,
    ///   not pointers.
    ///
    /// - In SPIR-V, a binding array `OpVariable` produces a pointer to an
    ///   array, and `OpAccessChain` instructions operate on pointers,
    ///   regardless of whether the elements are opaque types or not.
    ///
    /// See also the documentation for [`BindingArray`].
    ///
    /// [`Handle`]: crate::AddressSpace::Handle
    /// [`GlobalVariable`]: crate::GlobalVariable
    /// [`BindingArray`]: crate::TypeInner::BindingArray
    /// [`Access`]: crate::Expression::Access
    /// [`AccessIndex`]: crate::Expression::AccessIndex
    IntroducePointer(spirv::StorageClass),
}

/// The results of emitting code for a left-hand-side expression.
///
/// On success, `write_access_chain` returns one of these.
enum ExpressionPointer {
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

/// The termination statement to be added to the end of the block
enum BlockExit {
    /// Generates an OpReturn (void return)
    Return,
    /// Generates an OpBranch to the specified block
    Branch {
        /// The branch target block
        target: Word,
    },
    /// Translates a loop `break if` into an `OpBranchConditional` to the
    /// merge block if true (the merge block is passed through [`LoopContext::break_id`]
    /// or else to the loop header (passed through [`preamble_id`])
    ///
    /// [`preamble_id`]: Self::BreakIf::preamble_id
    BreakIf {
        /// The condition of the `break if`
        condition: Handle<crate::Expression>,
        /// The loop header block id
        preamble_id: Word,
    },
}

/// What code generation did with a provided [`BlockExit`] value.
///
/// A function that accepts a [`BlockExit`] argument should return a value of
/// this type, to indicate whether the code it generated ended up using the
/// provided exit, or ignored it and did a non-local exit of some other kind
/// (say, [`Break`] or [`Continue`]). Some callers must use this information to
/// decide whether to generate the target block at all.
///
/// [`Break`]: Statement::Break
/// [`Continue`]: Statement::Continue
#[must_use]
enum BlockExitDisposition {
    /// The generated code used the provided `BlockExit` value. If it included a
    /// block label, the caller should be sure to actually emit the block it
    /// refers to.
    Used,

    /// The generated code did not use the provided `BlockExit` value. If it
    /// included a block label, the caller should not bother to actually emit
    /// the block it refers to, unless it knows the block is needed for
    /// something else.
    Discarded,
}

#[derive(Clone, Copy, Default)]
struct LoopContext {
    continuing_id: Option<Word>,
    break_id: Option<Word>,
}

#[derive(Debug)]
pub(crate) struct DebugInfoInner<'a> {
    pub source_code: &'a str,
    pub source_file_id: Word,
}

impl Writer {
    // Flip Y coordinate to adjust for coordinate space difference
    // between SPIR-V and our IR.
    // The `position_id` argument is a pointer to a `vecN<f32>`,
    // whose `y` component we will negate.
    fn write_epilogue_position_y_flip(
        &mut self,
        position_id: Word,
        body: &mut Vec<Instruction>,
    ) -> Result<(), Error> {
        let float_ptr_type_id = self.get_type_id(LookupType::Local(LocalType::LocalPointer {
            base: NumericType::Scalar(crate::Scalar::F32),
            class: spirv::StorageClass::Output,
        }));
        let index_y_id = self.get_index_constant(1);
        let access_id = self.id_gen.next();
        body.push(Instruction::access_chain(
            float_ptr_type_id,
            access_id,
            position_id,
            &[index_y_id],
        ));

        let float_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
            NumericType::Scalar(crate::Scalar::F32),
        )));
        let load_id = self.id_gen.next();
        body.push(Instruction::load(float_type_id, load_id, access_id, None));

        let neg_id = self.id_gen.next();
        body.push(Instruction::unary(
            spirv::Op::FNegate,
            float_type_id,
            neg_id,
            load_id,
        ));

        body.push(Instruction::store(access_id, neg_id, None));
        Ok(())
    }

    // Clamp fragment depth between 0 and 1.
    fn write_epilogue_frag_depth_clamp(
        &mut self,
        frag_depth_id: Word,
        body: &mut Vec<Instruction>,
    ) -> Result<(), Error> {
        let float_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
            NumericType::Scalar(crate::Scalar::F32),
        )));
        let zero_scalar_id = self.get_constant_scalar(crate::Literal::F32(0.0));
        let one_scalar_id = self.get_constant_scalar(crate::Literal::F32(1.0));

        let original_id = self.id_gen.next();
        body.push(Instruction::load(
            float_type_id,
            original_id,
            frag_depth_id,
            None,
        ));

        let clamp_id = self.id_gen.next();
        body.push(Instruction::ext_inst(
            self.gl450_ext_inst_id,
            spirv::GLOp::FClamp,
            float_type_id,
            clamp_id,
            &[original_id, zero_scalar_id, one_scalar_id],
        ));

        body.push(Instruction::store(frag_depth_id, clamp_id, None));
        Ok(())
    }

    fn write_entry_point_return(
        &mut self,
        value_id: Word,
        ir_result: &crate::FunctionResult,
        result_members: &[ResultMember],
        body: &mut Vec<Instruction>,
    ) -> Result<(), Error> {
        for (index, res_member) in result_members.iter().enumerate() {
            let member_value_id = match ir_result.binding {
                Some(_) => value_id,
                None => {
                    let member_value_id = self.id_gen.next();
                    body.push(Instruction::composite_extract(
                        res_member.type_id,
                        member_value_id,
                        value_id,
                        &[index as u32],
                    ));
                    member_value_id
                }
            };

            body.push(Instruction::store(res_member.id, member_value_id, None));

            match res_member.built_in {
                Some(crate::BuiltIn::Position { .. })
                    if self.flags.contains(WriterFlags::ADJUST_COORDINATE_SPACE) =>
                {
                    self.write_epilogue_position_y_flip(res_member.id, body)?;
                }
                Some(crate::BuiltIn::FragDepth)
                    if self.flags.contains(WriterFlags::CLAMP_FRAG_DEPTH) =>
                {
                    self.write_epilogue_frag_depth_clamp(res_member.id, body)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl<'w> BlockContext<'w> {
    /// Cache an expression for a value.
    pub(super) fn cache_expression_value(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        let is_named_expression = self
            .ir_function
            .named_expressions
            .contains_key(&expr_handle);

        if self.fun_info[expr_handle].ref_count == 0 && !is_named_expression {
            return Ok(());
        }

        let result_type_id = self.get_expression_type_id(&self.fun_info[expr_handle].ty);
        let id = match self.ir_function.expressions[expr_handle] {
            crate::Expression::Literal(literal) => self.writer.get_constant_scalar(literal),
            crate::Expression::Constant(handle) => {
                let init = self.ir_module.constants[handle].init;
                self.writer.constant_ids[init]
            }
            crate::Expression::Override(_) => return Err(Error::Override),
            crate::Expression::ZeroValue(_) => self.writer.get_constant_null(result_type_id),
            crate::Expression::Compose { ty, ref components } => {
                self.temp_list.clear();
                if self.expression_constness.is_const(expr_handle) {
                    self.temp_list.extend(
                        crate::proc::flatten_compose(
                            ty,
                            components,
                            &self.ir_function.expressions,
                            &self.ir_module.types,
                        )
                        .map(|component| self.cached[component]),
                    );
                    self.writer
                        .get_constant_composite(LookupType::Handle(ty), &self.temp_list)
                } else {
                    self.temp_list
                        .extend(components.iter().map(|&component| self.cached[component]));

                    let id = self.gen_id();
                    block.body.push(Instruction::composite_construct(
                        result_type_id,
                        id,
                        &self.temp_list,
                    ));
                    id
                }
            }
            crate::Expression::Splat { size, value } => {
                let value_id = self.cached[value];
                let components = &[value_id; 4][..size as usize];

                if self.expression_constness.is_const(expr_handle) {
                    let ty = self
                        .writer
                        .get_expression_lookup_type(&self.fun_info[expr_handle].ty);
                    self.writer.get_constant_composite(ty, components)
                } else {
                    let id = self.gen_id();
                    block.body.push(Instruction::composite_construct(
                        result_type_id,
                        id,
                        components,
                    ));
                    id
                }
            }
            crate::Expression::Access { base, index } => {
                let base_ty_inner = self.fun_info[base].ty.inner_with(&self.ir_module.types);
                match *base_ty_inner {
                    crate::TypeInner::Pointer { .. } | crate::TypeInner::ValuePointer { .. } => {
                        // When we have a chain of `Access` and `AccessIndex` expressions
                        // operating on pointers, we want to generate a single
                        // `OpAccessChain` instruction for the whole chain. Put off
                        // generating any code for this until we find the `Expression`
                        // that actually dereferences the pointer.
                        0
                    }
                    _ if self.function.spilled_accesses.contains(base) => {
                        // As far as Naga IR is concerned, this expression does not yield
                        // a pointer (we just checked, above), but this backend spilled it
                        // to a temporary variable, so SPIR-V thinks we're accessing it
                        // via a pointer.

                        // Since the base expression was spilled, mark this access to it
                        // as spilled, too.
                        self.function.spilled_accesses.insert(expr_handle);
                        self.maybe_access_spilled_composite(expr_handle, block, result_type_id)?
                    }
                    crate::TypeInner::Vector { .. } => {
                        self.write_vector_access(expr_handle, base, index, block)?
                    }
                    crate::TypeInner::Array { .. } | crate::TypeInner::Matrix { .. } => {
                        // See if `index` is known at compile time.
                        match GuardedIndex::from_expression(
                            index,
                            &self.ir_function.expressions,
                            self.ir_module,
                        ) {
                            GuardedIndex::Known(value) => {
                                // If `index` is known and in bounds, we can just use
                                // `OpCompositeExtract`.
                                //
                                // At the moment, validation rejects programs if this
                                // index is out of bounds, so we don't need bounds checks.
                                // However, that rejection is incorrect, since WGSL says
                                // that `let` bindings are not constant expressions
                                // (#6396). So eventually we will need to emulate bounds
                                // checks here.
                                let id = self.gen_id();
                                let base_id = self.cached[base];
                                block.body.push(Instruction::composite_extract(
                                    result_type_id,
                                    id,
                                    base_id,
                                    &[value],
                                ));
                                id
                            }
                            GuardedIndex::Expression(_) => {
                                // We are subscripting an array or matrix that is not
                                // behind a pointer, using an index computed at runtime.
                                // SPIR-V has no instructions that do this, so the best we
                                // can do is spill the value to a new temporary variable,
                                // at which point we can get a pointer to that and just
                                // use `OpAccessChain` in the usual way.
                                self.spill_to_internal_variable(base, block);

                                // Since the base was spilled, mark this access to it as
                                // spilled, too.
                                self.function.spilled_accesses.insert(expr_handle);
                                self.maybe_access_spilled_composite(
                                    expr_handle,
                                    block,
                                    result_type_id,
                                )?
                            }
                        }
                    }
                    crate::TypeInner::BindingArray {
                        base: binding_type, ..
                    } => {
                        // Only binding arrays in the `Handle` address space will take
                        // this path, since we handled the `Pointer` case above.
                        let result_id = match self.write_access_chain(
                            expr_handle,
                            block,
                            AccessTypeAdjustment::IntroducePointer(
                                spirv::StorageClass::UniformConstant,
                            ),
                        )? {
                            ExpressionPointer::Ready { pointer_id } => pointer_id,
                            ExpressionPointer::Conditional { .. } => {
                                return Err(Error::FeatureNotImplemented(
                                    "Texture array out-of-bounds handling",
                                ));
                            }
                        };

                        let binding_type_id = self.get_type_id(LookupType::Handle(binding_type));

                        let load_id = self.gen_id();
                        block.body.push(Instruction::load(
                            binding_type_id,
                            load_id,
                            result_id,
                            None,
                        ));

                        // Subsequent image operations require the image/sampler to be decorated as NonUniform
                        // if the image/sampler binding array was accessed with a non-uniform index
                        // see VUID-RuntimeSpirv-NonUniform-06274
                        if self.fun_info[index].uniformity.non_uniform_result.is_some() {
                            self.writer
                                .decorate_non_uniform_binding_array_access(load_id)?;
                        }

                        load_id
                    }
                    ref other => {
                        log::error!(
                            "Unable to access base {:?} of type {:?}",
                            self.ir_function.expressions[base],
                            other
                        );
                        return Err(Error::Validation(
                            "only vectors and arrays may be dynamically indexed by value",
                        ));
                    }
                }
            }
            crate::Expression::AccessIndex { base, index } => {
                match *self.fun_info[base].ty.inner_with(&self.ir_module.types) {
                    crate::TypeInner::Pointer { .. } | crate::TypeInner::ValuePointer { .. } => {
                        // When we have a chain of `Access` and `AccessIndex` expressions
                        // operating on pointers, we want to generate a single
                        // `OpAccessChain` instruction for the whole chain. Put off
                        // generating any code for this until we find the `Expression`
                        // that actually dereferences the pointer.
                        0
                    }
                    _ if self.function.spilled_accesses.contains(base) => {
                        // As far as Naga IR is concerned, this expression does not yield
                        // a pointer (we just checked, above), but this backend spilled it
                        // to a temporary variable, so SPIR-V thinks we're accessing it
                        // via a pointer.

                        // Since the base expression was spilled, mark this access to it
                        // as spilled, too.
                        self.function.spilled_accesses.insert(expr_handle);
                        self.maybe_access_spilled_composite(expr_handle, block, result_type_id)?
                    }
                    crate::TypeInner::Vector { .. }
                    | crate::TypeInner::Matrix { .. }
                    | crate::TypeInner::Array { .. }
                    | crate::TypeInner::Struct { .. } => {
                        // We never need bounds checks here: dynamically sized arrays can
                        // only appear behind pointers, and are thus handled by the
                        // `is_intermediate` case above. Everything else's size is
                        // statically known and checked in validation.
                        let id = self.gen_id();
                        let base_id = self.cached[base];
                        block.body.push(Instruction::composite_extract(
                            result_type_id,
                            id,
                            base_id,
                            &[index],
                        ));
                        id
                    }
                    crate::TypeInner::BindingArray {
                        base: binding_type, ..
                    } => {
                        // Only binding arrays in the `Handle` address space will take
                        // this path, since we handled the `Pointer` case above.
                        let result_id = match self.write_access_chain(
                            expr_handle,
                            block,
                            AccessTypeAdjustment::IntroducePointer(
                                spirv::StorageClass::UniformConstant,
                            ),
                        )? {
                            ExpressionPointer::Ready { pointer_id } => pointer_id,
                            ExpressionPointer::Conditional { .. } => {
                                return Err(Error::FeatureNotImplemented(
                                    "Texture array out-of-bounds handling",
                                ));
                            }
                        };

                        let binding_type_id = self.get_type_id(LookupType::Handle(binding_type));

                        let load_id = self.gen_id();
                        block.body.push(Instruction::load(
                            binding_type_id,
                            load_id,
                            result_id,
                            None,
                        ));

                        load_id
                    }
                    ref other => {
                        log::error!("Unable to access index of {:?}", other);
                        return Err(Error::FeatureNotImplemented("access index for type"));
                    }
                }
            }
            crate::Expression::GlobalVariable(handle) => {
                self.writer.global_variables[handle].access_id
            }
            crate::Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let vector_id = self.cached[vector];
                self.temp_list.clear();
                for &sc in pattern[..size as usize].iter() {
                    self.temp_list.push(sc as Word);
                }
                let id = self.gen_id();
                block.body.push(Instruction::vector_shuffle(
                    result_type_id,
                    id,
                    vector_id,
                    vector_id,
                    &self.temp_list,
                ));
                id
            }
            crate::Expression::Unary { op, expr } => {
                let id = self.gen_id();
                let expr_id = self.cached[expr];
                let expr_ty_inner = self.fun_info[expr].ty.inner_with(&self.ir_module.types);

                let spirv_op = match op {
                    crate::UnaryOperator::Negate => match expr_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Float) => spirv::Op::FNegate,
                        Some(crate::ScalarKind::Sint) => spirv::Op::SNegate,
                        _ => return Err(Error::Validation("Unexpected kind for negation")),
                    },
                    crate::UnaryOperator::LogicalNot => spirv::Op::LogicalNot,
                    crate::UnaryOperator::BitwiseNot => spirv::Op::Not,
                };

                block
                    .body
                    .push(Instruction::unary(spirv_op, result_type_id, id, expr_id));
                id
            }
            crate::Expression::Binary { op, left, right } => {
                let id = self.gen_id();
                let left_id = self.cached[left];
                let right_id = self.cached[right];

                let left_ty_inner = self.fun_info[left].ty.inner_with(&self.ir_module.types);
                let right_ty_inner = self.fun_info[right].ty.inner_with(&self.ir_module.types);

                let left_dimension = get_dimension(left_ty_inner);
                let right_dimension = get_dimension(right_ty_inner);

                let mut reverse_operands = false;

                let spirv_op = match op {
                    crate::BinaryOperator::Add => match *left_ty_inner {
                        crate::TypeInner::Scalar(scalar)
                        | crate::TypeInner::Vector { scalar, .. } => match scalar.kind {
                            crate::ScalarKind::Float => spirv::Op::FAdd,
                            _ => spirv::Op::IAdd,
                        },
                        crate::TypeInner::Matrix {
                            columns,
                            rows,
                            scalar,
                        } => {
                            self.write_matrix_matrix_column_op(
                                block,
                                id,
                                result_type_id,
                                left_id,
                                right_id,
                                columns,
                                rows,
                                scalar.width,
                                spirv::Op::FAdd,
                            );

                            self.cached[expr_handle] = id;
                            return Ok(());
                        }
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Subtract => match *left_ty_inner {
                        crate::TypeInner::Scalar(scalar)
                        | crate::TypeInner::Vector { scalar, .. } => match scalar.kind {
                            crate::ScalarKind::Float => spirv::Op::FSub,
                            _ => spirv::Op::ISub,
                        },
                        crate::TypeInner::Matrix {
                            columns,
                            rows,
                            scalar,
                        } => {
                            self.write_matrix_matrix_column_op(
                                block,
                                id,
                                result_type_id,
                                left_id,
                                right_id,
                                columns,
                                rows,
                                scalar.width,
                                spirv::Op::FSub,
                            );

                            self.cached[expr_handle] = id;
                            return Ok(());
                        }
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Multiply => match (left_dimension, right_dimension) {
                        (Dimension::Scalar, Dimension::Vector) => {
                            self.write_vector_scalar_mult(
                                block,
                                id,
                                result_type_id,
                                right_id,
                                left_id,
                                right_ty_inner,
                            );

                            self.cached[expr_handle] = id;
                            return Ok(());
                        }
                        (Dimension::Vector, Dimension::Scalar) => {
                            self.write_vector_scalar_mult(
                                block,
                                id,
                                result_type_id,
                                left_id,
                                right_id,
                                left_ty_inner,
                            );

                            self.cached[expr_handle] = id;
                            return Ok(());
                        }
                        (Dimension::Vector, Dimension::Matrix) => spirv::Op::VectorTimesMatrix,
                        (Dimension::Matrix, Dimension::Scalar) => spirv::Op::MatrixTimesScalar,
                        (Dimension::Scalar, Dimension::Matrix) => {
                            reverse_operands = true;
                            spirv::Op::MatrixTimesScalar
                        }
                        (Dimension::Matrix, Dimension::Vector) => spirv::Op::MatrixTimesVector,
                        (Dimension::Matrix, Dimension::Matrix) => spirv::Op::MatrixTimesMatrix,
                        (Dimension::Vector, Dimension::Vector)
                        | (Dimension::Scalar, Dimension::Scalar)
                            if left_ty_inner.scalar_kind() == Some(crate::ScalarKind::Float) =>
                        {
                            spirv::Op::FMul
                        }
                        (Dimension::Vector, Dimension::Vector)
                        | (Dimension::Scalar, Dimension::Scalar) => spirv::Op::IMul,
                    },
                    crate::BinaryOperator::Divide => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SDiv,
                        Some(crate::ScalarKind::Uint) => spirv::Op::UDiv,
                        Some(crate::ScalarKind::Float) => spirv::Op::FDiv,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Modulo => match left_ty_inner.scalar_kind() {
                        // TODO: handle undefined behavior
                        // if right == 0 return 0
                        // if left == min(type_of(left)) && right == -1 return 0
                        Some(crate::ScalarKind::Sint) => spirv::Op::SRem,
                        // TODO: handle undefined behavior
                        // if right == 0 return 0
                        Some(crate::ScalarKind::Uint) => spirv::Op::UMod,
                        // TODO: handle undefined behavior
                        // if right == 0 return ? see https://github.com/gpuweb/gpuweb/issues/2798
                        Some(crate::ScalarKind::Float) => spirv::Op::FRem,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Equal => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint | crate::ScalarKind::Uint) => {
                            spirv::Op::IEqual
                        }
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdEqual,
                        Some(crate::ScalarKind::Bool) => spirv::Op::LogicalEqual,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::NotEqual => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint | crate::ScalarKind::Uint) => {
                            spirv::Op::INotEqual
                        }
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdNotEqual,
                        Some(crate::ScalarKind::Bool) => spirv::Op::LogicalNotEqual,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Less => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SLessThan,
                        Some(crate::ScalarKind::Uint) => spirv::Op::ULessThan,
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdLessThan,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::LessEqual => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SLessThanEqual,
                        Some(crate::ScalarKind::Uint) => spirv::Op::ULessThanEqual,
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdLessThanEqual,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Greater => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SGreaterThan,
                        Some(crate::ScalarKind::Uint) => spirv::Op::UGreaterThan,
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdGreaterThan,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::GreaterEqual => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SGreaterThanEqual,
                        Some(crate::ScalarKind::Uint) => spirv::Op::UGreaterThanEqual,
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdGreaterThanEqual,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::And => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Bool) => spirv::Op::LogicalAnd,
                        _ => spirv::Op::BitwiseAnd,
                    },
                    crate::BinaryOperator::ExclusiveOr => spirv::Op::BitwiseXor,
                    crate::BinaryOperator::InclusiveOr => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Bool) => spirv::Op::LogicalOr,
                        _ => spirv::Op::BitwiseOr,
                    },
                    crate::BinaryOperator::LogicalAnd => spirv::Op::LogicalAnd,
                    crate::BinaryOperator::LogicalOr => spirv::Op::LogicalOr,
                    crate::BinaryOperator::ShiftLeft => spirv::Op::ShiftLeftLogical,
                    crate::BinaryOperator::ShiftRight => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::ShiftRightArithmetic,
                        Some(crate::ScalarKind::Uint) => spirv::Op::ShiftRightLogical,
                        _ => unimplemented!(),
                    },
                };

                block.body.push(Instruction::binary(
                    spirv_op,
                    result_type_id,
                    id,
                    if reverse_operands { right_id } else { left_id },
                    if reverse_operands { left_id } else { right_id },
                ));
                id
            }
            crate::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                use crate::MathFunction as Mf;
                enum MathOp {
                    Ext(spirv::GLOp),
                    Custom(Instruction),
                }

                let arg0_id = self.cached[arg];
                let arg_ty = self.fun_info[arg].ty.inner_with(&self.ir_module.types);
                let arg_scalar_kind = arg_ty.scalar_kind();
                let arg1_id = match arg1 {
                    Some(handle) => self.cached[handle],
                    None => 0,
                };
                let arg2_id = match arg2 {
                    Some(handle) => self.cached[handle],
                    None => 0,
                };
                let arg3_id = match arg3 {
                    Some(handle) => self.cached[handle],
                    None => 0,
                };

                let id = self.gen_id();
                let math_op = match fun {
                    // comparison
                    Mf::Abs => {
                        match arg_scalar_kind {
                            Some(crate::ScalarKind::Float) => MathOp::Ext(spirv::GLOp::FAbs),
                            Some(crate::ScalarKind::Sint) => MathOp::Ext(spirv::GLOp::SAbs),
                            Some(crate::ScalarKind::Uint) => {
                                MathOp::Custom(Instruction::unary(
                                    spirv::Op::CopyObject, // do nothing
                                    result_type_id,
                                    id,
                                    arg0_id,
                                ))
                            }
                            other => unimplemented!("Unexpected abs({:?})", other),
                        }
                    }
                    Mf::Min => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Float) => spirv::GLOp::FMin,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::SMin,
                        Some(crate::ScalarKind::Uint) => spirv::GLOp::UMin,
                        other => unimplemented!("Unexpected min({:?})", other),
                    }),
                    Mf::Max => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Float) => spirv::GLOp::FMax,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::SMax,
                        Some(crate::ScalarKind::Uint) => spirv::GLOp::UMax,
                        other => unimplemented!("Unexpected max({:?})", other),
                    }),
                    Mf::Clamp => match arg_scalar_kind {
                        // Clamp is undefined if min > max. In practice this means it can use a median-of-three
                        // instruction to determine the value. This is fine according to the WGSL spec for float
                        // clamp, but integer clamp _must_ use min-max. As such we write out min/max.
                        Some(crate::ScalarKind::Float) => MathOp::Ext(spirv::GLOp::FClamp),
                        Some(_) => {
                            let (min_op, max_op) = match arg_scalar_kind {
                                Some(crate::ScalarKind::Sint) => {
                                    (spirv::GLOp::SMin, spirv::GLOp::SMax)
                                }
                                Some(crate::ScalarKind::Uint) => {
                                    (spirv::GLOp::UMin, spirv::GLOp::UMax)
                                }
                                _ => unreachable!(),
                            };

                            let max_id = self.gen_id();
                            block.body.push(Instruction::ext_inst(
                                self.writer.gl450_ext_inst_id,
                                max_op,
                                result_type_id,
                                max_id,
                                &[arg0_id, arg1_id],
                            ));

                            MathOp::Custom(Instruction::ext_inst(
                                self.writer.gl450_ext_inst_id,
                                min_op,
                                result_type_id,
                                id,
                                &[max_id, arg2_id],
                            ))
                        }
                        other => unimplemented!("Unexpected max({:?})", other),
                    },
                    Mf::Saturate => {
                        let (maybe_size, scalar) = match *arg_ty {
                            crate::TypeInner::Vector { size, scalar } => (Some(size), scalar),
                            crate::TypeInner::Scalar(scalar) => (None, scalar),
                            ref other => unimplemented!("Unexpected saturate({:?})", other),
                        };
                        let scalar = crate::Scalar::float(scalar.width);
                        let mut arg1_id = self.writer.get_constant_scalar_with(0, scalar)?;
                        let mut arg2_id = self.writer.get_constant_scalar_with(1, scalar)?;

                        if let Some(size) = maybe_size {
                            let ty =
                                LocalType::Numeric(NumericType::Vector { size, scalar }).into();

                            self.temp_list.clear();
                            self.temp_list.resize(size as _, arg1_id);

                            arg1_id = self.writer.get_constant_composite(ty, &self.temp_list);

                            self.temp_list.fill(arg2_id);

                            arg2_id = self.writer.get_constant_composite(ty, &self.temp_list);
                        }

                        MathOp::Custom(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            spirv::GLOp::FClamp,
                            result_type_id,
                            id,
                            &[arg0_id, arg1_id, arg2_id],
                        ))
                    }
                    // trigonometry
                    Mf::Sin => MathOp::Ext(spirv::GLOp::Sin),
                    Mf::Sinh => MathOp::Ext(spirv::GLOp::Sinh),
                    Mf::Asin => MathOp::Ext(spirv::GLOp::Asin),
                    Mf::Cos => MathOp::Ext(spirv::GLOp::Cos),
                    Mf::Cosh => MathOp::Ext(spirv::GLOp::Cosh),
                    Mf::Acos => MathOp::Ext(spirv::GLOp::Acos),
                    Mf::Tan => MathOp::Ext(spirv::GLOp::Tan),
                    Mf::Tanh => MathOp::Ext(spirv::GLOp::Tanh),
                    Mf::Atan => MathOp::Ext(spirv::GLOp::Atan),
                    Mf::Atan2 => MathOp::Ext(spirv::GLOp::Atan2),
                    Mf::Asinh => MathOp::Ext(spirv::GLOp::Asinh),
                    Mf::Acosh => MathOp::Ext(spirv::GLOp::Acosh),
                    Mf::Atanh => MathOp::Ext(spirv::GLOp::Atanh),
                    Mf::Radians => MathOp::Ext(spirv::GLOp::Radians),
                    Mf::Degrees => MathOp::Ext(spirv::GLOp::Degrees),
                    // decomposition
                    Mf::Ceil => MathOp::Ext(spirv::GLOp::Ceil),
                    Mf::Round => MathOp::Ext(spirv::GLOp::RoundEven),
                    Mf::Floor => MathOp::Ext(spirv::GLOp::Floor),
                    Mf::Fract => MathOp::Ext(spirv::GLOp::Fract),
                    Mf::Trunc => MathOp::Ext(spirv::GLOp::Trunc),
                    Mf::Modf => MathOp::Ext(spirv::GLOp::ModfStruct),
                    Mf::Frexp => MathOp::Ext(spirv::GLOp::FrexpStruct),
                    Mf::Ldexp => MathOp::Ext(spirv::GLOp::Ldexp),
                    // geometry
                    Mf::Dot => match *self.fun_info[arg].ty.inner_with(&self.ir_module.types) {
                        crate::TypeInner::Vector {
                            scalar:
                                crate::Scalar {
                                    kind: crate::ScalarKind::Float,
                                    ..
                                },
                            ..
                        } => MathOp::Custom(Instruction::binary(
                            spirv::Op::Dot,
                            result_type_id,
                            id,
                            arg0_id,
                            arg1_id,
                        )),
                        // TODO: consider using integer dot product if VK_KHR_shader_integer_dot_product is available
                        crate::TypeInner::Vector { size, .. } => {
                            self.write_dot_product(
                                id,
                                result_type_id,
                                arg0_id,
                                arg1_id,
                                size as u32,
                                block,
                            );
                            self.cached[expr_handle] = id;
                            return Ok(());
                        }
                        _ => unreachable!(
                            "Correct TypeInner for dot product should be already validated"
                        ),
                    },
                    Mf::Outer => MathOp::Custom(Instruction::binary(
                        spirv::Op::OuterProduct,
                        result_type_id,
                        id,
                        arg0_id,
                        arg1_id,
                    )),
                    Mf::Cross => MathOp::Ext(spirv::GLOp::Cross),
                    Mf::Distance => MathOp::Ext(spirv::GLOp::Distance),
                    Mf::Length => MathOp::Ext(spirv::GLOp::Length),
                    Mf::Normalize => MathOp::Ext(spirv::GLOp::Normalize),
                    Mf::FaceForward => MathOp::Ext(spirv::GLOp::FaceForward),
                    Mf::Reflect => MathOp::Ext(spirv::GLOp::Reflect),
                    Mf::Refract => MathOp::Ext(spirv::GLOp::Refract),
                    // exponent
                    Mf::Exp => MathOp::Ext(spirv::GLOp::Exp),
                    Mf::Exp2 => MathOp::Ext(spirv::GLOp::Exp2),
                    Mf::Log => MathOp::Ext(spirv::GLOp::Log),
                    Mf::Log2 => MathOp::Ext(spirv::GLOp::Log2),
                    Mf::Pow => MathOp::Ext(spirv::GLOp::Pow),
                    // computational
                    Mf::Sign => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Float) => spirv::GLOp::FSign,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::SSign,
                        other => unimplemented!("Unexpected sign({:?})", other),
                    }),
                    Mf::Fma => MathOp::Ext(spirv::GLOp::Fma),
                    Mf::Mix => {
                        let selector = arg2.unwrap();
                        let selector_ty =
                            self.fun_info[selector].ty.inner_with(&self.ir_module.types);
                        match (arg_ty, selector_ty) {
                            // if the selector is a scalar, we need to splat it
                            (
                                &crate::TypeInner::Vector { size, .. },
                                &crate::TypeInner::Scalar(scalar),
                            ) => {
                                let selector_type_id = self.get_type_id(LookupType::Local(
                                    LocalType::Numeric(NumericType::Vector { size, scalar }),
                                ));
                                self.temp_list.clear();
                                self.temp_list.resize(size as usize, arg2_id);

                                let selector_id = self.gen_id();
                                block.body.push(Instruction::composite_construct(
                                    selector_type_id,
                                    selector_id,
                                    &self.temp_list,
                                ));

                                MathOp::Custom(Instruction::ext_inst(
                                    self.writer.gl450_ext_inst_id,
                                    spirv::GLOp::FMix,
                                    result_type_id,
                                    id,
                                    &[arg0_id, arg1_id, selector_id],
                                ))
                            }
                            _ => MathOp::Ext(spirv::GLOp::FMix),
                        }
                    }
                    Mf::Step => MathOp::Ext(spirv::GLOp::Step),
                    Mf::SmoothStep => MathOp::Ext(spirv::GLOp::SmoothStep),
                    Mf::Sqrt => MathOp::Ext(spirv::GLOp::Sqrt),
                    Mf::InverseSqrt => MathOp::Ext(spirv::GLOp::InverseSqrt),
                    Mf::Inverse => MathOp::Ext(spirv::GLOp::MatrixInverse),
                    Mf::Transpose => MathOp::Custom(Instruction::unary(
                        spirv::Op::Transpose,
                        result_type_id,
                        id,
                        arg0_id,
                    )),
                    Mf::Determinant => MathOp::Ext(spirv::GLOp::Determinant),
                    Mf::ReverseBits => MathOp::Custom(Instruction::unary(
                        spirv::Op::BitReverse,
                        result_type_id,
                        id,
                        arg0_id,
                    )),
                    Mf::CountTrailingZeros => {
                        let uint_id = match *arg_ty {
                            crate::TypeInner::Vector { size, scalar } => {
                                let ty =
                                    LocalType::Numeric(NumericType::Vector { size, scalar }).into();

                                self.temp_list.clear();
                                self.temp_list.resize(
                                    size as _,
                                    self.writer
                                        .get_constant_scalar_with(scalar.width * 8, scalar)?,
                                );

                                self.writer.get_constant_composite(ty, &self.temp_list)
                            }
                            crate::TypeInner::Scalar(scalar) => self
                                .writer
                                .get_constant_scalar_with(scalar.width * 8, scalar)?,
                            _ => unreachable!(),
                        };

                        let lsb_id = self.gen_id();
                        block.body.push(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            spirv::GLOp::FindILsb,
                            result_type_id,
                            lsb_id,
                            &[arg0_id],
                        ));

                        MathOp::Custom(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            spirv::GLOp::UMin,
                            result_type_id,
                            id,
                            &[uint_id, lsb_id],
                        ))
                    }
                    Mf::CountLeadingZeros => {
                        let (int_type_id, int_id, width) = match *arg_ty {
                            crate::TypeInner::Vector { size, scalar } => {
                                let ty =
                                    LocalType::Numeric(NumericType::Vector { size, scalar }).into();

                                self.temp_list.clear();
                                self.temp_list.resize(
                                    size as _,
                                    self.writer
                                        .get_constant_scalar_with(scalar.width * 8 - 1, scalar)?,
                                );

                                (
                                    self.get_type_id(ty),
                                    self.writer.get_constant_composite(ty, &self.temp_list),
                                    scalar.width,
                                )
                            }
                            crate::TypeInner::Scalar(scalar) => (
                                self.get_type_id(LookupType::Local(LocalType::Numeric(
                                    NumericType::Scalar(scalar),
                                ))),
                                self.writer
                                    .get_constant_scalar_with(scalar.width * 8 - 1, scalar)?,
                                scalar.width,
                            ),
                            _ => unreachable!(),
                        };

                        if width != 4 {
                            unreachable!("This is validated out until a polyfill is implemented. https://github.com/gfx-rs/wgpu/issues/5276");
                        };

                        let msb_id = self.gen_id();
                        block.body.push(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            if width != 4 {
                                spirv::GLOp::FindILsb
                            } else {
                                spirv::GLOp::FindUMsb
                            },
                            int_type_id,
                            msb_id,
                            &[arg0_id],
                        ));

                        MathOp::Custom(Instruction::binary(
                            spirv::Op::ISub,
                            result_type_id,
                            id,
                            int_id,
                            msb_id,
                        ))
                    }
                    Mf::CountOneBits => MathOp::Custom(Instruction::unary(
                        spirv::Op::BitCount,
                        result_type_id,
                        id,
                        arg0_id,
                    )),
                    Mf::ExtractBits => {
                        let op = match arg_scalar_kind {
                            Some(crate::ScalarKind::Uint) => spirv::Op::BitFieldUExtract,
                            Some(crate::ScalarKind::Sint) => spirv::Op::BitFieldSExtract,
                            other => unimplemented!("Unexpected sign({:?})", other),
                        };

                        // The behavior of ExtractBits is undefined when offset + count > bit_width. We need
                        // to first sanitize the offset and count first. If we don't do this, AMD and Intel
                        // will return out-of-spec values if the extracted range is not within the bit width.
                        //
                        // This encodes the exact formula specified by the wgsl spec:
                        // https://gpuweb.github.io/gpuweb/wgsl/#extractBits-unsigned-builtin
                        //
                        // w = sizeof(x) * 8
                        // o = min(offset, w)
                        // tmp = w - o
                        // c = min(count, tmp)
                        //
                        // bitfieldExtract(x, o, c)

                        let bit_width = arg_ty.scalar_width().unwrap() * 8;
                        let width_constant = self
                            .writer
                            .get_constant_scalar(crate::Literal::U32(bit_width as u32));

                        let u32_type = self.get_type_id(LookupType::Local(LocalType::Numeric(
                            NumericType::Scalar(crate::Scalar::U32),
                        )));

                        // o = min(offset, w)
                        let offset_id = self.gen_id();
                        block.body.push(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            spirv::GLOp::UMin,
                            u32_type,
                            offset_id,
                            &[arg1_id, width_constant],
                        ));

                        // tmp = w - o
                        let max_count_id = self.gen_id();
                        block.body.push(Instruction::binary(
                            spirv::Op::ISub,
                            u32_type,
                            max_count_id,
                            width_constant,
                            offset_id,
                        ));

                        // c = min(count, tmp)
                        let count_id = self.gen_id();
                        block.body.push(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            spirv::GLOp::UMin,
                            u32_type,
                            count_id,
                            &[arg2_id, max_count_id],
                        ));

                        MathOp::Custom(Instruction::ternary(
                            op,
                            result_type_id,
                            id,
                            arg0_id,
                            offset_id,
                            count_id,
                        ))
                    }
                    Mf::InsertBits => {
                        // The behavior of InsertBits has the same undefined behavior as ExtractBits.

                        let bit_width = arg_ty.scalar_width().unwrap() * 8;
                        let width_constant = self
                            .writer
                            .get_constant_scalar(crate::Literal::U32(bit_width as u32));

                        let u32_type = self.get_type_id(LookupType::Local(LocalType::Numeric(
                            NumericType::Scalar(crate::Scalar::U32),
                        )));

                        // o = min(offset, w)
                        let offset_id = self.gen_id();
                        block.body.push(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            spirv::GLOp::UMin,
                            u32_type,
                            offset_id,
                            &[arg2_id, width_constant],
                        ));

                        // tmp = w - o
                        let max_count_id = self.gen_id();
                        block.body.push(Instruction::binary(
                            spirv::Op::ISub,
                            u32_type,
                            max_count_id,
                            width_constant,
                            offset_id,
                        ));

                        // c = min(count, tmp)
                        let count_id = self.gen_id();
                        block.body.push(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            spirv::GLOp::UMin,
                            u32_type,
                            count_id,
                            &[arg3_id, max_count_id],
                        ));

                        MathOp::Custom(Instruction::quaternary(
                            spirv::Op::BitFieldInsert,
                            result_type_id,
                            id,
                            arg0_id,
                            arg1_id,
                            offset_id,
                            count_id,
                        ))
                    }
                    Mf::FirstTrailingBit => MathOp::Ext(spirv::GLOp::FindILsb),
                    Mf::FirstLeadingBit => {
                        if arg_ty.scalar_width() == Some(4) {
                            let thing = match arg_scalar_kind {
                                Some(crate::ScalarKind::Uint) => spirv::GLOp::FindUMsb,
                                Some(crate::ScalarKind::Sint) => spirv::GLOp::FindSMsb,
                                other => unimplemented!("Unexpected firstLeadingBit({:?})", other),
                            };
                            MathOp::Ext(thing)
                        } else {
                            unreachable!("This is validated out until a polyfill is implemented. https://github.com/gfx-rs/wgpu/issues/5276");
                        }
                    }
                    Mf::Pack4x8unorm => MathOp::Ext(spirv::GLOp::PackUnorm4x8),
                    Mf::Pack4x8snorm => MathOp::Ext(spirv::GLOp::PackSnorm4x8),
                    Mf::Pack2x16float => MathOp::Ext(spirv::GLOp::PackHalf2x16),
                    Mf::Pack2x16unorm => MathOp::Ext(spirv::GLOp::PackUnorm2x16),
                    Mf::Pack2x16snorm => MathOp::Ext(spirv::GLOp::PackSnorm2x16),
                    fun @ (Mf::Pack4xI8 | Mf::Pack4xU8) => {
                        let (int_type, is_signed) = match fun {
                            Mf::Pack4xI8 => (crate::ScalarKind::Sint, true),
                            Mf::Pack4xU8 => (crate::ScalarKind::Uint, false),
                            _ => unreachable!(),
                        };
                        let uint_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
                            NumericType::Scalar(crate::Scalar::U32),
                        )));

                        let int_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
                            NumericType::Scalar(crate::Scalar {
                                kind: int_type,
                                width: 4,
                            }),
                        )));

                        let mut last_instruction = Instruction::new(spirv::Op::Nop);

                        let zero = self.writer.get_constant_scalar(crate::Literal::U32(0));
                        let mut preresult = zero;
                        block
                            .body
                            .reserve(usize::from(VEC_LENGTH) * (2 + usize::from(is_signed)));

                        let eight = self.writer.get_constant_scalar(crate::Literal::U32(8));
                        const VEC_LENGTH: u8 = 4;
                        for i in 0..u32::from(VEC_LENGTH) {
                            let offset =
                                self.writer.get_constant_scalar(crate::Literal::U32(i * 8));
                            let mut extracted = self.gen_id();
                            block.body.push(Instruction::binary(
                                spirv::Op::CompositeExtract,
                                int_type_id,
                                extracted,
                                arg0_id,
                                i,
                            ));
                            if is_signed {
                                let casted = self.gen_id();
                                block.body.push(Instruction::unary(
                                    spirv::Op::Bitcast,
                                    uint_type_id,
                                    casted,
                                    extracted,
                                ));
                                extracted = casted;
                            }
                            let is_last = i == u32::from(VEC_LENGTH - 1);
                            if is_last {
                                last_instruction = Instruction::quaternary(
                                    spirv::Op::BitFieldInsert,
                                    result_type_id,
                                    id,
                                    preresult,
                                    extracted,
                                    offset,
                                    eight,
                                )
                            } else {
                                let new_preresult = self.gen_id();
                                block.body.push(Instruction::quaternary(
                                    spirv::Op::BitFieldInsert,
                                    result_type_id,
                                    new_preresult,
                                    preresult,
                                    extracted,
                                    offset,
                                    eight,
                                ));
                                preresult = new_preresult;
                            }
                        }

                        MathOp::Custom(last_instruction)
                    }
                    Mf::Unpack4x8unorm => MathOp::Ext(spirv::GLOp::UnpackUnorm4x8),
                    Mf::Unpack4x8snorm => MathOp::Ext(spirv::GLOp::UnpackSnorm4x8),
                    Mf::Unpack2x16float => MathOp::Ext(spirv::GLOp::UnpackHalf2x16),
                    Mf::Unpack2x16unorm => MathOp::Ext(spirv::GLOp::UnpackUnorm2x16),
                    Mf::Unpack2x16snorm => MathOp::Ext(spirv::GLOp::UnpackSnorm2x16),
                    fun @ (Mf::Unpack4xI8 | Mf::Unpack4xU8) => {
                        let (int_type, extract_op, is_signed) = match fun {
                            Mf::Unpack4xI8 => {
                                (crate::ScalarKind::Sint, spirv::Op::BitFieldSExtract, true)
                            }
                            Mf::Unpack4xU8 => {
                                (crate::ScalarKind::Uint, spirv::Op::BitFieldUExtract, false)
                            }
                            _ => unreachable!(),
                        };

                        let sint_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
                            NumericType::Scalar(crate::Scalar::I32),
                        )));

                        let eight = self.writer.get_constant_scalar(crate::Literal::U32(8));
                        let int_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
                            NumericType::Scalar(crate::Scalar {
                                kind: int_type,
                                width: 4,
                            }),
                        )));
                        block
                            .body
                            .reserve(usize::from(VEC_LENGTH) * 2 + usize::from(is_signed));
                        let arg_id = if is_signed {
                            let new_arg_id = self.gen_id();
                            block.body.push(Instruction::unary(
                                spirv::Op::Bitcast,
                                sint_type_id,
                                new_arg_id,
                                arg0_id,
                            ));
                            new_arg_id
                        } else {
                            arg0_id
                        };

                        const VEC_LENGTH: u8 = 4;
                        let parts: [_; VEC_LENGTH as usize] =
                            std::array::from_fn(|_| self.gen_id());
                        for (i, part_id) in parts.into_iter().enumerate() {
                            let index = self
                                .writer
                                .get_constant_scalar(crate::Literal::U32(i as u32 * 8));
                            block.body.push(Instruction::ternary(
                                extract_op,
                                int_type_id,
                                part_id,
                                arg_id,
                                index,
                                eight,
                            ));
                        }

                        MathOp::Custom(Instruction::composite_construct(result_type_id, id, &parts))
                    }
                };

                block.body.push(match math_op {
                    MathOp::Ext(op) => Instruction::ext_inst(
                        self.writer.gl450_ext_inst_id,
                        op,
                        result_type_id,
                        id,
                        &[arg0_id, arg1_id, arg2_id, arg3_id][..fun.argument_count()],
                    ),
                    MathOp::Custom(inst) => inst,
                });
                id
            }
            crate::Expression::LocalVariable(variable) => self.function.variables[&variable].id,
            crate::Expression::Load { pointer } => {
                self.write_checked_load(pointer, block, AccessTypeAdjustment::None, result_type_id)?
            }
            crate::Expression::FunctionArgument(index) => self.function.parameter_id(index),
            crate::Expression::CallResult(_)
            | crate::Expression::AtomicResult { .. }
            | crate::Expression::WorkGroupUniformLoadResult { .. }
            | crate::Expression::RayQueryProceedResult
            | crate::Expression::SubgroupBallotResult
            | crate::Expression::SubgroupOperationResult { .. } => self.cached[expr_handle],
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                use crate::ScalarKind as Sk;

                let expr_id = self.cached[expr];
                let (src_scalar, src_size, is_matrix) =
                    match *self.fun_info[expr].ty.inner_with(&self.ir_module.types) {
                        crate::TypeInner::Scalar(scalar) => (scalar, None, false),
                        crate::TypeInner::Vector { scalar, size } => (scalar, Some(size), false),
                        crate::TypeInner::Matrix { scalar, .. } => (scalar, None, true),
                        ref other => {
                            log::error!("As source {:?}", other);
                            return Err(Error::Validation("Unexpected Expression::As source"));
                        }
                    };

                enum Cast {
                    Identity,
                    Unary(spirv::Op),
                    Binary(spirv::Op, Word),
                    Ternary(spirv::Op, Word, Word),
                }

                let cast = if is_matrix {
                    // we only support identity casts for matrices
                    Cast::Unary(spirv::Op::CopyObject)
                } else {
                    match (src_scalar.kind, kind, convert) {
                        // Filter out identity casts. Some Adreno drivers are
                        // confused by no-op OpBitCast instructions.
                        (src_kind, kind, convert)
                            if src_kind == kind
                                && convert.filter(|&width| width != src_scalar.width).is_none() =>
                        {
                            Cast::Identity
                        }
                        (Sk::Bool, Sk::Bool, _) => Cast::Unary(spirv::Op::CopyObject),
                        (_, _, None) => Cast::Unary(spirv::Op::Bitcast),
                        // casting to a bool - generate `OpXxxNotEqual`
                        (_, Sk::Bool, Some(_)) => {
                            let op = match src_scalar.kind {
                                Sk::Sint | Sk::Uint => spirv::Op::INotEqual,
                                Sk::Float => spirv::Op::FUnordNotEqual,
                                Sk::Bool | Sk::AbstractInt | Sk::AbstractFloat => unreachable!(),
                            };
                            let zero_scalar_id =
                                self.writer.get_constant_scalar_with(0, src_scalar)?;
                            let zero_id = match src_size {
                                Some(size) => {
                                    let ty = LocalType::Numeric(NumericType::Vector {
                                        size,
                                        scalar: src_scalar,
                                    })
                                    .into();

                                    self.temp_list.clear();
                                    self.temp_list.resize(size as _, zero_scalar_id);

                                    self.writer.get_constant_composite(ty, &self.temp_list)
                                }
                                None => zero_scalar_id,
                            };

                            Cast::Binary(op, zero_id)
                        }
                        // casting from a bool - generate `OpSelect`
                        (Sk::Bool, _, Some(dst_width)) => {
                            let dst_scalar = crate::Scalar {
                                kind,
                                width: dst_width,
                            };
                            let zero_scalar_id =
                                self.writer.get_constant_scalar_with(0, dst_scalar)?;
                            let one_scalar_id =
                                self.writer.get_constant_scalar_with(1, dst_scalar)?;
                            let (accept_id, reject_id) = match src_size {
                                Some(size) => {
                                    let ty = LocalType::Numeric(NumericType::Vector {
                                        size,
                                        scalar: dst_scalar,
                                    })
                                    .into();

                                    self.temp_list.clear();
                                    self.temp_list.resize(size as _, zero_scalar_id);

                                    let vec0_id =
                                        self.writer.get_constant_composite(ty, &self.temp_list);

                                    self.temp_list.fill(one_scalar_id);

                                    let vec1_id =
                                        self.writer.get_constant_composite(ty, &self.temp_list);

                                    (vec1_id, vec0_id)
                                }
                                None => (one_scalar_id, zero_scalar_id),
                            };

                            Cast::Ternary(spirv::Op::Select, accept_id, reject_id)
                        }
                        (Sk::Float, Sk::Uint, Some(_)) => Cast::Unary(spirv::Op::ConvertFToU),
                        (Sk::Float, Sk::Sint, Some(_)) => Cast::Unary(spirv::Op::ConvertFToS),
                        (Sk::Float, Sk::Float, Some(dst_width))
                            if src_scalar.width != dst_width =>
                        {
                            Cast::Unary(spirv::Op::FConvert)
                        }
                        (Sk::Sint, Sk::Float, Some(_)) => Cast::Unary(spirv::Op::ConvertSToF),
                        (Sk::Sint, Sk::Sint, Some(dst_width)) if src_scalar.width != dst_width => {
                            Cast::Unary(spirv::Op::SConvert)
                        }
                        (Sk::Uint, Sk::Float, Some(_)) => Cast::Unary(spirv::Op::ConvertUToF),
                        (Sk::Uint, Sk::Uint, Some(dst_width)) if src_scalar.width != dst_width => {
                            Cast::Unary(spirv::Op::UConvert)
                        }
                        (Sk::Uint, Sk::Sint, Some(dst_width)) if src_scalar.width != dst_width => {
                            Cast::Unary(spirv::Op::SConvert)
                        }
                        (Sk::Sint, Sk::Uint, Some(dst_width)) if src_scalar.width != dst_width => {
                            Cast::Unary(spirv::Op::UConvert)
                        }
                        // We assume it's either an identity cast, or int-uint.
                        _ => Cast::Unary(spirv::Op::Bitcast),
                    }
                };

                let id = self.gen_id();
                let instruction = match cast {
                    Cast::Identity => None,
                    Cast::Unary(op) => Some(Instruction::unary(op, result_type_id, id, expr_id)),
                    Cast::Binary(op, operand) => Some(Instruction::binary(
                        op,
                        result_type_id,
                        id,
                        expr_id,
                        operand,
                    )),
                    Cast::Ternary(op, op1, op2) => Some(Instruction::ternary(
                        op,
                        result_type_id,
                        id,
                        expr_id,
                        op1,
                        op2,
                    )),
                };
                if let Some(instruction) = instruction {
                    block.body.push(instruction);
                    id
                } else {
                    expr_id
                }
            }
            crate::Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => self.write_image_load(
                result_type_id,
                image,
                coordinate,
                array_index,
                level,
                sample,
                block,
            )?,
            crate::Expression::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => self.write_image_sample(
                result_type_id,
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                block,
            )?,
            crate::Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let id = self.gen_id();
                let mut condition_id = self.cached[condition];
                let accept_id = self.cached[accept];
                let reject_id = self.cached[reject];

                let condition_ty = self.fun_info[condition]
                    .ty
                    .inner_with(&self.ir_module.types);
                let object_ty = self.fun_info[accept].ty.inner_with(&self.ir_module.types);

                if let (
                    &crate::TypeInner::Scalar(
                        condition_scalar @ crate::Scalar {
                            kind: crate::ScalarKind::Bool,
                            ..
                        },
                    ),
                    &crate::TypeInner::Vector { size, .. },
                ) = (condition_ty, object_ty)
                {
                    self.temp_list.clear();
                    self.temp_list.resize(size as usize, condition_id);

                    let bool_vector_type_id = self.get_type_id(LookupType::Local(
                        LocalType::Numeric(NumericType::Vector {
                            size,
                            scalar: condition_scalar,
                        }),
                    ));

                    let id = self.gen_id();
                    block.body.push(Instruction::composite_construct(
                        bool_vector_type_id,
                        id,
                        &self.temp_list,
                    ));
                    condition_id = id
                }

                let instruction =
                    Instruction::select(result_type_id, id, condition_id, accept_id, reject_id);
                block.body.push(instruction);
                id
            }
            crate::Expression::Derivative { axis, ctrl, expr } => {
                use crate::{DerivativeAxis as Axis, DerivativeControl as Ctrl};
                match ctrl {
                    Ctrl::Coarse | Ctrl::Fine => {
                        self.writer.require_any(
                            "DerivativeControl",
                            &[spirv::Capability::DerivativeControl],
                        )?;
                    }
                    Ctrl::None => {}
                }
                let id = self.gen_id();
                let expr_id = self.cached[expr];
                let op = match (axis, ctrl) {
                    (Axis::X, Ctrl::Coarse) => spirv::Op::DPdxCoarse,
                    (Axis::X, Ctrl::Fine) => spirv::Op::DPdxFine,
                    (Axis::X, Ctrl::None) => spirv::Op::DPdx,
                    (Axis::Y, Ctrl::Coarse) => spirv::Op::DPdyCoarse,
                    (Axis::Y, Ctrl::Fine) => spirv::Op::DPdyFine,
                    (Axis::Y, Ctrl::None) => spirv::Op::DPdy,
                    (Axis::Width, Ctrl::Coarse) => spirv::Op::FwidthCoarse,
                    (Axis::Width, Ctrl::Fine) => spirv::Op::FwidthFine,
                    (Axis::Width, Ctrl::None) => spirv::Op::Fwidth,
                };
                block
                    .body
                    .push(Instruction::derivative(op, result_type_id, id, expr_id));
                id
            }
            crate::Expression::ImageQuery { image, query } => {
                self.write_image_query(result_type_id, image, query, block)?
            }
            crate::Expression::Relational { fun, argument } => {
                use crate::RelationalFunction as Rf;
                let arg_id = self.cached[argument];
                let op = match fun {
                    Rf::All => spirv::Op::All,
                    Rf::Any => spirv::Op::Any,
                    Rf::IsNan => spirv::Op::IsNan,
                    Rf::IsInf => spirv::Op::IsInf,
                };
                let id = self.gen_id();
                block
                    .body
                    .push(Instruction::relational(op, result_type_id, id, arg_id));
                id
            }
            crate::Expression::ArrayLength(expr) => self.write_runtime_array_length(expr, block)?,
            crate::Expression::RayQueryGetIntersection { query, committed } => {
                if !committed {
                    return Err(Error::FeatureNotImplemented("candidate intersection"));
                }
                self.write_ray_query_get_intersection(query, block)
            }
        };

        self.cached[expr_handle] = id;
        Ok(())
    }

    /// Build an `OpAccessChain` instruction.
    ///
    /// Emit any needed bounds-checking expressions to `block`.
    ///
    /// Give the `OpAccessChain` a result type based on `expr_handle`, adjusted
    /// according to `type_adjustment`; see the documentation for
    /// [`AccessTypeAdjustment`] for details.
    ///
    /// On success, the return value is an [`ExpressionPointer`] value; see the
    /// documentation for that type.
    fn write_access_chain(
        &mut self,
        mut expr_handle: Handle<crate::Expression>,
        block: &mut Block,
        type_adjustment: AccessTypeAdjustment,
    ) -> Result<ExpressionPointer, Error> {
        let result_type_id = {
            let resolution = &self.fun_info[expr_handle].ty;
            match type_adjustment {
                AccessTypeAdjustment::None => self.writer.get_expression_type_id(resolution),
                AccessTypeAdjustment::IntroducePointer(class) => {
                    self.writer.get_resolution_pointer_id(resolution, class)
                }
            }
        };

        // The id of the boolean `and` of all dynamic bounds checks up to this point.
        //
        // See `extend_bounds_check_condition_chain` for a full explanation.
        let mut accumulated_checks = None;

        // Is true if we are accessing into a binding array with a non-uniform index.
        let mut is_non_uniform_binding_array = false;

        self.temp_list.clear();
        let root_id = loop {
            // If `expr_handle` was spilled, then the temporary variable has exactly
            // the value we want to start from.
            if let Some(spilled) = self.function.spilled_composites.get(&expr_handle) {
                // The root id of the `OpAccessChain` instruction is the temporary
                // variable we spilled the composite to.
                break spilled.id;
            }

            expr_handle = match self.ir_function.expressions[expr_handle] {
                crate::Expression::Access { base, index } => {
                    is_non_uniform_binding_array |=
                        self.is_nonuniform_binding_array_access(base, index);

                    let index = GuardedIndex::Expression(index);
                    let index_id =
                        self.write_access_chain_index(base, index, &mut accumulated_checks, block)?;
                    self.temp_list.push(index_id);

                    base
                }
                crate::Expression::AccessIndex { base, index } => {
                    // Decide whether we're indexing a struct (bounds checks
                    // forbidden) or anything else (bounds checks required).
                    let mut base_ty = self.fun_info[base].ty.inner_with(&self.ir_module.types);
                    if let crate::TypeInner::Pointer { base, .. } = *base_ty {
                        base_ty = &self.ir_module.types[base].inner;
                    }
                    let index_id = if let crate::TypeInner::Struct { .. } = *base_ty {
                        self.get_index_constant(index)
                    } else {
                        // `index` is constant, so this can't possibly require
                        // setting `is_nonuniform_binding_array_access`.

                        // Even though the index value is statically known, `base`
                        // may be a runtime-sized array, so we still need to go
                        // through the bounds check process.
                        self.write_access_chain_index(
                            base,
                            GuardedIndex::Known(index),
                            &mut accumulated_checks,
                            block,
                        )?
                    };

                    self.temp_list.push(index_id);
                    base
                }
                crate::Expression::GlobalVariable(handle) => {
                    let gv = &self.writer.global_variables[handle];
                    break gv.access_id;
                }
                crate::Expression::LocalVariable(variable) => {
                    let local_var = &self.function.variables[&variable];
                    break local_var.id;
                }
                crate::Expression::FunctionArgument(index) => {
                    break self.function.parameter_id(index);
                }
                ref other => unimplemented!("Unexpected pointer expression {:?}", other),
            }
        };

        let (pointer_id, expr_pointer) = if self.temp_list.is_empty() {
            (
                root_id,
                ExpressionPointer::Ready {
                    pointer_id: root_id,
                },
            )
        } else {
            self.temp_list.reverse();
            let pointer_id = self.gen_id();
            let access =
                Instruction::access_chain(result_type_id, pointer_id, root_id, &self.temp_list);

            // If we generated some bounds checks, we need to leave it to our
            // caller to generate the branch, the access, the load or store, and
            // the zero value (for loads). Otherwise, we can emit the access
            // ourselves, and just hand them the id of the pointer.
            let expr_pointer = match accumulated_checks {
                Some(condition) => ExpressionPointer::Conditional { condition, access },
                None => {
                    block.body.push(access);
                    ExpressionPointer::Ready { pointer_id }
                }
            };
            (pointer_id, expr_pointer)
        };
        // Subsequent load, store and atomic operations require the pointer to be decorated as NonUniform
        // if the binding array was accessed with a non-uniform index
        // see VUID-RuntimeSpirv-NonUniform-06274
        if is_non_uniform_binding_array {
            self.writer
                .decorate_non_uniform_binding_array_access(pointer_id)?;
        }

        Ok(expr_pointer)
    }

    fn is_nonuniform_binding_array_access(
        &mut self,
        base: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
    ) -> bool {
        let crate::Expression::GlobalVariable(var_handle) = self.ir_function.expressions[base]
        else {
            return false;
        };

        // The access chain needs to be decorated as NonUniform
        // see VUID-RuntimeSpirv-NonUniform-06274
        let gvar = &self.ir_module.global_variables[var_handle];
        let crate::TypeInner::BindingArray { .. } = self.ir_module.types[gvar.ty].inner else {
            return false;
        };

        self.fun_info[index].uniformity.non_uniform_result.is_some()
    }

    /// Compute a single index operand to an `OpAccessChain` instruction.
    ///
    /// Given that we are indexing `base` with `index`, apply the appropriate
    /// bounds check policies, emitting code to `block` to clamp `index` or
    /// determine whether it's in bounds. Return the SPIR-V instruction id of
    /// the index value we should actually use.
    ///
    /// Extend `accumulated_checks` to include the results of any needed bounds
    /// checks. See [`BlockContext::extend_bounds_check_condition_chain`].
    fn write_access_chain_index(
        &mut self,
        base: Handle<crate::Expression>,
        index: GuardedIndex,
        accumulated_checks: &mut Option<Word>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        match self.write_bounds_check(base, index, block)? {
            BoundsCheckResult::KnownInBounds(known_index) => {
                // Even if the index is known, `OpAccessChain`
                // requires expression operands, not literals.
                let scalar = crate::Literal::U32(known_index);
                Ok(self.writer.get_constant_scalar(scalar))
            }
            BoundsCheckResult::Computed(computed_index_id) => Ok(computed_index_id),
            BoundsCheckResult::Conditional {
                condition_id: condition,
                index_id: index,
            } => {
                self.extend_bounds_check_condition_chain(accumulated_checks, condition, block);

                // Use the index from the `Access` expression unchanged.
                Ok(index)
            }
        }
    }

    /// Add a condition to a chain of bounds checks.
    ///
    /// As we build an `OpAccessChain` instruction govered by
    /// [`BoundsCheckPolicy::ReadZeroSkipWrite`], we accumulate a chain of
    /// dynamic bounds checks, one for each index in the chain, which must all
    /// be true for that `OpAccessChain`'s execution to be well-defined. This
    /// function adds the boolean instruction id `comparison_id` to `chain`.
    ///
    /// If `chain` is `None`, that means there are no bounds checks in the chain
    /// yet. If chain is `Some(id)`, then `id` is the conjunction of all the
    /// bounds checks in the chain.
    ///
    /// When we have multiple bounds checks, we combine them with
    /// `OpLogicalAnd`, not a short-circuit branch. This means we might do
    /// comparisons we don't need to, but we expect these checks to almost
    /// always succeed, and keeping branches to a minimum is essential.
    ///
    /// [`BoundsCheckPolicy::ReadZeroSkipWrite`]: crate::proc::BoundsCheckPolicy
    fn extend_bounds_check_condition_chain(
        &mut self,
        chain: &mut Option<Word>,
        comparison_id: Word,
        block: &mut Block,
    ) {
        match *chain {
            Some(ref mut prior_checks) => {
                let combined = self.gen_id();
                block.body.push(Instruction::binary(
                    spirv::Op::LogicalAnd,
                    self.writer.get_bool_type_id(),
                    combined,
                    *prior_checks,
                    comparison_id,
                ));
                *prior_checks = combined;
            }
            None => {
                // Start a fresh chain of checks.
                *chain = Some(comparison_id);
            }
        }
    }

    fn write_checked_load(
        &mut self,
        pointer: Handle<crate::Expression>,
        block: &mut Block,
        access_type_adjustment: AccessTypeAdjustment,
        result_type_id: Word,
    ) -> Result<Word, Error> {
        match self.write_access_chain(pointer, block, access_type_adjustment)? {
            ExpressionPointer::Ready { pointer_id } => {
                let id = self.gen_id();
                let atomic_space =
                    match *self.fun_info[pointer].ty.inner_with(&self.ir_module.types) {
                        crate::TypeInner::Pointer { base, space } => {
                            match self.ir_module.types[base].inner {
                                crate::TypeInner::Atomic { .. } => Some(space),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                let instruction = if let Some(space) = atomic_space {
                    let (semantics, scope) = space.to_spirv_semantics_and_scope();
                    let scope_constant_id = self.get_scope_constant(scope as u32);
                    let semantics_id = self.get_index_constant(semantics.bits());
                    Instruction::atomic_load(
                        result_type_id,
                        id,
                        pointer_id,
                        scope_constant_id,
                        semantics_id,
                    )
                } else {
                    Instruction::load(result_type_id, id, pointer_id, None)
                };
                block.body.push(instruction);
                Ok(id)
            }
            ExpressionPointer::Conditional { condition, access } => {
                //TODO: support atomics?
                let value = self.write_conditional_indexed_load(
                    result_type_id,
                    condition,
                    block,
                    move |id_gen, block| {
                        // The in-bounds path. Perform the access and the load.
                        let pointer_id = access.result_id.unwrap();
                        let value_id = id_gen.next();
                        block.body.push(access);
                        block.body.push(Instruction::load(
                            result_type_id,
                            value_id,
                            pointer_id,
                            None,
                        ));
                        value_id
                    },
                );
                Ok(value)
            }
        }
    }

    fn spill_to_internal_variable(&mut self, base: Handle<crate::Expression>, block: &mut Block) {
        // Generate an internal variable of the appropriate type for `base`.
        let variable_id = self.writer.id_gen.next();
        let pointer_type_id = self
            .writer
            .get_resolution_pointer_id(&self.fun_info[base].ty, spirv::StorageClass::Function);
        let variable = super::LocalVariable {
            id: variable_id,
            instruction: Instruction::variable(
                pointer_type_id,
                variable_id,
                spirv::StorageClass::Function,
                None,
            ),
        };

        let base_id = self.cached[base];
        block
            .body
            .push(Instruction::store(variable.id, base_id, None));
        self.function.spilled_composites.insert(base, variable);
    }

    /// Generate an access to a spilled temporary, if necessary.
    ///
    /// Given `access`, an [`Access`] or [`AccessIndex`] expression that refers
    /// to a component of a composite value that has been spilled to a temporary
    /// variable, determine whether other expressions are going to use
    /// `access`'s value:
    ///
    /// - If so, perform the access and cache that as the value of `access`.
    ///
    /// - Otherwise, generate no code and cache no value for `access`.
    ///
    /// Return `Ok(0)` if no value was fetched, or `Ok(id)` if we loaded it into
    /// the instruction given by `id`.
    ///
    /// [`Access`]: crate::Expression::Access
    /// [`AccessIndex`]: crate::Expression::AccessIndex
    fn maybe_access_spilled_composite(
        &mut self,
        access: Handle<crate::Expression>,
        block: &mut Block,
        result_type_id: Word,
    ) -> Result<Word, Error> {
        let access_uses = self.function.access_uses.get(&access).map_or(0, |r| *r);
        if access_uses == self.fun_info[access].ref_count {
            // This expression is only used by other `Access` and
            // `AccessIndex` expressions, so we don't need to cache a
            // value for it yet.
            Ok(0)
        } else {
            // There are other expressions that are going to expect this
            // expression's value to be cached, not just other `Access` or
            // `AccessIndex` expressions. We must actually perform the
            // access on the spill variable now.
            self.write_checked_load(
                access,
                block,
                AccessTypeAdjustment::IntroducePointer(spirv::StorageClass::Function),
                result_type_id,
            )
        }
    }

    /// Build the instructions for matrix - matrix column operations
    #[allow(clippy::too_many_arguments)]
    fn write_matrix_matrix_column_op(
        &mut self,
        block: &mut Block,
        result_id: Word,
        result_type_id: Word,
        left_id: Word,
        right_id: Word,
        columns: crate::VectorSize,
        rows: crate::VectorSize,
        width: u8,
        op: spirv::Op,
    ) {
        self.temp_list.clear();

        let vector_type_id =
            self.get_type_id(LookupType::Local(LocalType::Numeric(NumericType::Vector {
                size: rows,
                scalar: crate::Scalar::float(width),
            })));

        for index in 0..columns as u32 {
            let column_id_left = self.gen_id();
            let column_id_right = self.gen_id();
            let column_id_res = self.gen_id();

            block.body.push(Instruction::composite_extract(
                vector_type_id,
                column_id_left,
                left_id,
                &[index],
            ));
            block.body.push(Instruction::composite_extract(
                vector_type_id,
                column_id_right,
                right_id,
                &[index],
            ));
            block.body.push(Instruction::binary(
                op,
                vector_type_id,
                column_id_res,
                column_id_left,
                column_id_right,
            ));

            self.temp_list.push(column_id_res);
        }

        block.body.push(Instruction::composite_construct(
            result_type_id,
            result_id,
            &self.temp_list,
        ));
    }

    /// Build the instructions for vector - scalar multiplication
    fn write_vector_scalar_mult(
        &mut self,
        block: &mut Block,
        result_id: Word,
        result_type_id: Word,
        vector_id: Word,
        scalar_id: Word,
        vector: &crate::TypeInner,
    ) {
        let (size, kind) = match *vector {
            crate::TypeInner::Vector {
                size,
                scalar: crate::Scalar { kind, .. },
            } => (size, kind),
            _ => unreachable!(),
        };

        let (op, operand_id) = match kind {
            crate::ScalarKind::Float => (spirv::Op::VectorTimesScalar, scalar_id),
            _ => {
                let operand_id = self.gen_id();
                self.temp_list.clear();
                self.temp_list.resize(size as usize, scalar_id);
                block.body.push(Instruction::composite_construct(
                    result_type_id,
                    operand_id,
                    &self.temp_list,
                ));
                (spirv::Op::IMul, operand_id)
            }
        };

        block.body.push(Instruction::binary(
            op,
            result_type_id,
            result_id,
            vector_id,
            operand_id,
        ));
    }

    /// Build the instructions for the arithmetic expression of a dot product
    fn write_dot_product(
        &mut self,
        result_id: Word,
        result_type_id: Word,
        arg0_id: Word,
        arg1_id: Word,
        size: u32,
        block: &mut Block,
    ) {
        let mut partial_sum = self.writer.get_constant_null(result_type_id);
        let last_component = size - 1;
        for index in 0..=last_component {
            // compute the product of the current components
            let a_id = self.gen_id();
            block.body.push(Instruction::composite_extract(
                result_type_id,
                a_id,
                arg0_id,
                &[index],
            ));
            let b_id = self.gen_id();
            block.body.push(Instruction::composite_extract(
                result_type_id,
                b_id,
                arg1_id,
                &[index],
            ));
            let prod_id = self.gen_id();
            block.body.push(Instruction::binary(
                spirv::Op::IMul,
                result_type_id,
                prod_id,
                a_id,
                b_id,
            ));

            // choose the id for the next sum, depending on current index
            let id = if index == last_component {
                result_id
            } else {
                self.gen_id()
            };

            // sum the computed product with the partial sum
            block.body.push(Instruction::binary(
                spirv::Op::IAdd,
                result_type_id,
                id,
                partial_sum,
                prod_id,
            ));
            // set the id of the result as the previous partial sum
            partial_sum = id;
        }
    }

    /// Generate one or more SPIR-V blocks for `naga_block`.
    ///
    /// Use `label_id` as the label for the SPIR-V entry point block.
    ///
    /// If control reaches the end of the SPIR-V block, terminate it according
    /// to `exit`. This function's return value indicates whether it acted on
    /// this parameter or not; see [`BlockExitDisposition`].
    ///
    /// If the block contains [`Break`] or [`Continue`] statements,
    /// `loop_context` supplies the labels of the SPIR-V blocks to jump to. If
    /// either of these labels are `None`, then it should have been a Naga
    /// validation error for the corresponding statement to occur in this
    /// context.
    ///
    /// [`Break`]: Statement::Break
    /// [`Continue`]: Statement::Continue
    fn write_block(
        &mut self,
        label_id: Word,
        naga_block: &crate::Block,
        exit: BlockExit,
        loop_context: LoopContext,
        debug_info: Option<&DebugInfoInner>,
    ) -> Result<BlockExitDisposition, Error> {
        let mut block = Block::new(label_id);
        for (statement, span) in naga_block.span_iter() {
            if let (Some(debug_info), false) = (
                debug_info,
                matches!(
                    statement,
                    &(Statement::Block(..)
                        | Statement::Break
                        | Statement::Continue
                        | Statement::Kill
                        | Statement::Return { .. }
                        | Statement::Loop { .. })
                ),
            ) {
                let loc: crate::SourceLocation = span.location(debug_info.source_code);
                block.body.push(Instruction::line(
                    debug_info.source_file_id,
                    loc.line_number,
                    loc.line_position,
                ));
            };
            match *statement {
                Statement::Emit(ref range) => {
                    for handle in range.clone() {
                        // omit const expressions as we've already cached those
                        if !self.expression_constness.is_const(handle) {
                            self.cache_expression_value(handle, &mut block)?;
                        }
                    }
                }
                Statement::Block(ref block_statements) => {
                    let scope_id = self.gen_id();
                    self.function.consume(block, Instruction::branch(scope_id));

                    let merge_id = self.gen_id();
                    let merge_used = self.write_block(
                        scope_id,
                        block_statements,
                        BlockExit::Branch { target: merge_id },
                        loop_context,
                        debug_info,
                    )?;

                    match merge_used {
                        BlockExitDisposition::Used => {
                            block = Block::new(merge_id);
                        }
                        BlockExitDisposition::Discarded => {
                            return Ok(BlockExitDisposition::Discarded);
                        }
                    }
                }
                Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    let condition_id = self.cached[condition];

                    let merge_id = self.gen_id();
                    block.body.push(Instruction::selection_merge(
                        merge_id,
                        spirv::SelectionControl::NONE,
                    ));

                    let accept_id = if accept.is_empty() {
                        None
                    } else {
                        Some(self.gen_id())
                    };
                    let reject_id = if reject.is_empty() {
                        None
                    } else {
                        Some(self.gen_id())
                    };

                    self.function.consume(
                        block,
                        Instruction::branch_conditional(
                            condition_id,
                            accept_id.unwrap_or(merge_id),
                            reject_id.unwrap_or(merge_id),
                        ),
                    );

                    if let Some(block_id) = accept_id {
                        // We can ignore the `BlockExitDisposition` returned here because,
                        // even if `merge_id` is not actually reachable, it is always
                        // referred to by the `OpSelectionMerge` instruction we emitted
                        // earlier.
                        let _ = self.write_block(
                            block_id,
                            accept,
                            BlockExit::Branch { target: merge_id },
                            loop_context,
                            debug_info,
                        )?;
                    }
                    if let Some(block_id) = reject_id {
                        // We can ignore the `BlockExitDisposition` returned here because,
                        // even if `merge_id` is not actually reachable, it is always
                        // referred to by the `OpSelectionMerge` instruction we emitted
                        // earlier.
                        let _ = self.write_block(
                            block_id,
                            reject,
                            BlockExit::Branch { target: merge_id },
                            loop_context,
                            debug_info,
                        )?;
                    }

                    block = Block::new(merge_id);
                }
                Statement::Switch {
                    selector,
                    ref cases,
                } => {
                    let selector_id = self.cached[selector];

                    let merge_id = self.gen_id();
                    block.body.push(Instruction::selection_merge(
                        merge_id,
                        spirv::SelectionControl::NONE,
                    ));

                    let mut default_id = None;
                    // id of previous empty fall-through case
                    let mut last_id = None;

                    let mut raw_cases = Vec::with_capacity(cases.len());
                    let mut case_ids = Vec::with_capacity(cases.len());
                    for case in cases.iter() {
                        // take id of previous empty fall-through case or generate a new one
                        let label_id = last_id.take().unwrap_or_else(|| self.gen_id());

                        if case.fall_through && case.body.is_empty() {
                            last_id = Some(label_id);
                        }

                        case_ids.push(label_id);

                        match case.value {
                            crate::SwitchValue::I32(value) => {
                                raw_cases.push(super::instructions::Case {
                                    value: value as Word,
                                    label_id,
                                });
                            }
                            crate::SwitchValue::U32(value) => {
                                raw_cases.push(super::instructions::Case { value, label_id });
                            }
                            crate::SwitchValue::Default => {
                                default_id = Some(label_id);
                            }
                        }
                    }

                    let default_id = default_id.unwrap();

                    self.function.consume(
                        block,
                        Instruction::switch(selector_id, default_id, &raw_cases),
                    );

                    let inner_context = LoopContext {
                        break_id: Some(merge_id),
                        ..loop_context
                    };

                    for (i, (case, label_id)) in cases
                        .iter()
                        .zip(case_ids.iter())
                        .filter(|&(case, _)| !(case.fall_through && case.body.is_empty()))
                        .enumerate()
                    {
                        let case_finish_id = if case.fall_through {
                            case_ids[i + 1]
                        } else {
                            merge_id
                        };
                        // We can ignore the `BlockExitDisposition` returned here because
                        // `case_finish_id` is always referred to by either:
                        //
                        // - the `OpSwitch`, if it's the next case's label for a
                        //   fall-through, or
                        //
                        // - the `OpSelectionMerge`, if it's the switch's overall merge
                        //   block because there's no fall-through.
                        let _ = self.write_block(
                            *label_id,
                            &case.body,
                            BlockExit::Branch {
                                target: case_finish_id,
                            },
                            inner_context,
                            debug_info,
                        )?;
                    }

                    block = Block::new(merge_id);
                }
                Statement::Loop {
                    ref body,
                    ref continuing,
                    break_if,
                } => {
                    let preamble_id = self.gen_id();
                    self.function
                        .consume(block, Instruction::branch(preamble_id));

                    let merge_id = self.gen_id();
                    let body_id = self.gen_id();
                    let continuing_id = self.gen_id();

                    // SPIR-V requires the continuing to the `OpLoopMerge`,
                    // so we have to start a new block with it.
                    block = Block::new(preamble_id);
                    // HACK the loop statement is begin with branch instruction,
                    // so we need to put `OpLine` debug info before merge instruction
                    if let Some(debug_info) = debug_info {
                        let loc: crate::SourceLocation = span.location(debug_info.source_code);
                        block.body.push(Instruction::line(
                            debug_info.source_file_id,
                            loc.line_number,
                            loc.line_position,
                        ))
                    }
                    block.body.push(Instruction::loop_merge(
                        merge_id,
                        continuing_id,
                        spirv::SelectionControl::NONE,
                    ));
                    self.function.consume(block, Instruction::branch(body_id));

                    // We can ignore the `BlockExitDisposition` returned here because,
                    // even if `continuing_id` is not actually reachable, it is always
                    // referred to by the `OpLoopMerge` instruction we emitted earlier.
                    let _ = self.write_block(
                        body_id,
                        body,
                        BlockExit::Branch {
                            target: continuing_id,
                        },
                        LoopContext {
                            continuing_id: Some(continuing_id),
                            break_id: Some(merge_id),
                        },
                        debug_info,
                    )?;

                    let exit = match break_if {
                        Some(condition) => BlockExit::BreakIf {
                            condition,
                            preamble_id,
                        },
                        None => BlockExit::Branch {
                            target: preamble_id,
                        },
                    };

                    // We can ignore the `BlockExitDisposition` returned here because,
                    // even if `merge_id` is not actually reachable, it is always referred
                    // to by the `OpLoopMerge` instruction we emitted earlier.
                    let _ = self.write_block(
                        continuing_id,
                        continuing,
                        exit,
                        LoopContext {
                            continuing_id: None,
                            break_id: Some(merge_id),
                        },
                        debug_info,
                    )?;

                    block = Block::new(merge_id);
                }
                Statement::Break => {
                    self.function
                        .consume(block, Instruction::branch(loop_context.break_id.unwrap()));
                    return Ok(BlockExitDisposition::Discarded);
                }
                Statement::Continue => {
                    self.function.consume(
                        block,
                        Instruction::branch(loop_context.continuing_id.unwrap()),
                    );
                    return Ok(BlockExitDisposition::Discarded);
                }
                Statement::Return { value: Some(value) } => {
                    let value_id = self.cached[value];
                    let instruction = match self.function.entry_point_context {
                        // If this is an entry point, and we need to return anything,
                        // let's instead store the output variables and return `void`.
                        Some(ref context) => {
                            self.writer.write_entry_point_return(
                                value_id,
                                self.ir_function.result.as_ref().unwrap(),
                                &context.results,
                                &mut block.body,
                            )?;
                            Instruction::return_void()
                        }
                        None => Instruction::return_value(value_id),
                    };
                    self.function.consume(block, instruction);
                    return Ok(BlockExitDisposition::Discarded);
                }
                Statement::Return { value: None } => {
                    self.function.consume(block, Instruction::return_void());
                    return Ok(BlockExitDisposition::Discarded);
                }
                Statement::Kill => {
                    self.function.consume(block, Instruction::kill());
                    return Ok(BlockExitDisposition::Discarded);
                }
                Statement::Barrier(flags) => {
                    self.writer.write_barrier(flags, &mut block);
                }
                Statement::Store { pointer, value } => {
                    let value_id = self.cached[value];
                    match self.write_access_chain(
                        pointer,
                        &mut block,
                        AccessTypeAdjustment::None,
                    )? {
                        ExpressionPointer::Ready { pointer_id } => {
                            let atomic_space = match *self.fun_info[pointer]
                                .ty
                                .inner_with(&self.ir_module.types)
                            {
                                crate::TypeInner::Pointer { base, space } => {
                                    match self.ir_module.types[base].inner {
                                        crate::TypeInner::Atomic { .. } => Some(space),
                                        _ => None,
                                    }
                                }
                                _ => None,
                            };
                            let instruction = if let Some(space) = atomic_space {
                                let (semantics, scope) = space.to_spirv_semantics_and_scope();
                                let scope_constant_id = self.get_scope_constant(scope as u32);
                                let semantics_id = self.get_index_constant(semantics.bits());
                                Instruction::atomic_store(
                                    pointer_id,
                                    scope_constant_id,
                                    semantics_id,
                                    value_id,
                                )
                            } else {
                                Instruction::store(pointer_id, value_id, None)
                            };
                            block.body.push(instruction);
                        }
                        ExpressionPointer::Conditional { condition, access } => {
                            let mut selection = Selection::start(&mut block, ());
                            selection.if_true(self, condition, ());

                            // The in-bounds path. Perform the access and the store.
                            let pointer_id = access.result_id.unwrap();
                            selection.block().body.push(access);
                            selection
                                .block()
                                .body
                                .push(Instruction::store(pointer_id, value_id, None));

                            // Finish the in-bounds block and start the merge block. This
                            // is the block we'll leave current on return.
                            selection.finish(self, ());
                        }
                    };
                }
                Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => self.write_image_store(image, coordinate, array_index, value, &mut block)?,
                Statement::Call {
                    function: local_function,
                    ref arguments,
                    result,
                } => {
                    let id = self.gen_id();
                    self.temp_list.clear();
                    for &argument in arguments {
                        self.temp_list.push(self.cached[argument]);
                    }

                    let type_id = match result {
                        Some(expr) => {
                            self.cached[expr] = id;
                            self.get_expression_type_id(&self.fun_info[expr].ty)
                        }
                        None => self.writer.void_type,
                    };

                    block.body.push(Instruction::function_call(
                        type_id,
                        id,
                        self.writer.lookup_function[&local_function],
                        &self.temp_list,
                    ));
                }
                Statement::Atomic {
                    pointer,
                    ref fun,
                    value,
                    result,
                } => {
                    let id = self.gen_id();
                    // Compare-and-exchange operations produce a struct result,
                    // so use `result`'s type if it is available. For no-result
                    // operations, fall back to `value`'s type.
                    let result_type_id =
                        self.get_expression_type_id(&self.fun_info[result.unwrap_or(value)].ty);

                    if let Some(result) = result {
                        self.cached[result] = id;
                    }

                    let pointer_id = match self.write_access_chain(
                        pointer,
                        &mut block,
                        AccessTypeAdjustment::None,
                    )? {
                        ExpressionPointer::Ready { pointer_id } => pointer_id,
                        ExpressionPointer::Conditional { .. } => {
                            return Err(Error::FeatureNotImplemented(
                                "Atomics out-of-bounds handling",
                            ));
                        }
                    };

                    let space = self.fun_info[pointer]
                        .ty
                        .inner_with(&self.ir_module.types)
                        .pointer_space()
                        .unwrap();
                    let (semantics, scope) = space.to_spirv_semantics_and_scope();
                    let scope_constant_id = self.get_scope_constant(scope as u32);
                    let semantics_id = self.get_index_constant(semantics.bits());
                    let value_id = self.cached[value];
                    let value_inner = self.fun_info[value].ty.inner_with(&self.ir_module.types);

                    let instruction = match *fun {
                        crate::AtomicFunction::Add => Instruction::atomic_binary(
                            spirv::Op::AtomicIAdd,
                            result_type_id,
                            id,
                            pointer_id,
                            scope_constant_id,
                            semantics_id,
                            value_id,
                        ),
                        crate::AtomicFunction::Subtract => Instruction::atomic_binary(
                            spirv::Op::AtomicISub,
                            result_type_id,
                            id,
                            pointer_id,
                            scope_constant_id,
                            semantics_id,
                            value_id,
                        ),
                        crate::AtomicFunction::And => Instruction::atomic_binary(
                            spirv::Op::AtomicAnd,
                            result_type_id,
                            id,
                            pointer_id,
                            scope_constant_id,
                            semantics_id,
                            value_id,
                        ),
                        crate::AtomicFunction::InclusiveOr => Instruction::atomic_binary(
                            spirv::Op::AtomicOr,
                            result_type_id,
                            id,
                            pointer_id,
                            scope_constant_id,
                            semantics_id,
                            value_id,
                        ),
                        crate::AtomicFunction::ExclusiveOr => Instruction::atomic_binary(
                            spirv::Op::AtomicXor,
                            result_type_id,
                            id,
                            pointer_id,
                            scope_constant_id,
                            semantics_id,
                            value_id,
                        ),
                        crate::AtomicFunction::Min => {
                            let spirv_op = match *value_inner {
                                crate::TypeInner::Scalar(crate::Scalar {
                                    kind: crate::ScalarKind::Sint,
                                    width: _,
                                }) => spirv::Op::AtomicSMin,
                                crate::TypeInner::Scalar(crate::Scalar {
                                    kind: crate::ScalarKind::Uint,
                                    width: _,
                                }) => spirv::Op::AtomicUMin,
                                _ => unimplemented!(),
                            };
                            Instruction::atomic_binary(
                                spirv_op,
                                result_type_id,
                                id,
                                pointer_id,
                                scope_constant_id,
                                semantics_id,
                                value_id,
                            )
                        }
                        crate::AtomicFunction::Max => {
                            let spirv_op = match *value_inner {
                                crate::TypeInner::Scalar(crate::Scalar {
                                    kind: crate::ScalarKind::Sint,
                                    width: _,
                                }) => spirv::Op::AtomicSMax,
                                crate::TypeInner::Scalar(crate::Scalar {
                                    kind: crate::ScalarKind::Uint,
                                    width: _,
                                }) => spirv::Op::AtomicUMax,
                                _ => unimplemented!(),
                            };
                            Instruction::atomic_binary(
                                spirv_op,
                                result_type_id,
                                id,
                                pointer_id,
                                scope_constant_id,
                                semantics_id,
                                value_id,
                            )
                        }
                        crate::AtomicFunction::Exchange { compare: None } => {
                            Instruction::atomic_binary(
                                spirv::Op::AtomicExchange,
                                result_type_id,
                                id,
                                pointer_id,
                                scope_constant_id,
                                semantics_id,
                                value_id,
                            )
                        }
                        crate::AtomicFunction::Exchange { compare: Some(cmp) } => {
                            let scalar_type_id = match *value_inner {
                                crate::TypeInner::Scalar(scalar) => {
                                    self.get_type_id(LookupType::Local(LocalType::Numeric(
                                        NumericType::Scalar(scalar),
                                    )))
                                }
                                _ => unimplemented!(),
                            };
                            let bool_type_id = self.get_type_id(LookupType::Local(
                                LocalType::Numeric(NumericType::Scalar(crate::Scalar::BOOL)),
                            ));

                            let cas_result_id = self.gen_id();
                            let equality_result_id = self.gen_id();
                            let mut cas_instr = Instruction::new(spirv::Op::AtomicCompareExchange);
                            cas_instr.set_type(scalar_type_id);
                            cas_instr.set_result(cas_result_id);
                            cas_instr.add_operand(pointer_id);
                            cas_instr.add_operand(scope_constant_id);
                            cas_instr.add_operand(semantics_id); // semantics if equal
                            cas_instr.add_operand(semantics_id); // semantics if not equal
                            cas_instr.add_operand(value_id);
                            cas_instr.add_operand(self.cached[cmp]);
                            block.body.push(cas_instr);
                            block.body.push(Instruction::binary(
                                spirv::Op::IEqual,
                                bool_type_id,
                                equality_result_id,
                                cas_result_id,
                                self.cached[cmp],
                            ));
                            Instruction::composite_construct(
                                result_type_id,
                                id,
                                &[cas_result_id, equality_result_id],
                            )
                        }
                    };

                    block.body.push(instruction);
                }
                Statement::WorkGroupUniformLoad { pointer, result } => {
                    self.writer
                        .write_barrier(crate::Barrier::WORK_GROUP, &mut block);
                    let result_type_id = self.get_expression_type_id(&self.fun_info[result].ty);
                    // Embed the body of
                    match self.write_access_chain(
                        pointer,
                        &mut block,
                        AccessTypeAdjustment::None,
                    )? {
                        ExpressionPointer::Ready { pointer_id } => {
                            let id = self.gen_id();
                            block.body.push(Instruction::load(
                                result_type_id,
                                id,
                                pointer_id,
                                None,
                            ));
                            self.cached[result] = id;
                        }
                        ExpressionPointer::Conditional { condition, access } => {
                            self.cached[result] = self.write_conditional_indexed_load(
                                result_type_id,
                                condition,
                                &mut block,
                                move |id_gen, block| {
                                    // The in-bounds path. Perform the access and the load.
                                    let pointer_id = access.result_id.unwrap();
                                    let value_id = id_gen.next();
                                    block.body.push(access);
                                    block.body.push(Instruction::load(
                                        result_type_id,
                                        value_id,
                                        pointer_id,
                                        None,
                                    ));
                                    value_id
                                },
                            )
                        }
                    }
                    self.writer
                        .write_barrier(crate::Barrier::WORK_GROUP, &mut block);
                }
                Statement::RayQuery { query, ref fun } => {
                    self.write_ray_query_function(query, fun, &mut block);
                }
                Statement::SubgroupBallot {
                    result,
                    ref predicate,
                } => {
                    self.write_subgroup_ballot(predicate, result, &mut block)?;
                }
                Statement::SubgroupCollectiveOperation {
                    ref op,
                    ref collective_op,
                    argument,
                    result,
                } => {
                    self.write_subgroup_operation(op, collective_op, argument, result, &mut block)?;
                }
                Statement::SubgroupGather {
                    ref mode,
                    argument,
                    result,
                } => {
                    self.write_subgroup_gather(mode, argument, result, &mut block)?;
                }
            }
        }

        let termination = match exit {
            // We're generating code for the top-level Block of the function, so we
            // need to end it with some kind of return instruction.
            BlockExit::Return => match self.ir_function.result {
                Some(ref result) if self.function.entry_point_context.is_none() => {
                    let type_id = self.get_type_id(LookupType::Handle(result.ty));
                    let null_id = self.writer.get_constant_null(type_id);
                    Instruction::return_value(null_id)
                }
                _ => Instruction::return_void(),
            },
            BlockExit::Branch { target } => Instruction::branch(target),
            BlockExit::BreakIf {
                condition,
                preamble_id,
            } => {
                let condition_id = self.cached[condition];

                Instruction::branch_conditional(
                    condition_id,
                    loop_context.break_id.unwrap(),
                    preamble_id,
                )
            }
        };

        self.function.consume(block, termination);
        Ok(BlockExitDisposition::Used)
    }

    pub(super) fn write_function_body(
        &mut self,
        entry_id: Word,
        debug_info: Option<&DebugInfoInner>,
    ) -> Result<(), Error> {
        // We can ignore the `BlockExitDisposition` returned here because
        // `BlockExit::Return` doesn't refer to a block.
        let _ = self.write_block(
            entry_id,
            &self.ir_function.body,
            BlockExit::Return,
            LoopContext::default(),
            debug_info,
        )?;

        Ok(())
    }
}
