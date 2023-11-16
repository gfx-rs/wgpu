/*!
Implementations for `BlockContext` methods.
*/

use super::{
    helpers, index::BoundsCheckResult, make_local, selection::Selection, Block, BlockContext,
    Dimension, Error, Instruction, LocalType, LookupType, LoopContext, ResultMember, Writer,
    WriterFlags,
};
use crate::{arena::Handle, proc::TypeResolution, Statement};
use spirv::Word;

fn get_dimension(type_inner: &crate::TypeInner) -> Dimension {
    match *type_inner {
        crate::TypeInner::Scalar(_) => Dimension::Scalar,
        crate::TypeInner::Vector { .. } => Dimension::Vector,
        crate::TypeInner::Matrix { .. } => Dimension::Matrix,
        _ => unreachable!(),
    }
}

/// The results of emitting code for a left-hand-side expression.
///
/// On success, `write_expression_pointer` returns one of these.
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
pub enum BlockExit {
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
        let float_ptr_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: None,
            scalar: crate::Scalar::F32,
            pointer_space: Some(spirv::StorageClass::Output),
        }));
        let index_y_id = self.get_index_constant(1);
        let access_id = self.id_gen.next();
        body.push(Instruction::access_chain(
            float_ptr_type_id,
            access_id,
            position_id,
            &[index_y_id],
        ));

        let float_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: None,
            scalar: crate::Scalar::F32,
            pointer_space: None,
        }));
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
        let float_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: None,
            scalar: crate::Scalar::F32,
            pointer_space: None,
        }));
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
    /// Decide whether to put off emitting instructions for `expr_handle`.
    ///
    /// We would like to gather together chains of `Access` and `AccessIndex`
    /// Naga expressions into a single `OpAccessChain` SPIR-V instruction. To do
    /// this, we don't generate instructions for these exprs when we first
    /// encounter them. Their ids in `self.writer.cached.ids` are left as zero. Then,
    /// once we encounter a `Load` or `Store` expression that actually needs the
    /// chain's value, we call `write_expression_pointer` to handle the whole
    /// thing in one fell swoop.
    fn is_intermediate(&self, expr_handle: Handle<crate::Expression>) -> bool {
        match self.ir_function.expressions[expr_handle] {
            crate::Expression::GlobalVariable(handle) => {
                match self.ir_module.global_variables[handle].space {
                    crate::AddressSpace::Handle => false,
                    _ => true,
                }
            }
            crate::Expression::LocalVariable(_) => true,
            crate::Expression::FunctionArgument(index) => {
                let arg = &self.ir_function.arguments[index as usize];
                self.ir_module.types[arg.ty].inner.pointer_space().is_some()
            }

            // The chain rule: if this `Access...`'s `base` operand was
            // previously omitted, then omit this one, too.
            _ => self.cached.ids[expr_handle.index()] == 0,
        }
    }

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
                self.writer.constant_ids[init.index()]
            }
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
            crate::Expression::Access { base, index: _ } if self.is_intermediate(base) => {
                // See `is_intermediate`; we'll handle this later in
                // `write_expression_pointer`.
                0
            }
            crate::Expression::Access { base, index } => {
                let base_ty_inner = self.fun_info[base].ty.inner_with(&self.ir_module.types);
                match *base_ty_inner {
                    crate::TypeInner::Vector { .. } => {
                        self.write_vector_access(expr_handle, base, index, block)?
                    }
                    // Only binding arrays in the Handle address space will take this path (due to `is_intermediate`)
                    crate::TypeInner::BindingArray {
                        base: binding_type, ..
                    } => {
                        let space = match self.ir_function.expressions[base] {
                            crate::Expression::GlobalVariable(gvar) => {
                                self.ir_module.global_variables[gvar].space
                            }
                            _ => unreachable!(),
                        };
                        let binding_array_false_pointer = LookupType::Local(LocalType::Pointer {
                            base: binding_type,
                            class: helpers::map_storage_class(space),
                        });

                        let result_id = match self.write_expression_pointer(
                            expr_handle,
                            block,
                            Some(binding_array_false_pointer),
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
                            "only vectors may be dynamically indexed by value",
                        ));
                    }
                }
            }
            crate::Expression::AccessIndex { base, index: _ } if self.is_intermediate(base) => {
                // See `is_intermediate`; we'll handle this later in
                // `write_expression_pointer`.
                0
            }
            crate::Expression::AccessIndex { base, index } => {
                match *self.fun_info[base].ty.inner_with(&self.ir_module.types) {
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
                    // Only binding arrays in the Handle address space will take this path (due to `is_intermediate`)
                    crate::TypeInner::BindingArray {
                        base: binding_type, ..
                    } => {
                        let space = match self.ir_function.expressions[base] {
                            crate::Expression::GlobalVariable(gvar) => {
                                self.ir_module.global_variables[gvar].space
                            }
                            _ => unreachable!(),
                        };
                        let binding_array_false_pointer = LookupType::Local(LocalType::Pointer {
                            base: binding_type,
                            class: helpers::map_storage_class(space),
                        });

                        let result_id = match self.write_expression_pointer(
                            expr_handle,
                            block,
                            Some(binding_array_false_pointer),
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
                self.writer.global_variables[handle.index()].access_id
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
                            width,
                        } => {
                            self.write_matrix_matrix_column_op(
                                block,
                                id,
                                result_type_id,
                                left_id,
                                right_id,
                                columns,
                                rows,
                                width,
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
                            width,
                        } => {
                            self.write_matrix_matrix_column_op(
                                block,
                                id,
                                result_type_id,
                                left_id,
                                right_id,
                                columns,
                                rows,
                                width,
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
                    Mf::Clamp => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Float) => spirv::GLOp::FClamp,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::SClamp,
                        Some(crate::ScalarKind::Uint) => spirv::GLOp::UClamp,
                        other => unimplemented!("Unexpected max({:?})", other),
                    }),
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
                            let ty = LocalType::Value {
                                vector_size: Some(size),
                                scalar,
                                pointer_space: None,
                            }
                            .into();

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
                                let selector_type_id =
                                    self.get_type_id(LookupType::Local(LocalType::Value {
                                        vector_size: Some(size),
                                        scalar,
                                        pointer_space: None,
                                    }));
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
                            crate::TypeInner::Vector { size, mut scalar } => {
                                scalar.kind = crate::ScalarKind::Uint;
                                let ty = LocalType::Value {
                                    vector_size: Some(size),
                                    scalar,
                                    pointer_space: None,
                                }
                                .into();

                                self.temp_list.clear();
                                self.temp_list.resize(
                                    size as _,
                                    self.writer.get_constant_scalar_with(32, scalar)?,
                                );

                                self.writer.get_constant_composite(ty, &self.temp_list)
                            }
                            crate::TypeInner::Scalar(mut scalar) => {
                                scalar.kind = crate::ScalarKind::Uint;
                                self.writer.get_constant_scalar_with(32, scalar)?
                            }
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
                        let (int_type_id, int_id) = match *arg_ty {
                            crate::TypeInner::Vector { size, mut scalar } => {
                                scalar.kind = crate::ScalarKind::Sint;
                                let ty = LocalType::Value {
                                    vector_size: Some(size),
                                    scalar,
                                    pointer_space: None,
                                }
                                .into();

                                self.temp_list.clear();
                                self.temp_list.resize(
                                    size as _,
                                    self.writer.get_constant_scalar_with(31, scalar)?,
                                );

                                (
                                    self.get_type_id(ty),
                                    self.writer.get_constant_composite(ty, &self.temp_list),
                                )
                            }
                            crate::TypeInner::Scalar(mut scalar) => {
                                scalar.kind = crate::ScalarKind::Sint;
                                (
                                    self.get_type_id(LookupType::Local(LocalType::Value {
                                        vector_size: None,
                                        scalar,
                                        pointer_space: None,
                                    })),
                                    self.writer.get_constant_scalar_with(31, scalar)?,
                                )
                            }
                            _ => unreachable!(),
                        };

                        let msb_id = self.gen_id();
                        block.body.push(Instruction::ext_inst(
                            self.writer.gl450_ext_inst_id,
                            spirv::GLOp::FindUMsb,
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
                        MathOp::Custom(Instruction::ternary(
                            op,
                            result_type_id,
                            id,
                            arg0_id,
                            arg1_id,
                            arg2_id,
                        ))
                    }
                    Mf::InsertBits => MathOp::Custom(Instruction::quaternary(
                        spirv::Op::BitFieldInsert,
                        result_type_id,
                        id,
                        arg0_id,
                        arg1_id,
                        arg2_id,
                        arg3_id,
                    )),
                    Mf::FindLsb => MathOp::Ext(spirv::GLOp::FindILsb),
                    Mf::FindMsb => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Uint) => spirv::GLOp::FindUMsb,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::FindSMsb,
                        other => unimplemented!("Unexpected findMSB({:?})", other),
                    }),
                    Mf::Pack4x8unorm => MathOp::Ext(spirv::GLOp::PackUnorm4x8),
                    Mf::Pack4x8snorm => MathOp::Ext(spirv::GLOp::PackSnorm4x8),
                    Mf::Pack2x16float => MathOp::Ext(spirv::GLOp::PackHalf2x16),
                    Mf::Pack2x16unorm => MathOp::Ext(spirv::GLOp::PackUnorm2x16),
                    Mf::Pack2x16snorm => MathOp::Ext(spirv::GLOp::PackSnorm2x16),
                    Mf::Unpack4x8unorm => MathOp::Ext(spirv::GLOp::UnpackUnorm4x8),
                    Mf::Unpack4x8snorm => MathOp::Ext(spirv::GLOp::UnpackSnorm4x8),
                    Mf::Unpack2x16float => MathOp::Ext(spirv::GLOp::UnpackHalf2x16),
                    Mf::Unpack2x16unorm => MathOp::Ext(spirv::GLOp::UnpackUnorm2x16),
                    Mf::Unpack2x16snorm => MathOp::Ext(spirv::GLOp::UnpackSnorm2x16),
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
                match self.write_expression_pointer(pointer, block, None)? {
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
                        id
                    }
                    ExpressionPointer::Conditional { condition, access } => {
                        //TODO: support atomics?
                        self.write_conditional_indexed_load(
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
                        )
                    }
                }
            }
            crate::Expression::FunctionArgument(index) => self.function.parameter_id(index),
            crate::Expression::CallResult(_)
            | crate::Expression::AtomicResult { .. }
            | crate::Expression::WorkGroupUniformLoadResult { .. }
            | crate::Expression::RayQueryProceedResult => self.cached[expr_handle],
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
                        crate::TypeInner::Matrix { width, .. } => {
                            (crate::Scalar::float(width), None, true)
                        }
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
                                Sk::Bool => unreachable!(),
                            };
                            let zero_scalar_id =
                                self.writer.get_constant_scalar_with(0, src_scalar)?;
                            let zero_id = match src_size {
                                Some(size) => {
                                    let ty = LocalType::Value {
                                        vector_size: Some(size),
                                        scalar: src_scalar,
                                        pointer_space: None,
                                    }
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
                                    let ty = LocalType::Value {
                                        vector_size: Some(size),
                                        scalar: dst_scalar,
                                        pointer_space: None,
                                    }
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

                    let bool_vector_type_id =
                        self.get_type_id(LookupType::Local(LocalType::Value {
                            vector_size: Some(size),
                            scalar: condition_scalar,
                            pointer_space: None,
                        }));

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
    /// Some cases we need to generate a different return type than what the IR gives us.
    /// This is because pointers to binding arrays of handles (such as images or samplers)
    /// don't exist in the IR, but we need to create them to create an access chain in SPIRV.
    ///
    /// On success, the return value is an [`ExpressionPointer`] value; see the
    /// documentation for that type.
    fn write_expression_pointer(
        &mut self,
        mut expr_handle: Handle<crate::Expression>,
        block: &mut Block,
        return_type_override: Option<LookupType>,
    ) -> Result<ExpressionPointer, Error> {
        let result_lookup_ty = match self.fun_info[expr_handle].ty {
            TypeResolution::Handle(ty_handle) => match return_type_override {
                // We use the return type override as a special case for handle binding arrays as the OpAccessChain
                // needs to return a pointer, but indexing into a handle binding array just gives you the type of
                // the binding in the IR.
                Some(ty) => ty,
                None => LookupType::Handle(ty_handle),
            },
            TypeResolution::Value(ref inner) => LookupType::Local(make_local(inner).unwrap()),
        };
        let result_type_id = self.get_type_id(result_lookup_ty);

        // The id of the boolean `and` of all dynamic bounds checks up to this point. If
        // `None`, then we haven't done any dynamic bounds checks yet.
        //
        // When we have a chain of bounds checks, we combine them with `OpLogicalAnd`, not
        // a short-circuit branch. This means we might do comparisons we don't need to,
        // but we expect these checks to almost always succeed, and keeping branches to a
        // minimum is essential.
        let mut accumulated_checks = None;
        // Is true if we are accessing into a binding array of buffers with a non-uniform index.
        let mut is_non_uniform_binding_array = false;

        self.temp_list.clear();
        let root_id = loop {
            expr_handle = match self.ir_function.expressions[expr_handle] {
                crate::Expression::Access { base, index } => {
                    if let crate::Expression::GlobalVariable(var_handle) =
                        self.ir_function.expressions[base]
                    {
                        let gvar: &crate::GlobalVariable =
                            &self.ir_module.global_variables[var_handle];
                        match gvar.space {
                            crate::AddressSpace::Storage { .. } | crate::AddressSpace::Uniform => {
                                if let crate::TypeInner::BindingArray { .. } =
                                    self.ir_module.types[gvar.ty].inner
                                {
                                    is_non_uniform_binding_array = self.fun_info[index]
                                        .uniformity
                                        .non_uniform_result
                                        .is_some();
                                }
                            }
                            _ => {}
                        }
                    }

                    let index_id = match self.write_bounds_check(base, index, block)? {
                        BoundsCheckResult::KnownInBounds(known_index) => {
                            // Even if the index is known, `OpAccessIndex`
                            // requires expression operands, not literals.
                            let scalar = crate::Literal::U32(known_index);
                            self.writer.get_constant_scalar(scalar)
                        }
                        BoundsCheckResult::Computed(computed_index_id) => computed_index_id,
                        BoundsCheckResult::Conditional(comparison_id) => {
                            match accumulated_checks {
                                Some(prior_checks) => {
                                    let combined = self.gen_id();
                                    block.body.push(Instruction::binary(
                                        spirv::Op::LogicalAnd,
                                        self.writer.get_bool_type_id(),
                                        combined,
                                        prior_checks,
                                        comparison_id,
                                    ));
                                    accumulated_checks = Some(combined);
                                }
                                None => {
                                    // Start a fresh chain of checks.
                                    accumulated_checks = Some(comparison_id);
                                }
                            }

                            // Either way, the index to use is unchanged.
                            self.cached[index]
                        }
                    };
                    self.temp_list.push(index_id);
                    base
                }
                crate::Expression::AccessIndex { base, index } => {
                    let const_id = self.get_index_constant(index);
                    self.temp_list.push(const_id);
                    base
                }
                crate::Expression::GlobalVariable(handle) => {
                    let gv = &self.writer.global_variables[handle.index()];
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
        // if the buffer binding array was accessed with a non-uniform index
        // see VUID-RuntimeSpirv-NonUniform-06274
        if is_non_uniform_binding_array {
            self.writer
                .decorate_non_uniform_binding_array_access(pointer_id)?;
        }

        Ok(expr_pointer)
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

        let vector_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: Some(rows),
            scalar: crate::Scalar::float(width),
            pointer_space: None,
        }));

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

    pub(super) fn write_block(
        &mut self,
        label_id: Word,
        naga_block: &crate::Block,
        exit: BlockExit,
        loop_context: LoopContext,
        debug_info: Option<&DebugInfoInner>,
    ) -> Result<(), Error> {
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
                crate::Statement::Emit(ref range) => {
                    for handle in range.clone() {
                        // omit const expressions as we've already cached those
                        if !self.expression_constness.is_const(handle) {
                            self.cache_expression_value(handle, &mut block)?;
                        }
                    }
                }
                crate::Statement::Block(ref block_statements) => {
                    let scope_id = self.gen_id();
                    self.function.consume(block, Instruction::branch(scope_id));

                    let merge_id = self.gen_id();
                    self.write_block(
                        scope_id,
                        block_statements,
                        BlockExit::Branch { target: merge_id },
                        loop_context,
                        debug_info,
                    )?;

                    block = Block::new(merge_id);
                }
                crate::Statement::If {
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
                        self.write_block(
                            block_id,
                            accept,
                            BlockExit::Branch { target: merge_id },
                            loop_context,
                            debug_info,
                        )?;
                    }
                    if let Some(block_id) = reject_id {
                        self.write_block(
                            block_id,
                            reject,
                            BlockExit::Branch { target: merge_id },
                            loop_context,
                            debug_info,
                        )?;
                    }

                    block = Block::new(merge_id);
                }
                crate::Statement::Switch {
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
                        self.write_block(
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
                crate::Statement::Loop {
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

                    self.write_block(
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

                    self.write_block(
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
                crate::Statement::Break => {
                    self.function
                        .consume(block, Instruction::branch(loop_context.break_id.unwrap()));
                    return Ok(());
                }
                crate::Statement::Continue => {
                    self.function.consume(
                        block,
                        Instruction::branch(loop_context.continuing_id.unwrap()),
                    );
                    return Ok(());
                }
                crate::Statement::Return { value: Some(value) } => {
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
                    return Ok(());
                }
                crate::Statement::Return { value: None } => {
                    self.function.consume(block, Instruction::return_void());
                    return Ok(());
                }
                crate::Statement::Kill => {
                    self.function.consume(block, Instruction::kill());
                    return Ok(());
                }
                crate::Statement::Barrier(flags) => {
                    self.writer.write_barrier(flags, &mut block);
                }
                crate::Statement::Store { pointer, value } => {
                    let value_id = self.cached[value];
                    match self.write_expression_pointer(pointer, &mut block, None)? {
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
                crate::Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => self.write_image_store(image, coordinate, array_index, value, &mut block)?,
                crate::Statement::Call {
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
                crate::Statement::Atomic {
                    pointer,
                    ref fun,
                    value,
                    result,
                } => {
                    let id = self.gen_id();
                    let result_type_id = self.get_expression_type_id(&self.fun_info[result].ty);

                    self.cached[result] = id;

                    let pointer_id =
                        match self.write_expression_pointer(pointer, &mut block, None)? {
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
                                    self.get_type_id(LookupType::Local(LocalType::Value {
                                        vector_size: None,
                                        scalar,
                                        pointer_space: None,
                                    }))
                                }
                                _ => unimplemented!(),
                            };
                            let bool_type_id =
                                self.get_type_id(LookupType::Local(LocalType::Value {
                                    vector_size: None,
                                    scalar: crate::Scalar::BOOL,
                                    pointer_space: None,
                                }));

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
                crate::Statement::WorkGroupUniformLoad { pointer, result } => {
                    self.writer
                        .write_barrier(crate::Barrier::WORK_GROUP, &mut block);
                    let result_type_id = self.get_expression_type_id(&self.fun_info[result].ty);
                    // Embed the body of
                    match self.write_expression_pointer(pointer, &mut block, None)? {
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
                crate::Statement::RayQuery { query, ref fun } => {
                    self.write_ray_query_function(query, fun, &mut block);
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
        Ok(())
    }
}
