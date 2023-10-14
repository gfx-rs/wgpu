use super::{Block, BlockContext, Error, Instruction};
use crate::{arena::Handle, TypeInner};

impl<'w> BlockContext<'w> {
    pub(super) fn write_subgroup_operation(
        &mut self,
        op: &crate::SubgroupOperation,
        collective_op: &crate::CollectiveOperation,
        argument: Handle<crate::Expression>,
        result: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        use crate::SubgroupOperation as sg;
        match op {
            sg::All | sg::Any => {
                self.writer.require_any(
                    "GroupNonUniformVote",
                    &[spirv::Capability::GroupNonUniformVote],
                )?;
            }
            _ => {
                self.writer.require_any(
                    "GroupNonUniformArithmetic",
                    &[
                        spirv::Capability::GroupNonUniformArithmetic,
                        spirv::Capability::GroupNonUniformClustered,
                        spirv::Capability::GroupNonUniformPartitionedNV,
                    ],
                )?;
            }
        }

        let id = self.gen_id();
        let result_ty = &self.fun_info[result].ty;
        let result_type_id = self.get_expression_type_id(result_ty);
        let result_ty_inner = result_ty.inner_with(&self.ir_module.types);

        let (is_scalar, kind) = match result_ty_inner {
            TypeInner::Scalar { kind, .. } => (true, kind),
            TypeInner::Vector { kind, .. } => (false, kind),
            _ => unimplemented!(),
        };

        use crate::ScalarKind as sk;
        let spirv_op = match (kind, op) {
            (sk::Bool, sg::All) if is_scalar => spirv::Op::GroupNonUniformAll,
            (sk::Bool, sg::Any) if is_scalar => spirv::Op::GroupNonUniformAny,
            (_, sg::All | sg::Any) => unimplemented!(),

            (sk::Sint | sk::Uint, sg::Add) => spirv::Op::GroupNonUniformIAdd,
            (sk::Float, sg::Add) => spirv::Op::GroupNonUniformFAdd,
            (sk::Sint | sk::Uint, sg::Mul) => spirv::Op::GroupNonUniformIMul,
            (sk::Float, sg::Mul) => spirv::Op::GroupNonUniformFMul,
            (sk::Sint, sg::Max) => spirv::Op::GroupNonUniformSMax,
            (sk::Uint, sg::Max) => spirv::Op::GroupNonUniformUMax,
            (sk::Float, sg::Max) => spirv::Op::GroupNonUniformFMax,
            (sk::Sint, sg::Min) => spirv::Op::GroupNonUniformSMin,
            (sk::Uint, sg::Min) => spirv::Op::GroupNonUniformUMin,
            (sk::Float, sg::Min) => spirv::Op::GroupNonUniformFMin,
            (sk::Bool, sg::Add | sg::Mul | sg::Min | sg::Max) => unimplemented!(),

            (sk::Sint | sk::Uint, sg::And) => spirv::Op::GroupNonUniformBitwiseAnd,
            (sk::Sint | sk::Uint, sg::Or) => spirv::Op::GroupNonUniformBitwiseOr,
            (sk::Sint | sk::Uint, sg::Xor) => spirv::Op::GroupNonUniformBitwiseXor,
            (sk::Float, sg::And | sg::Or | sg::Xor) => unimplemented!(),
            (sk::Bool, sg::And) => spirv::Op::GroupNonUniformLogicalAnd,
            (sk::Bool, sg::Or) => spirv::Op::GroupNonUniformLogicalOr,
            (sk::Bool, sg::Xor) => spirv::Op::GroupNonUniformLogicalXor,
        };

        let exec_scope_id = self.get_index_constant(spirv::Scope::Subgroup as u32);

        use crate::CollectiveOperation as c;
        let group_op = match op {
            sg::All | sg::Any => None,
            _ => Some(match collective_op {
                c::Reduce => spirv::GroupOperation::Reduce,
                c::InclusiveScan => spirv::GroupOperation::InclusiveScan,
                c::ExclusiveScan => spirv::GroupOperation::ExclusiveScan,
            }),
        };

        let arg_id = self.cached[argument];
        block.body.push(Instruction::group_non_uniform_arithmetic(
            spirv_op,
            result_type_id,
            id,
            exec_scope_id,
            group_op,
            arg_id,
        ));
        self.cached[result] = id;
        Ok(())
    }
    pub(super) fn write_subgroup_broadcast(
        &mut self,
        mode: &crate::GatherMode,
        argument: Handle<crate::Expression>,
        result: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        self.writer.require_any(
            "GroupNonUniformBallot",
            &[spirv::Capability::GroupNonUniformBallot],
        )?;

        let id = self.gen_id();
        let result_ty = &self.fun_info[result].ty;
        let result_type_id = self.get_expression_type_id(result_ty);

        let exec_scope_id = self.get_index_constant(spirv::Scope::Subgroup as u32);

        let arg_id = self.cached[argument];
        match mode {
            crate::GatherMode::Broadcast(index) => {
                let index_id = self.cached[*index];
                block.body.push(Instruction::group_non_uniform_broadcast(
                    result_type_id,
                    id,
                    exec_scope_id,
                    arg_id,
                    index_id,
                ));
            }
            crate::GatherMode::BroadcastFirst => {
                block
                    .body
                    .push(Instruction::group_non_uniform_broadcast_first(
                        result_type_id,
                        id,
                        exec_scope_id,
                        arg_id,
                    ));
            }
            crate::GatherMode::Shuffle(index)
            | crate::GatherMode::ShuffleDown(index)
            | crate::GatherMode::ShuffleUp(index)
            | crate::GatherMode::ShuffleXor(index) => todo!(),
        }
        self.cached[result] = id;
        Ok(())
    }
}
