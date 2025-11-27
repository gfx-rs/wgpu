/*!
Module for code shared between ray queries and ray tracing pipeline code.
Ray tracing pipelines are not yet implemented, so this is empty.
*/

pub mod query;

use alloc::vec::Vec;

use super::{Block, Function, FunctionArgument, Instruction, LookupFunctionType, Writer};

/// helper function to check if a particular flag is set in a u32.
fn write_ray_flags_contains_flags(
    writer: &mut Writer,
    block: &mut Block,
    id: spirv::Word,
    flag: u32,
) -> spirv::Word {
    let bit_id = writer.get_constant_scalar(crate::Literal::U32(flag));
    let zero_id = writer.get_constant_scalar(crate::Literal::U32(0));
    let u32_type_id = writer.get_u32_type_id();
    let bool_ty = writer.get_bool_type_id();

    let and_id = writer.id_gen.next();
    block.body.push(Instruction::binary(
        spirv::Op::BitwiseAnd,
        u32_type_id,
        and_id,
        id,
        bit_id,
    ));

    let eq_id = writer.id_gen.next();
    block.body.push(Instruction::binary(
        spirv::Op::INotEqual,
        bool_ty,
        eq_id,
        and_id,
        zero_id,
    ));

    eq_id
}

impl Writer {
    /// writes a logical and of two scalar booleans
    fn write_logical_and(
        &mut self,
        block: &mut Block,
        one: spirv::Word,
        two: spirv::Word,
    ) -> spirv::Word {
        let id = self.id_gen.next();
        let bool_id = self.get_bool_type_id();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalAnd,
            bool_id,
            id,
            one,
            two,
        ));
        id
    }

    fn write_reduce_and(&mut self, block: &mut Block, mut bools: Vec<spirv::Word>) -> spirv::Word {
        // The combined `and`ed together of all of the bools up to this point.
        let mut current_combined = bools.pop().unwrap();
        for boolean in bools {
            current_combined = self.write_logical_and(block, current_combined, boolean)
        }
        current_combined
    }

    // returns the id of the function, the function, and ids for its arguments.
    fn write_function_signature(
        &mut self,
        arg_types: &[spirv::Word],
        return_ty: spirv::Word,
    ) -> (spirv::Word, Function, Vec<spirv::Word>) {
        let func_ty = self.get_function_type(LookupFunctionType {
            parameter_type_ids: Vec::from(arg_types),
            return_type_id: return_ty,
        });

        let mut function = Function::default();
        let func_id = self.id_gen.next();
        function.signature = Some(Instruction::function(
            return_ty,
            func_id,
            spirv::FunctionControl::empty(),
            func_ty,
        ));

        let mut arg_ids = Vec::with_capacity(arg_types.len());

        for (idx, &arg_ty) in arg_types.iter().enumerate() {
            let id = self.id_gen.next();
            let instruction = Instruction::function_parameter(arg_ty, id);
            function.parameters.push(FunctionArgument {
                instruction,
                handle_id: idx as u32,
            });
            arg_ids.push(id);
        }
        (func_id, function, arg_ids)
    }
}
