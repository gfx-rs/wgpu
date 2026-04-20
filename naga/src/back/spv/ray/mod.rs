/*!
Module for code shared between ray queries and ray tracing pipeline code.
*/

pub mod pipeline;
pub mod query;

use alloc::{vec, vec::Vec};

use super::{Block, Function, FunctionArgument, Instruction, LookupFunctionType, Writer};

struct ExtractedRayDesc {
    ray_flags_id: spirv::Word,
    cull_mask_id: spirv::Word,
    tmin_id: spirv::Word,
    tmax_id: spirv::Word,
    ray_origin_id: spirv::Word,
    ray_dir_id: spirv::Word,
    valid_id: Option<spirv::Word>,
}

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
    fn write_extract_ray_desc(
        &mut self,
        block: &mut Block,
        desc_id: spirv::Word,
        validate: bool,
    ) -> ExtractedRayDesc {
        let bool_type_id = self.get_bool_type_id();
        let bool_vec3_type_id = self.get_vec3_bool_type_id();
        let f32_type_id = self.get_f32_type_id();
        let flag_type_id = self.get_numeric_type_id(super::NumericType::Scalar(crate::Scalar::U32));

        //Note: composite extract indices and types must match `generate_ray_desc_type`
        let ray_flags_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            flag_type_id,
            ray_flags_id,
            desc_id,
            &[0],
        ));
        let cull_mask_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            flag_type_id,
            cull_mask_id,
            desc_id,
            &[1],
        ));

        let tmin_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            f32_type_id,
            tmin_id,
            desc_id,
            &[2],
        ));
        let tmax_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            f32_type_id,
            tmax_id,
            desc_id,
            &[3],
        ));

        let vector_type_id = self.get_numeric_type_id(super::NumericType::Vector {
            size: crate::VectorSize::Tri,
            scalar: crate::Scalar::F32,
        });
        let ray_origin_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            vector_type_id,
            ray_origin_id,
            desc_id,
            &[4],
        ));
        let ray_dir_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            vector_type_id,
            ray_dir_id,
            desc_id,
            &[5],
        ));

        let valid_id = validate.then(||{
            let tmin_le_tmax_id = self.id_gen.next();
            // Check both that tmin is less than or equal to tmax (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06350)
            // and implicitly that neither tmin or tmax are NaN (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06351)
            // because this checks if tmin and tmax are ordered too (i.e: not NaN).
            block.body.push(Instruction::binary(
                spirv::Op::FOrdLessThanEqual,
                bool_type_id,
                tmin_le_tmax_id,
                tmin_id,
                tmax_id,
            ));

            // Check that tmin is greater than or equal to 0 (and
            // therefore also tmax is too because it is greater than
            // or equal to tmin) (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06349).
            let tmin_ge_zero_id = self.id_gen.next();
            let zero_id = self.get_constant_scalar(crate::Literal::F32(0.0));
            block.body.push(Instruction::binary(
                spirv::Op::FOrdGreaterThanEqual,
                bool_type_id,
                tmin_ge_zero_id,
                tmin_id,
                zero_id,
            ));

            // Check that ray origin is finite (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06348)
            let ray_origin_infinite_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::IsInf,
                bool_vec3_type_id,
                ray_origin_infinite_id,
                ray_origin_id,
            ));
            let any_ray_origin_infinite_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::Any,
                bool_type_id,
                any_ray_origin_infinite_id,
                ray_origin_infinite_id,
            ));

            let ray_origin_nan_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::IsNan,
                bool_vec3_type_id,
                ray_origin_nan_id,
                ray_origin_id,
            ));
            let any_ray_origin_nan_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::Any,
                bool_type_id,
                any_ray_origin_nan_id,
                ray_origin_nan_id,
            ));

            let ray_origin_not_finite_id = self.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::LogicalOr,
                bool_type_id,
                ray_origin_not_finite_id,
                any_ray_origin_nan_id,
                any_ray_origin_infinite_id,
            ));

            let all_ray_origin_finite_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::LogicalNot,
                bool_type_id,
                all_ray_origin_finite_id,
                ray_origin_not_finite_id,
            ));

            // Check that ray direction is finite (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06348)
            let ray_dir_infinite_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::IsInf,
                bool_vec3_type_id,
                ray_dir_infinite_id,
                ray_dir_id,
            ));
            let any_ray_dir_infinite_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::Any,
                bool_type_id,
                any_ray_dir_infinite_id,
                ray_dir_infinite_id,
            ));

            let ray_dir_nan_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::IsNan,
                bool_vec3_type_id,
                ray_dir_nan_id,
                ray_dir_id,
            ));
            let any_ray_dir_nan_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::Any,
                bool_type_id,
                any_ray_dir_nan_id,
                ray_dir_nan_id,
            ));

            let ray_dir_not_finite_id = self.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::LogicalOr,
                bool_type_id,
                ray_dir_not_finite_id,
                any_ray_dir_nan_id,
                any_ray_dir_infinite_id,
            ));

            let all_ray_dir_finite_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::LogicalNot,
                bool_type_id,
                all_ray_dir_finite_id,
                ray_dir_not_finite_id,
            ));

            /// Writes spirv to check that less than two booleans are true
            ///
            /// For each boolean: removes it, `and`s it with all others (i.e for all possible combinations of two booleans in the list checks to see if both are true).
            /// Then `or`s all of these checks together. This produces whether two or more booleans are true.
            fn write_less_than_2_true(
                writer: &mut Writer,
                block: &mut Block,
                mut bools: Vec<spirv::Word>,
            ) -> spirv::Word {
                assert!(bools.len() > 1, "Must have multiple booleans!");
                let bool_ty = writer.get_bool_type_id();
                let mut each_two_true = Vec::new();
                while let Some(last_bool) = bools.pop() {
                    for &bool in &bools {
                        let both_true_id = writer.write_logical_and(
                            block,
                            last_bool,
                            bool,
                        );
                        each_two_true.push(both_true_id);
                    }
                }
                let mut all_or_id = each_two_true.pop().expect("since this must have multiple booleans, there must be at least one thing in `each_two_true`");
                for two_true in each_two_true {
                    let new_all_or_id = writer.id_gen.next();
                    block.body.push(Instruction::binary(
                        spirv::Op::LogicalOr,
                        bool_ty,
                        new_all_or_id,
                        all_or_id,
                        two_true,
                    ));
                    all_or_id = new_all_or_id;
                }

                let less_than_two_id = writer.id_gen.next();
                block.body.push(Instruction::unary(
                    spirv::Op::LogicalNot,
                    bool_ty,
                    less_than_two_id,
                    all_or_id,
                ));
                less_than_two_id
            }

            // Check that at most one of skip triangles and skip AABBs is
            // present (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06889)
            let contains_skip_triangles = write_ray_flags_contains_flags(
                self,
                block,
                ray_flags_id,
                crate::RayFlag::SKIP_TRIANGLES.bits(),
            );
            let contains_skip_aabbs = write_ray_flags_contains_flags(
                self,
                block,
                ray_flags_id,
                crate::RayFlag::SKIP_AABBS.bits(),
            );

            let not_contain_skip_triangles_aabbs = write_less_than_2_true(
                self,
                block,
                vec![contains_skip_triangles, contains_skip_aabbs],
            );

            // Check that at most one of skip triangles (taken from above check),
            // cull back facing, and cull front face is present (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06890)
            let contains_cull_back = write_ray_flags_contains_flags(
                self,
                block,
                ray_flags_id,
                crate::RayFlag::CULL_BACK_FACING.bits(),
            );
            let contains_cull_front = write_ray_flags_contains_flags(
                self,
                block,
                ray_flags_id,
                crate::RayFlag::CULL_FRONT_FACING.bits(),
            );

            let not_contain_skip_triangles_cull = write_less_than_2_true(
                self,
                block,
                vec![
                    contains_skip_triangles,
                    contains_cull_back,
                    contains_cull_front,
                ],
            );

            // Check that at most one of force opaque, force not opaque, cull opaque,
            // and cull not opaque are present (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06891)
            let contains_opaque = write_ray_flags_contains_flags(
                self,
                block,
                ray_flags_id,
                crate::RayFlag::FORCE_OPAQUE.bits(),
            );
            let contains_no_opaque = write_ray_flags_contains_flags(
                self,
                block,
                ray_flags_id,
                crate::RayFlag::FORCE_NO_OPAQUE.bits(),
            );
            let contains_cull_opaque = write_ray_flags_contains_flags(
                self,
                block,
                ray_flags_id,
                crate::RayFlag::CULL_OPAQUE.bits(),
            );
            let contains_cull_no_opaque = write_ray_flags_contains_flags(
                self,
                block,
                ray_flags_id,
                crate::RayFlag::CULL_NO_OPAQUE.bits(),
            );

            let not_contain_multiple_opaque = write_less_than_2_true(
                self,
                block,
                vec![
                    contains_opaque,
                    contains_no_opaque,
                    contains_cull_opaque,
                    contains_cull_no_opaque,
                ],
            );

            // Combine all checks into a single flag saying whether the call is valid or not.
            self.write_reduce_and(
                block,
                vec![
                    tmin_le_tmax_id,
                    tmin_ge_zero_id,
                    all_ray_origin_finite_id,
                    all_ray_dir_finite_id,
                    not_contain_skip_triangles_aabbs,
                    not_contain_skip_triangles_cull,
                    not_contain_multiple_opaque,
                ],
            )
        });

        ExtractedRayDesc {
            ray_flags_id,
            cull_mask_id,
            tmin_id,
            tmax_id,
            ray_origin_id,
            ray_dir_id,
            valid_id,
        }
    }
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
