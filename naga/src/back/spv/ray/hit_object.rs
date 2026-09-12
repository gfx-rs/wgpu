/*!
Generating SPIR-V for ray tracing invocation reordering (hit objects).

This implements `SPV_EXT_shader_invocation_reorder`, which is exposed in WGSL by
the `wgpu_ray_tracing_invocation_reorder` enable-extension.
*/

use super::write_ray_flags_contains_flags;
use crate::back::spv::{
    Block, BlockContext, Instruction, LocalType, LookupRaytracingFunction, NumericType, Writer,
    WriterFlags,
};
use crate::back::RayQueryPoint;

/// Values of `OpHitObjectGetHitKindEXT` greater than or equal to this indicate a
/// triangle hit; smaller values come from intersection shaders.
///
/// `0xFE` is a front-facing triangle and `0xFF` is a back-facing one.
const HIT_KIND_FRONT_FACING_TRIANGLE: u32 = 0xFE;

impl Writer {
    /// Write a helper function that traces a ray into a hit object.
    ///
    /// The generated function takes a pointer to the hit object, the
    /// acceleration structure and a `RayDesc`. The payload is baked into the
    /// function, exactly like [`Writer::write_trace_ray`].
    fn write_hit_object_trace_ray(
        &mut self,
        ir_module: &crate::Module,
        payload: crate::Handle<crate::GlobalVariable>,
    ) -> Result<spirv::Word, super::super::Error> {
        if let Some(&word) = self
            .ray_tracing_functions
            .get(&LookupRaytracingFunction::HitObjectTraceRay { payload })
        {
            return Ok(word);
        }

        self.require_hit_objects()?;

        let hit_object_pointer_type_id = self.get_hit_object_pointer_id();
        let acceleration_structure_type_id =
            self.get_localtype_id(LocalType::AccelerationStructure);
        let ray_desc_type_id = self.get_handle_type_id(
            ir_module
                .special_types
                .ray_desc
                .expect("ray desc should be set if `hitObjectTraceRay` is called"),
        );

        let (func_id, mut function, arg_ids) = self.write_function_signature(
            &[
                hit_object_pointer_type_id,
                acceleration_structure_type_id,
                ray_desc_type_id,
            ],
            self.void_type,
        );

        let hit_object_id = arg_ids[0];
        let acceleration_structure_id = arg_ids[1];
        let desc_id = arg_ids[2];
        let payload_id = self.global_variables[payload].access_id;

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        let super::ExtractedRayDesc {
            ray_flags_id,
            cull_mask_id,
            tmin_id,
            tmax_id,
            ray_origin_id,
            ray_dir_id,
            valid_id,
        } = self.write_extract_ray_desc(&mut block, desc_id, self.trace_ray_argument_validation);

        let merge_label_id = self.id_gen.next();
        let merge_block = Block::new(merge_label_id);

        // NOTE: this block will be unreachable if trace ray validation is disabled.
        let invalid_label_id = self.id_gen.next();
        let mut invalid_block = Block::new(invalid_label_id);

        let valid_label_id = self.id_gen.next();
        let mut valid_block = Block::new(valid_label_id);

        match valid_id {
            Some(all_valid_id) => {
                block.body.push(Instruction::selection_merge(
                    merge_label_id,
                    spirv::SelectionControl::NONE,
                ));
                function.consume(
                    block,
                    Instruction::branch_conditional(all_valid_id, valid_label_id, invalid_label_id),
                );
            }
            None => {
                function.consume(block, Instruction::branch(valid_label_id));
            }
        }

        let zero = self.get_constant_scalar(crate::Literal::U32(0));

        valid_block.body.push(Instruction::hit_object_trace_ray(
            hit_object_id,
            acceleration_structure_id,
            ray_flags_id,
            cull_mask_id,
            zero,
            zero,
            zero,
            ray_origin_id,
            tmin_id,
            ray_dir_id,
            tmax_id,
            payload_id,
        ));

        function.consume(valid_block, Instruction::branch(merge_label_id));

        if self.flags.contains(WriterFlags::PRINT_ON_TRACE_RAYS_FAIL) {
            self.write_debug_printf(
                &mut invalid_block,
                "Naga ignored invalid arguments to hitObjectTraceRay with flags: %u t_min: %f t_max: %f origin: %v4f dir: %v4f",
                &[
                    ray_flags_id,
                    tmin_id,
                    tmax_id,
                    ray_origin_id,
                    ray_dir_id,
                ],
            );
        }

        // Leave the hit object in a well-defined state even when we skipped the
        // trace, so that later queries on it are not reading uninitialized data.
        invalid_block
            .body
            .push(Instruction::hit_object_record_empty(hit_object_id));

        function.consume(invalid_block, Instruction::branch(merge_label_id));

        function.consume(merge_block, Instruction::return_void());

        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_tracing_functions.insert(
            LookupRaytracingFunction::HitObjectTraceRay { payload },
            func_id,
        );

        Ok(func_id)
    }

    /// Write a helper function that records a miss in a hit object.
    ///
    /// The generated function takes a pointer to the hit object and a
    /// `RayDesc`. The descriptor's cull mask is unused, and the miss index is
    /// always zero.
    fn write_hit_object_record_miss(
        &mut self,
        ir_module: &crate::Module,
    ) -> Result<spirv::Word, super::super::Error> {
        if let Some(&word) = self
            .ray_tracing_functions
            .get(&LookupRaytracingFunction::HitObjectRecordMiss)
        {
            return Ok(word);
        }

        self.require_hit_objects()?;

        let hit_object_pointer_type_id = self.get_hit_object_pointer_id();
        let ray_desc_type_id = self.get_handle_type_id(
            ir_module
                .special_types
                .ray_desc
                .expect("ray desc should be set if `hitObjectRecordMiss` is called"),
        );

        let (func_id, mut function, arg_ids) = self.write_function_signature(
            &[hit_object_pointer_type_id, ray_desc_type_id],
            self.void_type,
        );

        let hit_object_id = arg_ids[0];
        let desc_id = arg_ids[1];

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        let super::ExtractedRayDesc {
            ray_flags_id,
            cull_mask_id: _,
            tmin_id,
            tmax_id,
            ray_origin_id,
            ray_dir_id,
            valid_id: _,
        } = self.write_extract_ray_desc(&mut block, desc_id, false);

        let zero = self.get_constant_scalar(crate::Literal::U32(0));

        block.body.push(Instruction::hit_object_record_miss(
            hit_object_id,
            ray_flags_id,
            zero,
            ray_origin_id,
            tmin_id,
            ray_dir_id,
            tmax_id,
        ));

        function.consume(block, Instruction::return_void());
        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_tracing_functions
            .insert(LookupRaytracingFunction::HitObjectRecordMiss, func_id);

        Ok(func_id)
    }

    /// Write a helper function that records a ray query's committed
    /// intersection in a hit object.
    ///
    /// The generated function takes a pointer to the hit object, a pointer to
    /// the ray query, a pointer to that query's initialization tracker, and a
    /// pointer to that query's `t_max` tracker.
    ///
    /// `OpRayQueryGetIntersection*` may only be used on a query that has
    /// finished traversal, so the tracker is checked first exactly as
    /// [`Writer::write_ray_query_get_intersection_function`] does for committed
    /// intersections; if the query is not usable the hit object is recorded as
    /// empty instead.
    ///
    /// `OpHitObjectRecordFromQueryEXT` is only defined for a query whose
    /// committed intersection is a hit. When the query has no committed
    /// intersection we instead record a miss with `OpHitObjectRecordMissEXT`,
    /// using the ray the query was initialized with and a miss index of zero,
    /// which is what `hitObjectTraceRay` would have recorded.
    ///
    /// For a hit, the attribute variable is the source of the recorded
    /// attributes when the committed intersection is procedural (an AABB), so
    /// it is zeroed before recording; for triangles the barycentrics come from
    /// the query. See [`Instruction::hit_object_record_from_query`] for why the
    /// `Hit Kind` operand is not emitted.
    fn write_hit_object_record_from_query(&mut self) -> Result<spirv::Word, super::super::Error> {
        if let Some(&word) = self
            .ray_tracing_functions
            .get(&LookupRaytracingFunction::HitObjectRecordFromQuery)
        {
            return Ok(word);
        }

        self.require_hit_objects()?;

        let hit_object_pointer_type_id = self.get_hit_object_pointer_id();
        let ray_query_pointer_type_id = self.get_ray_query_pointer_id();
        let u32_type_id = self.get_u32_type_id();
        let u32_pointer_type_id =
            self.get_pointer_type_id(u32_type_id, spirv::StorageClass::Function);
        let f32_type_id = self.get_f32_type_id();
        let f32_pointer_type_id =
            self.get_pointer_type_id(f32_type_id, spirv::StorageClass::Function);
        let vec3_type_id = self.get_numeric_type_id(NumericType::Vector {
            size: crate::VectorSize::Tri,
            scalar: crate::Scalar::F32,
        });
        let barycentrics_type_id = self.get_numeric_type_id(NumericType::Vector {
            size: crate::VectorSize::Bi,
            scalar: crate::Scalar::F32,
        });
        let bool_type_id = self.get_bool_type_id();

        let attribute_var_id = self.get_hit_object_attribute_var();

        let (func_id, mut function, arg_ids) = self.write_function_signature(
            &[
                hit_object_pointer_type_id,
                ray_query_pointer_type_id,
                u32_pointer_type_id,
                f32_pointer_type_id,
            ],
            self.void_type,
        );

        let hit_object_id = arg_ids[0];
        let query_id = arg_ids[1];
        let tracker_id = arg_ids[2];
        let t_max_tracker_id = arg_ids[3];

        let entry_label_id = self.id_gen.next();
        let mut entry_block = Block::new(entry_label_id);

        let loaded_tracker_id = self.id_gen.next();
        entry_block.body.push(Instruction::load(
            u32_type_id,
            loaded_tracker_id,
            tracker_id,
            None,
        ));
        let finished_traversal_id = write_ray_flags_contains_flags(
            self,
            &mut entry_block,
            loaded_tracker_id,
            RayQueryPoint::FINISHED_TRAVERSAL.bits(),
        );

        let merge_label_id = self.id_gen.next();
        let merge_block = Block::new(merge_label_id);

        let unusable_label_id = self.id_gen.next();
        let mut unusable_block = Block::new(unusable_label_id);

        let usable_label_id = self.id_gen.next();
        let mut usable_block = Block::new(usable_label_id);

        entry_block.body.push(Instruction::selection_merge(
            merge_label_id,
            spirv::SelectionControl::NONE,
        ));
        function.consume(
            entry_block,
            Instruction::branch_conditional(
                finished_traversal_id,
                usable_label_id,
                unusable_label_id,
            ),
        );

        let committed_id = self.get_constant_scalar(crate::Literal::U32(
            spirv::RayQueryIntersection::RayQueryCommittedIntersectionKHR as _,
        ));
        let kind_id = self.id_gen.next();
        usable_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionTypeKHR,
                u32_type_id,
                kind_id,
                query_id,
                committed_id,
            ));
        let none_id = self.get_constant_scalar(crate::Literal::U32(
            spirv::RayQueryCommittedIntersectionType::RayQueryCommittedIntersectionNoneKHR as _,
        ));
        let is_hit_id = self.id_gen.next();
        usable_block.body.push(Instruction::binary(
            spirv::Op::INotEqual,
            bool_type_id,
            is_hit_id,
            kind_id,
            none_id,
        ));

        let hit_label_id = self.id_gen.next();
        let mut hit_block = Block::new(hit_label_id);

        let miss_label_id = self.id_gen.next();
        let mut miss_block = Block::new(miss_label_id);

        let recorded_label_id = self.id_gen.next();
        let recorded_block = Block::new(recorded_label_id);

        usable_block.body.push(Instruction::selection_merge(
            recorded_label_id,
            spirv::SelectionControl::NONE,
        ));
        function.consume(
            usable_block,
            Instruction::branch_conditional(is_hit_id, hit_label_id, miss_label_id),
        );

        let zero_id = self.get_constant_scalar(crate::Literal::U32(0));

        // A hit. Like `traceRay`, the shader binding table offset and stride
        // are zero, so the hit group is just the instance's record offset,
        // which is only defined when there is a committed intersection.
        let hit_sbt_id = self.id_gen.next();
        hit_block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR,
            u32_type_id,
            hit_sbt_id,
            query_id,
            committed_id,
        ));
        // For procedural hits the attribute variable is the source of the
        // recorded attributes, so give it a defined value. For triangle hits
        // it is ignored and the barycentrics come from the query.
        let zero_barycentrics_id = self.get_constant_null(barycentrics_type_id);
        hit_block.body.push(Instruction::store(
            attribute_var_id,
            zero_barycentrics_id,
            None,
        ));
        // Revision 3 of the extension requires a `Hit Kind` operand for
        // procedural hits. We cannot emit one yet: the `rspirv` grammar we
        // disassemble with in tests predates that revision and rejects the
        // extra operand. The specification's issue 1 notes that, for
        // compatibility with such SPIR-V, implementations should accept the
        // instruction for AABB intersections without a hit kind.
        //
        // TODO: pass `Some(zero_id)` once `rspirv` ships a grammar that knows
        // about the operand.
        hit_block
            .body
            .push(Instruction::hit_object_record_from_query(
                hit_object_id,
                query_id,
                hit_sbt_id,
                attribute_var_id,
                None,
            ));
        function.consume(hit_block, Instruction::branch(recorded_label_id));

        // No committed intersection: record a miss along the query's ray.
        let ray_flags_id = self.id_gen.next();
        miss_block.body.push(Instruction::ray_query_get(
            spirv::Op::RayQueryGetRayFlagsKHR,
            u32_type_id,
            ray_flags_id,
            query_id,
        ));
        let ray_origin_id = self.id_gen.next();
        miss_block.body.push(Instruction::ray_query_get(
            spirv::Op::RayQueryGetWorldRayOriginKHR,
            vec3_type_id,
            ray_origin_id,
            query_id,
        ));
        let tmin_id = self.id_gen.next();
        miss_block.body.push(Instruction::ray_query_get_t_min(
            f32_type_id,
            tmin_id,
            query_id,
        ));
        let ray_dir_id = self.id_gen.next();
        miss_block.body.push(Instruction::ray_query_get(
            spirv::Op::RayQueryGetWorldRayDirectionKHR,
            vec3_type_id,
            ray_dir_id,
            query_id,
        ));
        // SPIR-V has no getter for the `t_max` a query was initialized with,
        // so use the value the initialization tracked for us.
        let tmax_id = self.id_gen.next();
        miss_block.body.push(Instruction::load(
            f32_type_id,
            tmax_id,
            t_max_tracker_id,
            None,
        ));
        miss_block.body.push(Instruction::hit_object_record_miss(
            hit_object_id,
            ray_flags_id,
            zero_id,
            ray_origin_id,
            tmin_id,
            ray_dir_id,
            tmax_id,
        ));
        function.consume(miss_block, Instruction::branch(recorded_label_id));

        function.consume(recorded_block, Instruction::branch(merge_label_id));

        // A query that has not finished traversal has no committed
        // intersection to read, so leave the hit object in a well-defined
        // state instead.
        unusable_block
            .body
            .push(Instruction::hit_object_record_empty(hit_object_id));
        function.consume(unusable_block, Instruction::branch(merge_label_id));

        function.consume(merge_block, Instruction::return_void());
        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_tracing_functions
            .insert(LookupRaytracingFunction::HitObjectRecordFromQuery, func_id);

        Ok(func_id)
    }

    /// Write a helper function that runs the closest hit or miss shader
    /// recorded in a hit object.
    ///
    /// Like [`Writer::write_trace_ray`], the payload is baked into the function.
    fn write_hit_object_execute_shader(
        &mut self,
        payload: crate::Handle<crate::GlobalVariable>,
    ) -> Result<spirv::Word, super::super::Error> {
        if let Some(&word) = self
            .ray_tracing_functions
            .get(&LookupRaytracingFunction::HitObjectExecuteShader { payload })
        {
            return Ok(word);
        }

        self.require_hit_objects()?;

        let hit_object_pointer_type_id = self.get_hit_object_pointer_id();

        let (func_id, mut function, arg_ids) =
            self.write_function_signature(&[hit_object_pointer_type_id], self.void_type);

        let hit_object_id = arg_ids[0];
        let payload_id = self.global_variables[payload].access_id;

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        block.body.push(Instruction::hit_object_execute_shader(
            hit_object_id,
            payload_id,
        ));

        function.consume(block, Instruction::return_void());
        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_tracing_functions.insert(
            LookupRaytracingFunction::HitObjectExecuteShader { payload },
            func_id,
        );

        Ok(func_id)
    }

    /// Write a helper function that builds a `RayIntersection` value out of a
    /// hit object.
    ///
    /// The member indices and types used here must match
    /// [`Module::generate_ray_intersection_type`].
    ///
    /// Note that the hit object getters may only be executed when the hit
    /// object actually records a hit, and `OpHitObjectGetAttributesEXT` only
    /// when that hit is a triangle. The generated function therefore keeps
    /// those getters inside guarded blocks and merges the results with `OpPhi`,
    /// rather than accumulating them into a local variable: a `Function`
    /// storage class variable of the `RayIntersection` struct type would carry
    /// that struct's explicit layout decorations, which Vulkan forbids
    /// (`VUID-StandaloneSpirv-None-10684`).
    ///
    /// [`Module::generate_ray_intersection_type`]: crate::Module::generate_ray_intersection_type
    pub(in super::super) fn write_hit_object_get_intersection_function(
        &mut self,
        ir_module: &crate::Module,
    ) -> Result<spirv::Word, super::super::Error> {
        if let Some(&word) = self
            .ray_tracing_functions
            .get(&LookupRaytracingFunction::HitObjectGetIntersection)
        {
            return Ok(word);
        }

        self.require_hit_objects()?;

        let ray_intersection = ir_module
            .special_types
            .ray_intersection
            .expect("ray intersection should be set if `hitObjectGetIntersection` is called");
        let intersection_type_id = self.get_handle_type_id(ray_intersection);

        let flag_type_id = self.get_u32_type_id();

        let transform_type_id = self.get_numeric_type_id(NumericType::Matrix {
            columns: crate::VectorSize::Quad,
            rows: crate::VectorSize::Tri,
            scalar: crate::Scalar::F32,
        });

        let barycentrics_type_id = self.get_numeric_type_id(NumericType::Vector {
            size: crate::VectorSize::Bi,
            scalar: crate::Scalar::F32,
        });

        let bool_type_id = self.get_bool_type_id();

        let scalar_type_id = self.get_f32_type_id();

        let attribute_var_id = self.get_hit_object_attribute_var();

        let argument_type_id = self.get_hit_object_pointer_id();

        let (func_id, mut function, arg_ids) =
            self.write_function_signature(&[argument_type_id], intersection_type_id);

        let hit_object_id = arg_ids[0];

        let entry_label_id = self.id_gen.next();
        let mut block = Block::new(entry_label_id);

        let is_hit_id = self.id_gen.next();
        block.body.push(Instruction::hit_object_get(
            spirv::Op::HitObjectIsHitEXT,
            bool_type_id,
            is_hit_id,
            hit_object_id,
        ));

        let hit_label_id = self.id_gen.next();
        let mut hit_block = Block::new(hit_label_id);

        let final_label_id = self.id_gen.next();
        let mut final_block = Block::new(final_label_id);

        block.body.push(Instruction::selection_merge(
            final_label_id,
            spirv::SelectionControl::NONE,
        ));
        function.consume(
            block,
            Instruction::branch_conditional(is_hit_id, hit_label_id, final_label_id),
        );

        // `kind`: the hit kind tells us whether this was a triangle or a
        // generated (procedural) intersection.
        let hit_kind_id = self.id_gen.next();
        hit_block.body.push(Instruction::hit_object_get(
            spirv::Op::HitObjectGetHitKindEXT,
            flag_type_id,
            hit_kind_id,
            hit_object_id,
        ));

        let front_facing_triangle_id =
            self.get_constant_scalar(crate::Literal::U32(HIT_KIND_FRONT_FACING_TRIANGLE));

        let is_triangle_id = self.id_gen.next();
        hit_block.body.push(Instruction::binary(
            spirv::Op::UGreaterThanEqual,
            bool_type_id,
            is_triangle_id,
            hit_kind_id,
            front_facing_triangle_id,
        ));

        let triangle_kind_id = self.get_constant_scalar(crate::Literal::U32(
            crate::RayQueryIntersection::Triangle as _,
        ));
        let generated_kind_id = self.get_constant_scalar(crate::Literal::U32(
            crate::RayQueryIntersection::Generated as _,
        ));
        let kind_id = self.id_gen.next();
        hit_block.body.push(Instruction::select(
            flag_type_id,
            kind_id,
            is_triangle_id,
            triangle_kind_id,
            generated_kind_id,
        ));

        let front_face_id = self.id_gen.next();
        hit_block.body.push(Instruction::binary(
            spirv::Op::IEqual,
            bool_type_id,
            front_face_id,
            hit_kind_id,
            front_facing_triangle_id,
        ));

        let t_id = self.id_gen.next();
        hit_block.body.push(Instruction::hit_object_get(
            spirv::Op::HitObjectGetRayTMaxEXT,
            scalar_type_id,
            t_id,
            hit_object_id,
        ));

        // The plain `u32` members.
        let mut simple_getter = |writer: &mut Self, op| {
            let value_id = writer.id_gen.next();
            hit_block.body.push(Instruction::hit_object_get(
                op,
                flag_type_id,
                value_id,
                hit_object_id,
            ));
            value_id
        };
        let instance_custom_data_id =
            simple_getter(self, spirv::Op::HitObjectGetInstanceCustomIndexEXT);
        let instance_index_id = simple_getter(self, spirv::Op::HitObjectGetInstanceIdEXT);
        let sbt_record_offset_id = simple_getter(
            self,
            spirv::Op::HitObjectGetShaderBindingTableRecordIndexEXT,
        );
        let geometry_index_id = simple_getter(self, spirv::Op::HitObjectGetGeometryIndexEXT);
        let primitive_index_id = simple_getter(self, spirv::Op::HitObjectGetPrimitiveIndexEXT);

        let object_to_world_id = self.id_gen.next();
        hit_block.body.push(Instruction::hit_object_get(
            spirv::Op::HitObjectGetObjectToWorldEXT,
            transform_type_id,
            object_to_world_id,
            hit_object_id,
        ));
        let world_to_object_id = self.id_gen.next();
        hit_block.body.push(Instruction::hit_object_get(
            spirv::Op::HitObjectGetWorldToObjectEXT,
            transform_type_id,
            world_to_object_id,
            hit_object_id,
        ));

        // Barycentrics are only meaningful for triangle hits; for anything else
        // they stay at zero.
        let triangle_label_id = self.id_gen.next();
        let mut triangle_block = Block::new(triangle_label_id);

        let triangle_merge_label_id = self.id_gen.next();
        let mut triangle_merge_block = Block::new(triangle_merge_label_id);

        hit_block.body.push(Instruction::selection_merge(
            triangle_merge_label_id,
            spirv::SelectionControl::NONE,
        ));
        function.consume(
            hit_block,
            Instruction::branch_conditional(
                is_triangle_id,
                triangle_label_id,
                triangle_merge_label_id,
            ),
        );

        triangle_block
            .body
            .push(Instruction::hit_object_get_attributes(
                hit_object_id,
                attribute_var_id,
            ));
        let triangle_barycentrics_id = self.id_gen.next();
        triangle_block.body.push(Instruction::load(
            barycentrics_type_id,
            triangle_barycentrics_id,
            attribute_var_id,
            None,
        ));

        function.consume(triangle_block, Instruction::branch(triangle_merge_label_id));

        // `OpPhi` must come first in the block, before the composite that uses it.
        let zero_barycentrics_id = self.get_constant_null(barycentrics_type_id);
        let barycentrics_id = self.id_gen.next();
        triangle_merge_block.body.push(Instruction::phi(
            barycentrics_type_id,
            barycentrics_id,
            &[
                (triangle_barycentrics_id, triangle_label_id),
                (zero_barycentrics_id, hit_label_id),
            ],
        ));

        let intersection_id = self.id_gen.next();
        triangle_merge_block
            .body
            .push(Instruction::composite_construct(
                intersection_type_id,
                intersection_id,
                &[
                    kind_id,
                    t_id,
                    instance_custom_data_id,
                    instance_index_id,
                    sbt_record_offset_id,
                    geometry_index_id,
                    primitive_index_id,
                    barycentrics_id,
                    front_face_id,
                    object_to_world_id,
                    world_to_object_id,
                ],
            ));

        function.consume(triangle_merge_block, Instruction::branch(final_label_id));

        let blank_intersection_id = self.get_constant_null(intersection_type_id);
        let result_id = self.id_gen.next();
        final_block.body.push(Instruction::phi(
            intersection_type_id,
            result_id,
            &[
                (blank_intersection_id, entry_label_id),
                (intersection_id, triangle_merge_label_id),
            ],
        ));
        function.consume(final_block, Instruction::return_value(result_id));

        function.to_words(&mut self.logical_layout.function_definitions);
        self.ray_tracing_functions
            .insert(LookupRaytracingFunction::HitObjectGetIntersection, func_id);

        Ok(func_id)
    }
}

impl BlockContext<'_> {
    pub(in super::super) fn write_hit_object_function(
        &mut self,
        hit_object: crate::Handle<crate::Expression>,
        fun: &crate::HitObjectFunction,
        block: &mut Block,
    ) -> Result<(), super::super::Error> {
        let hit_object_id = self.cached[hit_object];

        match *fun {
            crate::HitObjectFunction::TraceRay {
                acceleration_structure,
                descriptor,
                payload,
            } => {
                // Checked for when validating the module in `validate_block_impl`.
                let crate::Expression::GlobalVariable(payload) =
                    self.ir_function.expressions[payload]
                else {
                    unreachable!()
                };

                let desc_id = self.cached[descriptor];
                let acc_struct_id = self.get_handle_id(acceleration_structure);

                let func = self
                    .writer
                    .write_hit_object_trace_ray(self.ir_module, payload)?;

                let func_id = self.gen_id();
                block.body.push(Instruction::function_call(
                    self.writer.void_type,
                    func_id,
                    func,
                    &[hit_object_id, acc_struct_id, desc_id],
                ));
            }
            crate::HitObjectFunction::RecordMiss { descriptor } => {
                let desc_id = self.cached[descriptor];

                let func = self.writer.write_hit_object_record_miss(self.ir_module)?;

                let func_id = self.gen_id();
                block.body.push(Instruction::function_call(
                    self.writer.void_type,
                    func_id,
                    func,
                    &[hit_object_id, desc_id],
                ));
            }
            crate::HitObjectFunction::RecordFromQuery { query } => {
                let query_id = self.cached[query];
                let trackers = *self
                    .ray_query_tracker_expr
                    .get(&query)
                    .expect("not a cached ray query");

                let func = self.writer.write_hit_object_record_from_query()?;

                let func_id = self.gen_id();
                block.body.push(Instruction::function_call(
                    self.writer.void_type,
                    func_id,
                    func,
                    &[
                        hit_object_id,
                        query_id,
                        trackers.initialized_tracker,
                        trackers.t_max_tracker,
                    ],
                ));
            }
            crate::HitObjectFunction::RecordEmpty => {
                self.writer.require_hit_objects()?;
                block
                    .body
                    .push(Instruction::hit_object_record_empty(hit_object_id));
            }
            crate::HitObjectFunction::ExecuteShader { payload } => {
                // Checked for when validating the module in `validate_block_impl`.
                let crate::Expression::GlobalVariable(payload) =
                    self.ir_function.expressions[payload]
                else {
                    unreachable!()
                };

                let func = self.writer.write_hit_object_execute_shader(payload)?;

                let func_id = self.gen_id();
                block.body.push(Instruction::function_call(
                    self.writer.void_type,
                    func_id,
                    func,
                    &[hit_object_id],
                ));
            }
            crate::HitObjectFunction::Reorder { hint } => {
                self.writer.require_hit_objects()?;
                let hint = hint.map(|crate::ReorderHint { hint, bits }| {
                    (self.cached[hint], self.cached[bits])
                });
                block.body.push(Instruction::reorder_thread_with_hit_object(
                    hit_object_id,
                    hint,
                ));
            }
        }

        Ok(())
    }

    /// Write an [`Expression::HitObjectQuery`].
    ///
    /// [`Expression::HitObjectQuery`]: crate::Expression::HitObjectQuery
    pub(in super::super) fn write_hit_object_query(
        &mut self,
        hit_object: crate::Handle<crate::Expression>,
        query: crate::HitObjectQuery,
        block: &mut Block,
    ) -> Result<spirv::Word, super::super::Error> {
        self.writer.require_hit_objects()?;
        let hit_object_id = self.cached[hit_object];

        let op = match query {
            crate::HitObjectQuery::IsEmpty => spirv::Op::HitObjectIsEmptyEXT,
            crate::HitObjectQuery::IsHit => spirv::Op::HitObjectIsHitEXT,
            crate::HitObjectQuery::IsMiss => spirv::Op::HitObjectIsMissEXT,
            crate::HitObjectQuery::Intersection => {
                let func = self
                    .writer
                    .write_hit_object_get_intersection_function(self.ir_module)?;
                let ray_intersection = self.ir_module.special_types.ray_intersection.unwrap();
                let intersection_type_id = self.get_handle_type_id(ray_intersection);
                let id = self.gen_id();
                block.body.push(Instruction::function_call(
                    intersection_type_id,
                    id,
                    func,
                    &[hit_object_id],
                ));
                return Ok(id);
            }
        };

        let bool_type_id = self.writer.get_bool_type_id();
        let id = self.gen_id();
        block.body.push(Instruction::hit_object_get(
            op,
            bool_type_id,
            id,
            hit_object_id,
        ));
        Ok(id)
    }
}
