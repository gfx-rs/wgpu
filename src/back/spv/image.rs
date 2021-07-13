//! Generating SPIR-V for image operations.

use super::{Block, BlockContext, Error, Instruction, LocalType, LookupType};
use crate::arena::Handle;
use spirv::Word;

impl<'w> BlockContext<'w> {
    /// Extend image coordinates with an array index, if necessary.
    ///
    /// Whereas [`Expression::ImageLoad`] and [`ImageSample`] treat the array
    /// index as a separate operand from the coordinates, SPIR-V image access
    /// instructions include the array index in the `coordinates` operand. This
    /// function builds a SPIR-V coordinate vector from a Naga coordinate vector
    /// and array index.
    ///
    /// If `array_index` is `Some(expr)`, then this function constructs a new
    /// vector that is `coordinates` with `array_index` concatenated onto the
    /// end: a `vec2` becomes a `vec3`, a scalar becomes a `vec2`, and so on.
    ///
    /// Naga's `ImageLoad` and SPIR-V's `OpImageRead`, `OpImageFetch`, and
    /// `OpImageWrite` all use integer coordinates, while Naga's `ImageSample`
    /// and SPIR-V's `OpImageSample...` instructions all take floating-point
    /// coordinate vectors. The array index, always an integer scalar, may need
    /// to be converted to match the component type of `coordinates`.
    ///
    /// If `array_index` is `None`, this function simply returns the id for
    /// `coordinates`.
    ///
    /// [`Expression::ImageLoad`]: crate::Expression::ImageLoad
    /// [`ImageSample`]: crate::Expression::ImageSample
    fn write_image_coordinates(
        &mut self,
        coordinates: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        use crate::TypeInner as Ti;
        use crate::VectorSize as Vs;

        let coordinate_id = self.cached[coordinates];

        // If there's no array index, the image coordinates are exactly the
        // `coordinate` field of the `Expression::ImageLoad`. No work is needed.
        let array_index = match array_index {
            None => return Ok(coordinate_id),
            Some(ix) => ix,
        };

        // Find the component type of `coordinates`, and figure out the size the
        // combined coordinate vector will have.
        let (component_kind, result_size) = match *self.fun_info[coordinates]
            .ty
            .inner_with(&self.ir_module.types)
        {
            Ti::Scalar { kind, width: 4 } => (kind, Vs::Bi),
            Ti::Vector {
                kind,
                width: 4,
                size: Vs::Bi,
            } => (kind, Vs::Tri),
            Ti::Vector {
                kind,
                width: 4,
                size: Vs::Tri,
            } => (kind, Vs::Quad),
            Ti::Vector { size: Vs::Quad, .. } => {
                return Err(Error::Validation("extending vec4 coordinate"));
            }
            ref other => {
                log::error!("wrong coordinate type {:?}", other);
                return Err(Error::Validation("coordinate type"));
            }
        };

        // Convert the index to the coordinate component type, if necessary.
        let array_index_i32_id = self.cached[array_index];
        let reconciled_array_index_id = if component_kind == crate::ScalarKind::Sint {
            array_index_i32_id
        } else {
            let component_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
                vector_size: None,
                kind: component_kind,
                width: 4,
                pointer_class: None,
            }));

            let reconciled_id = self.gen_id();
            block.body.push(Instruction::unary(
                spirv::Op::ConvertUToF,
                component_type_id,
                reconciled_id,
                array_index_i32_id,
            ));
            reconciled_id
        };

        // Find the SPIR-V type for the combined coordinates/index vector.
        let combined_coordinate_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: Some(result_size),
            kind: component_kind,
            width: 4,
            pointer_class: None,
        }));

        // Schmear the coordinates and index together.
        let id = self.gen_id();
        block.body.push(Instruction::composite_construct(
            combined_coordinate_type_id,
            id,
            &[coordinate_id, reconciled_array_index_id],
        ));
        Ok(id)
    }

    fn get_image_id(&mut self, expr_handle: Handle<crate::Expression>) -> Word {
        let id = match self.ir_function.expressions[expr_handle] {
            crate::Expression::GlobalVariable(handle) => {
                self.writer.global_variables[handle.index()].handle_id
            }
            crate::Expression::FunctionArgument(i) => {
                self.function.parameters[i as usize].handle_id
            }
            ref other => unreachable!("Unexpected image expression {:?}", other),
        };

        if id == 0 {
            unreachable!(
                "Image expression {:?} doesn't have a handle ID",
                expr_handle
            );
        }

        id
    }

    /// Generate code for an `ImageLoad` expression.
    ///
    /// The arguments are the components of an `Expression::ImageLoad` variant.
    pub(super) fn write_image_load(
        &mut self,
        result_type_id: Word,
        image: Handle<crate::Expression>,
        coordinate: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        index: Option<Handle<crate::Expression>>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        let image_id = self.get_image_id(image);
        let coordinate_id = self.write_image_coordinates(coordinate, array_index, block)?;

        let id = self.gen_id();

        let image_ty = self.fun_info[image].ty.inner_with(&self.ir_module.types);
        let mut instruction = match *image_ty {
            crate::TypeInner::Image {
                class: crate::ImageClass::Storage { .. },
                ..
            } => Instruction::image_read(result_type_id, id, image_id, coordinate_id),
            crate::TypeInner::Image {
                class: crate::ImageClass::Depth { multi: _ },
                ..
            } => {
                // Vulkan doesn't know about our `Depth` class, and it returns `vec4<f32>`,
                // so we need to grab the first component out of it.
                let load_result_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
                    vector_size: Some(crate::VectorSize::Quad),
                    kind: crate::ScalarKind::Float,
                    width: 4,
                    pointer_class: None,
                }));
                Instruction::image_fetch(load_result_type_id, id, image_id, coordinate_id)
            }
            _ => Instruction::image_fetch(result_type_id, id, image_id, coordinate_id),
        };

        if let Some(index) = index {
            let index_id = self.cached[index];
            let image_ops = match *self.fun_info[image].ty.inner_with(&self.ir_module.types) {
                crate::TypeInner::Image {
                    class: crate::ImageClass::Sampled { multi: true, .. },
                    ..
                }
                | crate::TypeInner::Image {
                    class: crate::ImageClass::Depth { multi: true },
                    ..
                } => spirv::ImageOperands::SAMPLE,
                _ => spirv::ImageOperands::LOD,
            };
            instruction.add_operand(image_ops.bits());
            instruction.add_operand(index_id);
        }

        let inst_type_id = instruction.type_id;
        block.body.push(instruction);
        let id = if inst_type_id != Some(result_type_id) {
            let sub_id = self.gen_id();
            block.body.push(Instruction::composite_extract(
                result_type_id,
                sub_id,
                id,
                &[0],
            ));
            sub_id
        } else {
            id
        };

        Ok(id)
    }

    /// Generate code for an `ImageSample` expression.
    ///
    /// The arguments are the components of an `Expression::ImageSample` variant.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_image_sample(
        &mut self,
        result_type_id: Word,
        image: Handle<crate::Expression>,
        sampler: Handle<crate::Expression>,
        coordinate: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        offset: Option<Handle<crate::Constant>>,
        level: crate::SampleLevel,
        depth_ref: Option<Handle<crate::Expression>>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        use super::instructions::SampleLod;
        // image
        let image_id = self.get_image_id(image);
        let image_type = self.fun_info[image].ty.handle().unwrap();
        // Vulkan doesn't know about our `Depth` class, and it returns `vec4<f32>`,
        // so we need to grab the first component out of it.
        let needs_sub_access = match self.ir_module.types[image_type].inner {
            crate::TypeInner::Image {
                class: crate::ImageClass::Depth { .. },
                ..
            } => depth_ref.is_none(),
            _ => false,
        };
        let sample_result_type_id = if needs_sub_access {
            self.get_type_id(LookupType::Local(LocalType::Value {
                vector_size: Some(crate::VectorSize::Quad),
                kind: crate::ScalarKind::Float,
                width: 4,
                pointer_class: None,
            }))
        } else {
            result_type_id
        };

        // OpTypeSampledImage
        let image_type_id = self.get_type_id(LookupType::Handle(image_type));
        let sampled_image_type_id =
            self.get_type_id(LookupType::Local(LocalType::SampledImage { image_type_id }));

        let sampler_id = self.get_image_id(sampler);
        let coordinate_id = self.write_image_coordinates(coordinate, array_index, block)?;

        let sampled_image_id = self.gen_id();
        block.body.push(Instruction::sampled_image(
            sampled_image_type_id,
            sampled_image_id,
            image_id,
            sampler_id,
        ));
        let id = self.gen_id();

        let depth_id = depth_ref.map(|handle| self.cached[handle]);
        let mut mask = spirv::ImageOperands::empty();
        mask.set(spirv::ImageOperands::CONST_OFFSET, offset.is_some());

        let mut main_instruction = match level {
            crate::SampleLevel::Zero => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Explicit,
                    sampled_image_id,
                    coordinate_id,
                    depth_id,
                );

                let zero_id = self
                    .writer
                    .get_constant_scalar(crate::ScalarValue::Float(0.0), 4);

                mask |= spirv::ImageOperands::LOD;
                inst.add_operand(mask.bits());
                inst.add_operand(zero_id);

                inst
            }
            crate::SampleLevel::Auto => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Implicit,
                    sampled_image_id,
                    coordinate_id,
                    depth_id,
                );
                if !mask.is_empty() {
                    inst.add_operand(mask.bits());
                }
                inst
            }
            crate::SampleLevel::Exact(lod_handle) => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Explicit,
                    sampled_image_id,
                    coordinate_id,
                    depth_id,
                );

                let lod_id = self.cached[lod_handle];
                mask |= spirv::ImageOperands::LOD;
                inst.add_operand(mask.bits());
                inst.add_operand(lod_id);

                inst
            }
            crate::SampleLevel::Bias(bias_handle) => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Implicit,
                    sampled_image_id,
                    coordinate_id,
                    depth_id,
                );

                let bias_id = self.cached[bias_handle];
                mask |= spirv::ImageOperands::BIAS;
                inst.add_operand(mask.bits());
                inst.add_operand(bias_id);

                inst
            }
            crate::SampleLevel::Gradient { x, y } => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Explicit,
                    sampled_image_id,
                    coordinate_id,
                    depth_id,
                );

                let x_id = self.cached[x];
                let y_id = self.cached[y];
                mask |= spirv::ImageOperands::GRAD;
                inst.add_operand(mask.bits());
                inst.add_operand(x_id);
                inst.add_operand(y_id);

                inst
            }
        };

        if let Some(offset_const) = offset {
            let offset_id = self.writer.constant_ids[offset_const.index()];
            main_instruction.add_operand(offset_id);
        }

        block.body.push(main_instruction);

        let id = if needs_sub_access {
            let sub_id = self.gen_id();
            block.body.push(Instruction::composite_extract(
                result_type_id,
                sub_id,
                id,
                &[0],
            ));
            sub_id
        } else {
            id
        };

        Ok(id)
    }

    /// Generate code for an `ImageQuery` expression.
    ///
    /// The arguments are the components of an `Expression::ImageQuery` variant.
    pub(super) fn write_image_query(
        &mut self,
        result_type_id: Word,
        image: Handle<crate::Expression>,
        query: crate::ImageQuery,
        block: &mut Block,
    ) -> Result<Word, Error> {
        use crate::{ImageClass as Ic, ImageDimension as Id, ImageQuery as Iq};

        let image_id = self.get_image_id(image);
        let image_type = self.fun_info[image].ty.handle().unwrap();
        let (dim, arrayed, class) = match self.ir_module.types[image_type].inner {
            crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => (dim, arrayed, class),
            _ => {
                return Err(Error::Validation("image type"));
            }
        };

        self.writer
            .require_any("image queries", &[spirv::Capability::ImageQuery])?;

        let id = match query {
            Iq::Size { level } => {
                let dim_coords = match dim {
                    Id::D1 => 1,
                    Id::D2 | Id::Cube => 2,
                    Id::D3 => 3,
                };
                let extended_size_type_id = {
                    let array_coords = if arrayed { 1 } else { 0 };
                    let vector_size = match dim_coords + array_coords {
                        2 => Some(crate::VectorSize::Bi),
                        3 => Some(crate::VectorSize::Tri),
                        4 => Some(crate::VectorSize::Quad),
                        _ => None,
                    };
                    self.get_type_id(LookupType::Local(LocalType::Value {
                        vector_size,
                        kind: crate::ScalarKind::Sint,
                        width: 4,
                        pointer_class: None,
                    }))
                };

                let (query_op, level_id) = match class {
                    Ic::Storage { .. } => (spirv::Op::ImageQuerySize, None),
                    _ => {
                        let level_id = match level {
                            Some(expr) => self.cached[expr],
                            None => self.get_index_constant(0),
                        };
                        (spirv::Op::ImageQuerySizeLod, Some(level_id))
                    }
                };

                // The ID of the vector returned by SPIR-V, which contains the dimensions
                // as well as the layer count.
                let id_extended = self.gen_id();
                let mut inst = Instruction::image_query(
                    query_op,
                    extended_size_type_id,
                    id_extended,
                    image_id,
                );
                if let Some(expr_id) = level_id {
                    inst.add_operand(expr_id);
                }
                block.body.push(inst);

                if result_type_id != extended_size_type_id {
                    let id = self.gen_id();
                    let components = match dim {
                        // always pick the first component, and duplicate it for all 3 dimensions
                        Id::Cube => &[0u32, 0][..],
                        _ => &[0u32, 1, 2, 3][..dim_coords],
                    };
                    block.body.push(Instruction::vector_shuffle(
                        result_type_id,
                        id,
                        id_extended,
                        id_extended,
                        components,
                    ));
                    id
                } else {
                    id_extended
                }
            }
            Iq::NumLevels => {
                let id = self.gen_id();
                block.body.push(Instruction::image_query(
                    spirv::Op::ImageQueryLevels,
                    result_type_id,
                    id,
                    image_id,
                ));
                id
            }
            Iq::NumLayers => {
                let vec_size = match dim {
                    Id::D1 => crate::VectorSize::Bi,
                    Id::D2 | Id::Cube => crate::VectorSize::Tri,
                    Id::D3 => crate::VectorSize::Quad,
                };
                let extended_size_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
                    vector_size: Some(vec_size),
                    kind: crate::ScalarKind::Sint,
                    width: 4,
                    pointer_class: None,
                }));
                let id_extended = self.gen_id();
                let mut inst = Instruction::image_query(
                    spirv::Op::ImageQuerySizeLod,
                    extended_size_type_id,
                    id_extended,
                    image_id,
                );
                inst.add_operand(self.get_index_constant(0));
                block.body.push(inst);
                let id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    result_type_id,
                    id,
                    id_extended,
                    &[vec_size as u32 - 1],
                ));
                id
            }
            Iq::NumSamples => {
                let id = self.gen_id();
                block.body.push(Instruction::image_query(
                    spirv::Op::ImageQuerySamples,
                    result_type_id,
                    id,
                    image_id,
                ));
                id
            }
        };

        Ok(id)
    }

    pub(super) fn write_image_store(
        &mut self,
        image: Handle<crate::Expression>,
        coordinate: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        value: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        let image_id = self.get_image_id(image);
        let coordinate_id = self.write_image_coordinates(coordinate, array_index, block)?;
        let value_id = self.cached[value];

        block
            .body
            .push(Instruction::image_write(image_id, coordinate_id, value_id));

        Ok(())
    }
}
