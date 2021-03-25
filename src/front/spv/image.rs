use crate::arena::{Arena, Handle};

use super::{Error, LookupExpression, LookupHelper as _};

#[derive(Clone, Debug)]
pub(super) struct LookupSampledImage {
    image: Handle<crate::Expression>,
    sampler: Handle<crate::Expression>,
}

bitflags::bitflags! {
    /// Flags describing sampling method.
    pub struct SamplingFlags: u32 {
        /// Regular sampling.
        const REGULAR = 0x1;
        /// Comparison sampling.
        const COMPARISON = 0x2;
    }
}

impl Arena<crate::Expression> {
    fn get_global_var(
        &self,
        handle: Handle<crate::Expression>,
    ) -> Result<Handle<crate::GlobalVariable>, Error> {
        match self[handle] {
            crate::Expression::GlobalVariable(handle) => Ok(handle),
            ref other => Err(Error::InvalidGlobalVar(other.clone())),
        }
    }
}

/// Return the texture coordinates separated from the array layer,
/// and/or divided by the projection term.
///
/// The Proj sampling ops expect an extra coordinate for the W.
/// The arrayed (can't be Proj!) images expect an extra coordinate for the layer.
fn extract_image_coordinates(
    image_dim: crate::ImageDimension,
    image_arrayed: bool,
    base: Handle<crate::Expression>,
    coordinate_ty: Handle<crate::Type>,
    type_arena: &Arena<crate::Type>,
    expressions: &mut Arena<crate::Expression>,
) -> (Handle<crate::Expression>, Option<Handle<crate::Expression>>) {
    let (given_size, kind) = match type_arena[coordinate_ty].inner {
        crate::TypeInner::Scalar { kind, .. } => (None, kind),
        crate::TypeInner::Vector { size, kind, .. } => (Some(size), kind),
        ref other => unreachable!("Unexpected texture coordinate {:?}", other),
    };

    let required_size = image_dim.required_coordinate_size();
    let required_ty = required_size.map(|size| {
        type_arena
            .fetch_if(|ty| {
                ty.inner
                    == crate::TypeInner::Vector {
                        size,
                        kind,
                        width: 4,
                    }
            })
            .expect("Required coordinate type should have been set up by `parse_type_image`!")
    });
    let extra_expr = crate::Expression::AccessIndex {
        base,
        index: required_size.map_or(1, |size| size as u32),
    };

    if given_size == required_size {
        // fast path, no complications
        (base, None)
    } else if image_arrayed {
        // extra coordinate for the array index
        let extracted = match required_size {
            None => expressions.append(crate::Expression::AccessIndex { base, index: 0 }),
            Some(size) => {
                let mut components = Vec::with_capacity(size as usize);
                for index in 0..size as u32 {
                    let comp = expressions.append(crate::Expression::AccessIndex { base, index });
                    components.push(comp);
                }
                expressions.append(crate::Expression::Compose {
                    ty: required_ty.unwrap(),
                    components,
                })
            }
        };
        let array_index_f32 = expressions.append(extra_expr);
        let array_index = expressions.append(crate::Expression::As {
            kind: crate::ScalarKind::Sint,
            expr: array_index_f32,
            convert: true,
        });
        (extracted, Some(array_index))
    } else {
        // extra coordinate for the projection W
        let projection = expressions.append(extra_expr);
        let divided = match required_size {
            None => {
                let temp = expressions.append(crate::Expression::AccessIndex { base, index: 0 });
                expressions.append(crate::Expression::Binary {
                    op: crate::BinaryOperator::Divide,
                    left: temp,
                    right: projection,
                })
            }
            Some(size) => {
                let mut components = Vec::with_capacity(size as usize);
                for index in 0..size as u32 {
                    let temp = expressions.append(crate::Expression::AccessIndex { base, index });
                    let comp = expressions.append(crate::Expression::Binary {
                        op: crate::BinaryOperator::Divide,
                        left: temp,
                        right: projection,
                    });
                    components.push(comp);
                }
                expressions.append(crate::Expression::Compose {
                    ty: required_ty.unwrap(),
                    components,
                })
            }
        };
        (divided, None)
    }
}

pub(super) fn patch_comparison_type(
    flags: SamplingFlags,
    var: &mut crate::GlobalVariable,
    arena: &mut Arena<crate::Type>,
) -> bool {
    if !flags.contains(SamplingFlags::COMPARISON) {
        return true;
    }
    if flags == SamplingFlags::all() {
        return false;
    }

    log::debug!("Flipping comparison for {:?}", var);
    let original_ty = &arena[var.ty];
    let ty_inner = match original_ty.inner {
        crate::TypeInner::Image {
            class: _,
            dim,
            arrayed,
        } => crate::TypeInner::Image {
            class: crate::ImageClass::Depth,
            dim,
            arrayed,
        },
        crate::TypeInner::Sampler { .. } => crate::TypeInner::Sampler { comparison: true },
        ref other => unreachable!("Unexpected type for comparison mutation: {:?}", other),
    };

    let name = original_ty.name.clone();
    var.ty = arena.append(crate::Type {
        name,
        inner: ty_inner,
    });
    true
}

impl<I: Iterator<Item = u32>> super::Parser<I> {
    pub(super) fn parse_image_couple(&mut self) -> Result<(), Error> {
        let _result_type_id = self.next()?;
        let result_id = self.next()?;
        let image_id = self.next()?;
        let sampler_id = self.next()?;
        let image_lexp = self.lookup_expression.lookup(image_id)?;
        let sampler_lexp = self.lookup_expression.lookup(sampler_id)?;
        self.lookup_sampled_image.insert(
            result_id,
            LookupSampledImage {
                image: image_lexp.handle,
                sampler: sampler_lexp.handle,
            },
        );
        Ok(())
    }

    pub(super) fn parse_image_uncouple(&mut self) -> Result<(), Error> {
        let result_type_id = self.next()?;
        let result_id = self.next()?;
        let sampled_image_id = self.next()?;
        self.lookup_expression.insert(
            result_id,
            LookupExpression {
                handle: self.lookup_sampled_image.lookup(sampled_image_id)?.image,
                type_id: result_type_id,
            },
        );
        Ok(())
    }

    pub(super) fn parse_image_write(
        &mut self,
        words_left: u16,
        type_arena: &Arena<crate::Type>,
        global_arena: &Arena<crate::GlobalVariable>,
        expressions: &mut Arena<crate::Expression>,
    ) -> Result<crate::Statement, Error> {
        let image_id = self.next()?;
        let coordinate_id = self.next()?;
        let value_id = self.next()?;

        let image_ops = if words_left != 0 { self.next()? } else { 0 };

        if image_ops != 0 {
            let other = spirv::ImageOperands::from_bits_truncate(image_ops);
            log::warn!("Unknown image write ops {:?}", other);
            for _ in 1..words_left {
                self.next()?;
            }
        }

        let image_lexp = self.lookup_expression.lookup(image_id)?;
        let image_var_handle = expressions.get_global_var(image_lexp.handle)?;
        let image_var = &global_arena[image_var_handle];

        let coord_lexp = self.lookup_expression.lookup(coordinate_id)?;
        let coord_type_handle = self.lookup_type.lookup(coord_lexp.type_id)?.handle;
        let (coordinate, array_index) = match type_arena[image_var.ty].inner {
            crate::TypeInner::Image {
                dim,
                arrayed,
                class: _,
            } => extract_image_coordinates(
                dim,
                arrayed,
                coord_lexp.handle,
                coord_type_handle,
                type_arena,
                expressions,
            ),
            _ => return Err(Error::InvalidImage(image_var.ty)),
        };

        let value_lexp = self.lookup_expression.lookup(value_id)?;

        Ok(crate::Statement::ImageStore {
            image: image_lexp.handle,
            coordinate,
            array_index,
            value: value_lexp.handle,
        })
    }

    pub(super) fn parse_image_load(
        &mut self,
        mut words_left: u16,
        type_arena: &Arena<crate::Type>,
        global_arena: &Arena<crate::GlobalVariable>,
        expressions: &mut Arena<crate::Expression>,
    ) -> Result<(), Error> {
        let result_type_id = self.next()?;
        let result_id = self.next()?;
        let image_id = self.next()?;
        let coordinate_id = self.next()?;

        let mut image_ops = if words_left != 0 {
            words_left -= 1;
            self.next()?
        } else {
            0
        };

        let mut index = None;
        while image_ops != 0 {
            let bit = 1 << image_ops.trailing_zeros();
            match spirv::ImageOperands::from_bits_truncate(bit) {
                spirv::ImageOperands::LOD => {
                    let lod_expr = self.next()?;
                    let lod_handle = self.lookup_expression.lookup(lod_expr)?.handle;
                    index = Some(lod_handle);
                    words_left -= 1;
                }
                spirv::ImageOperands::SAMPLE => {
                    let sample_expr = self.next()?;
                    let sample_handle = self.lookup_expression.lookup(sample_expr)?.handle;
                    index = Some(sample_handle);
                    words_left -= 1;
                }
                other => {
                    log::warn!("Unknown image load op {:?}", other);
                    for _ in 0..words_left {
                        self.next()?;
                    }
                    break;
                }
            }
            image_ops ^= bit;
        }

        let image_lexp = self.lookup_expression.lookup(image_id)?;
        let image_var_handle = expressions.get_global_var(image_lexp.handle)?;
        let image_var = &global_arena[image_var_handle];

        let coord_lexp = self.lookup_expression.lookup(coordinate_id)?;
        let coord_type_handle = self.lookup_type.lookup(coord_lexp.type_id)?.handle;
        let (coordinate, array_index) = match type_arena[image_var.ty].inner {
            crate::TypeInner::Image {
                dim,
                arrayed,
                class: _,
            } => extract_image_coordinates(
                dim,
                arrayed,
                coord_lexp.handle,
                coord_type_handle,
                type_arena,
                expressions,
            ),
            _ => return Err(Error::InvalidImage(image_var.ty)),
        };

        let expr = crate::Expression::ImageLoad {
            image: image_lexp.handle,
            coordinate,
            array_index,
            index,
        };
        self.lookup_expression.insert(
            result_id,
            LookupExpression {
                handle: expressions.append(expr),
                type_id: result_type_id,
            },
        );
        Ok(())
    }

    pub(super) fn parse_image_sample(
        &mut self,
        mut words_left: u16,
        type_arena: &Arena<crate::Type>,
        global_arena: &Arena<crate::GlobalVariable>,
        expressions: &mut Arena<crate::Expression>,
    ) -> Result<(), Error> {
        let result_type_id = self.next()?;
        let result_id = self.next()?;
        let sampled_image_id = self.next()?;
        let coordinate_id = self.next()?;

        let mut image_ops = if words_left != 0 {
            words_left -= 1;
            self.next()?
        } else {
            0
        };

        let mut level = crate::SampleLevel::Auto;
        let mut offset = None;
        while image_ops != 0 {
            let bit = 1 << image_ops.trailing_zeros();
            match spirv::ImageOperands::from_bits_truncate(bit) {
                spirv::ImageOperands::BIAS => {
                    let bias_expr = self.next()?;
                    let bias_handle = self.lookup_expression.lookup(bias_expr)?.handle;
                    level = crate::SampleLevel::Bias(bias_handle);
                    words_left -= 1;
                }
                spirv::ImageOperands::LOD => {
                    let lod_expr = self.next()?;
                    let lod_handle = self.lookup_expression.lookup(lod_expr)?.handle;
                    level = crate::SampleLevel::Exact(lod_handle);
                    words_left -= 1;
                }
                spirv::ImageOperands::CONST_OFFSET => {
                    let offset_constant = self.next()?;
                    let offset_handle = self.lookup_constant.lookup(offset_constant)?.handle;
                    offset = Some(offset_handle);
                    words_left -= 1;
                }
                other => {
                    log::warn!("Unknown image sample op {:?}", other);
                    for _ in 0..words_left {
                        self.next()?;
                    }
                    break;
                }
            }
            image_ops ^= bit;
        }

        let si_lexp = self.lookup_sampled_image.lookup(sampled_image_id)?;
        let coord_lexp = self.lookup_expression.lookup(coordinate_id)?;
        let coord_type_handle = self.lookup_type.lookup(coord_lexp.type_id)?.handle;

        let image_var_handle = expressions.get_global_var(si_lexp.image)?;
        let sampler_var_handle = expressions.get_global_var(si_lexp.sampler)?;
        log::debug!(
            "\t\t\tImage {:?} sampled with {:?}",
            image_var_handle,
            sampler_var_handle
        );
        if let Some(flags) = self.handle_sampling.get_mut(&image_var_handle) {
            *flags |= SamplingFlags::REGULAR;
        }
        *self.handle_sampling.get_mut(&sampler_var_handle).unwrap() |= SamplingFlags::REGULAR;

        let image_var = &global_arena[image_var_handle];
        let (coordinate, array_index) = match type_arena[image_var.ty].inner {
            crate::TypeInner::Image {
                dim,
                arrayed,
                class: _,
            } => extract_image_coordinates(
                dim,
                arrayed,
                coord_lexp.handle,
                coord_type_handle,
                type_arena,
                expressions,
            ),
            _ => return Err(Error::InvalidImage(image_var.ty)),
        };

        let expr = crate::Expression::ImageSample {
            image: si_lexp.image,
            sampler: si_lexp.sampler,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref: None,
        };
        self.lookup_expression.insert(
            result_id,
            LookupExpression {
                handle: expressions.append(expr),
                type_id: result_type_id,
            },
        );
        Ok(())
    }

    pub(super) fn parse_image_sample_dref(
        &mut self,
        mut words_left: u16,
        type_arena: &Arena<crate::Type>,
        global_arena: &Arena<crate::GlobalVariable>,
        expressions: &mut Arena<crate::Expression>,
    ) -> Result<(), Error> {
        let result_type_id = self.next()?;
        let result_id = self.next()?;
        let sampled_image_id = self.next()?;
        let coordinate_id = self.next()?;
        let dref_id = self.next()?;

        let mut image_ops = if words_left != 0 {
            words_left -= 1;
            self.next()?
        } else {
            0
        };

        let mut level = crate::SampleLevel::Auto;
        let mut offset = None;
        while image_ops != 0 {
            let bit = 1 << image_ops.trailing_zeros();
            match spirv::ImageOperands::from_bits_truncate(bit) {
                spirv::ImageOperands::BIAS => {
                    let bias_expr = self.next()?;
                    let bias_handle = self.lookup_expression.lookup(bias_expr)?.handle;
                    level = crate::SampleLevel::Bias(bias_handle);
                    words_left -= 1;
                }
                spirv::ImageOperands::LOD => {
                    let lod_expr = self.next()?;
                    let lod_handle = self.lookup_expression.lookup(lod_expr)?.handle;
                    level = crate::SampleLevel::Exact(lod_handle);
                    words_left -= 1;
                }
                spirv::ImageOperands::CONST_OFFSET => {
                    let offset_constant = self.next()?;
                    let offset_handle = self.lookup_constant.lookup(offset_constant)?.handle;
                    offset = Some(offset_handle);
                    words_left -= 1;
                }
                other => {
                    log::warn!("Unknown image sample dref op {:?}", other);
                    for _ in 0..words_left {
                        self.next()?;
                    }
                    break;
                }
            }
            image_ops ^= bit;
        }

        let si_lexp = self.lookup_sampled_image.lookup(sampled_image_id)?;
        let coord_lexp = self.lookup_expression.lookup(coordinate_id)?;
        let coord_type_handle = self.lookup_type.lookup(coord_lexp.type_id)?.handle;
        let image_var_handle = expressions.get_global_var(si_lexp.image)?;
        let sampler_var_handle = expressions.get_global_var(si_lexp.sampler)?;
        log::debug!(
            "\t\t\tImage {:?} sampled with comparison {:?}",
            image_var_handle,
            sampler_var_handle
        );
        if let Some(flags) = self.handle_sampling.get_mut(&image_var_handle) {
            *flags |= SamplingFlags::COMPARISON;
        }
        *self.handle_sampling.get_mut(&sampler_var_handle).unwrap() |= SamplingFlags::COMPARISON;

        let dref_lexp = self.lookup_expression.lookup(dref_id)?;
        let dref_type_handle = self.lookup_type.lookup(dref_lexp.type_id)?.handle;
        match type_arena[dref_type_handle].inner {
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Float,
                width: _,
            } => (),
            _ => return Err(Error::InvalidDepthReference(dref_type_handle)),
        }

        let image_var = &global_arena[image_var_handle];
        let (coordinate, array_index) = match type_arena[image_var.ty].inner {
            crate::TypeInner::Image {
                dim,
                arrayed,
                class: _,
            } => extract_image_coordinates(
                dim,
                arrayed,
                coord_lexp.handle,
                coord_type_handle,
                type_arena,
                expressions,
            ),
            _ => return Err(Error::InvalidImage(image_var.ty)),
        };

        let expr = crate::Expression::ImageSample {
            image: si_lexp.image,
            sampler: si_lexp.sampler,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref: Some(dref_lexp.handle),
        };
        self.lookup_expression.insert(
            result_id,
            LookupExpression {
                handle: expressions.append(expr),
                type_id: result_type_id,
            },
        );
        Ok(())
    }

    pub(super) fn parse_image_query_size(
        &mut self,
        at_level: bool,
        expressions: &mut Arena<crate::Expression>,
    ) -> Result<(), Error> {
        let result_type_id = self.next()?;
        let result_id = self.next()?;
        let image_id = self.next()?;
        let level = if at_level {
            let level_id = self.next()?;
            let level_lexp = self.lookup_expression.lookup(level_id)?;
            Some(level_lexp.handle)
        } else {
            None
        };

        //TODO: handle arrays and cubes
        let image_lexp = self.lookup_expression.lookup(image_id)?;

        let expr = crate::Expression::ImageQuery {
            image: image_lexp.handle,
            query: crate::ImageQuery::Size { level },
        };
        self.lookup_expression.insert(
            result_id,
            LookupExpression {
                handle: expressions.append(expr),
                type_id: result_type_id,
            },
        );
        Ok(())
    }

    pub(super) fn parse_image_query_other(
        &mut self,
        query: crate::ImageQuery,
        expressions: &mut Arena<crate::Expression>,
    ) -> Result<(), Error> {
        let result_type_id = self.next()?;
        let result_id = self.next()?;
        let image_id = self.next()?;

        let image_lexp = self.lookup_expression.lookup(image_id)?.clone();

        let expr = crate::Expression::ImageQuery {
            image: image_lexp.handle,
            query,
        };
        self.lookup_expression.insert(
            result_id,
            LookupExpression {
                handle: expressions.append(expr),
                type_id: result_type_id,
            },
        );
        Ok(())
    }
}
