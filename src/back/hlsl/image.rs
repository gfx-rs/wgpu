// Important note about `Expression::ImageQuery` and hlsl backend:
// Due to implementation of `GetDimensions` function in hlsl (https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-getdimensions)
// backend can't work with it as an expression.
// Instead, it generates a unique wrapped function per `Expression::ImageQuery`, based on texure info and query function.
// See `WrappedImageQuery` struct that represents a unique function and will be generated before writing all statements and expressions.
// This allowed to works with `Expression::ImageQuery` as expression and write wrapped function.
//
// For example:
// ```wgsl
// let dim_1d = textureDimensions(image_1d);
// ```
//
// ```hlsl
// int NagaDimensions1D(Texture1D<float4>)
// {
//    uint4 ret;
//    image_1d.GetDimensions(ret.x);
//    return ret.x;
// }
//
// int dim_1d = NagaDimensions1D(image_1d);
// ```

use super::{super::FunctionCtx, writer::BackendResult};
use crate::arena::Handle;
use std::fmt::Write;

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WrappedImageQuery {
    pub(super) dim: crate::ImageDimension,
    pub(super) arrayed: bool,
    pub(super) class: crate::ImageClass,
    pub(super) query: ImageQuery,
}

// HLSL backend requires its own `ImageQuery` enum.
// It is used inside `WrappedImageQuery` and should be unique per ImageQuery function.
// IR version can't be unique per function, because it's store mipmap level as an expression.
//
// For example:
// ```wgsl
// let dim_cube_array_lod = textureDimensions(image_cube_array, 1);
// let dim_cube_array_lod2 = textureDimensions(image_cube_array, 1);
// ```
//
// ```ir
// ImageQuery {
//  image: [1],
//  query: Size {
//      level: Some(
//          [1],
//      ),
//  },
// },
// ImageQuery {
//  image: [1],
//  query: Size {
//      level: Some(
//          [2],
//      ),
//  },
// },
// ```
//
// HLSL should generate only 1 function for this case.
//
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum ImageQuery {
    Size,
    SizeLevel,
    NumLevels,
    NumLayers,
    NumSamples,
}

impl From<crate::ImageQuery> for ImageQuery {
    fn from(q: crate::ImageQuery) -> Self {
        use crate::ImageQuery as Iq;
        match q {
            Iq::Size { level: Some(_) } => ImageQuery::SizeLevel,
            Iq::Size { level: None } => ImageQuery::Size,
            Iq::NumLevels => ImageQuery::NumLevels,
            Iq::NumLayers => ImageQuery::NumLayers,
            Iq::NumSamples => ImageQuery::NumSamples,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum MipLevelCoordinate {
    NotApplicable,
    Zero,
    Expression(Handle<crate::Expression>),
}

impl<'a, W: Write> super::Writer<'a, W> {
    pub(super) fn write_wrapped_image_query_function_name(
        &mut self,
        query: WrappedImageQuery,
    ) -> BackendResult {
        let dim_str = query.dim.to_hlsl_str();
        let class_str = match query.class {
            crate::ImageClass::Sampled { multi: true, .. } => "MS",
            crate::ImageClass::Depth { multi: true } => "DepthMS",
            crate::ImageClass::Depth { multi: false } => "Depth",
            crate::ImageClass::Sampled { multi: false, .. } | crate::ImageClass::Storage { .. } => {
                ""
            }
        };
        let arrayed_str = if query.arrayed { "Array" } else { "" };
        let query_str = match query.query {
            ImageQuery::Size => "Dimensions",
            ImageQuery::SizeLevel => "MipDimensions",
            ImageQuery::NumLevels => "NumLevels",
            ImageQuery::NumLayers => "NumLayers",
            ImageQuery::NumSamples => "NumSamples",
        };

        write!(
            self.out,
            "Naga{}{}{}{}",
            class_str, query_str, dim_str, arrayed_str
        )?;

        Ok(())
    }

    /// Helper function that write wrapped function for `Expression::ImageQuery`
    /// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-getdimensions
    pub(super) fn write_wrapped_image_query_functions(
        &mut self,
        module: &crate::Module,
        func_ctx: &FunctionCtx,
    ) -> BackendResult {
        use crate::{back::INDENT, ImageDimension as IDim, ImageQuery as Iq};

        const RETURN_VARIABLE_NAME: &str = "ret";
        const MIP_LEVEL_PARAM: &str = "MipLevel";

        for (handle, _) in func_ctx.expressions.iter() {
            let (image, query) = match func_ctx.expressions[handle] {
                crate::Expression::ImageQuery { image, query } => (image, query),
                _ => continue,
            };
            let image_ty = func_ctx.info[image].ty.inner_with(&module.types);
            let ret_ty = func_ctx.info[handle].ty.inner_with(&module.types);

            let (dim, arrayed, class) = match *image_ty {
                crate::TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => (dim, arrayed, class),
                _ => unreachable!("we only query images"),
            };

            let wrapped_image_query = WrappedImageQuery {
                dim,
                arrayed,
                class,
                query: query.into(),
            };

            if self.wrapped_image_queries.contains(&wrapped_image_query) {
                continue;
            }

            // Write function return type and name
            self.write_value_type(module, ret_ty)?;
            write!(self.out, " ")?;
            self.write_wrapped_image_query_function_name(wrapped_image_query)?;

            // Write function parameters
            write!(self.out, "(")?;
            // Texture always first parameter
            self.write_value_type(module, image_ty)?;
            // Mipmap is a second parameter if exists
            if let crate::ImageQuery::Size { level: Some(_) } = query {
                write!(self.out, ", uint {}", MIP_LEVEL_PARAM)?;
            }
            writeln!(self.out, ")")?;

            // Write function body
            writeln!(self.out, "{{")?;

            let array_coords = if arrayed { 1 } else { 0 };
            // GetDimensions Overloaded Methods
            // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-getdimensions#overloaded-methods
            let (ret_swizzle, number_of_params) = match query {
                Iq::Size { .. } => match dim {
                    IDim::D1 => ("x", 1 + array_coords),
                    IDim::D2 => ("xy", 3 + array_coords),
                    IDim::D3 => ("xyz", 4),
                    IDim::Cube => ("xy", 3 + array_coords),
                },
                Iq::NumLevels | Iq::NumSamples | Iq::NumLayers => {
                    if arrayed || dim == IDim::D3 {
                        ("w", 4)
                    } else {
                        ("z", 3)
                    }
                }
            };

            // Write `GetDimensions` function.
            writeln!(self.out, "{}uint4 {};", INDENT, RETURN_VARIABLE_NAME)?;
            write!(self.out, "{}", INDENT)?;
            self.write_expr(module, image, func_ctx)?;
            write!(self.out, ".GetDimensions(")?;
            match query {
                Iq::Size { level: Some(_) } => {
                    write!(self.out, "{}, ", MIP_LEVEL_PARAM)?;
                }
                _ => match class {
                    crate::ImageClass::Sampled { multi: true, .. }
                    | crate::ImageClass::Depth { multi: true } => {}
                    _ => match dim {
                        // Write zero mipmap level for supported types
                        IDim::D2 | IDim::D3 | IDim::Cube => {
                            write!(self.out, "0, ")?;
                        }
                        IDim::D1 => {}
                    },
                },
            }

            for component in crate::back::COMPONENTS[..number_of_params - 1].iter() {
                write!(self.out, "{}.{}, ", RETURN_VARIABLE_NAME, component)?;
            }

            // write last parameter without comma and space for last parameter
            write!(
                self.out,
                "{}.{}",
                RETURN_VARIABLE_NAME,
                crate::back::COMPONENTS[number_of_params - 1]
            )?;

            writeln!(self.out, ");")?;

            // Write return value
            writeln!(
                self.out,
                "{}return {}.{};",
                INDENT, RETURN_VARIABLE_NAME, ret_swizzle
            )?;

            // End of function body
            writeln!(self.out, "}}")?;
            // Write extra new line
            writeln!(self.out)?;

            self.wrapped_image_queries.insert(wrapped_image_query);
        }

        Ok(())
    }

    pub(super) fn write_texture_coordinates(
        &mut self,
        kind: &str,
        coordinate: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        mip_level: MipLevelCoordinate,
        module: &crate::Module,
        func_ctx: &FunctionCtx,
    ) -> BackendResult {
        // HLSL expects the array index to be merged with the coordinate
        let extra = array_index.is_some() as usize
            + (mip_level != MipLevelCoordinate::NotApplicable) as usize;
        if extra == 0 {
            self.write_expr(module, coordinate, func_ctx)?;
        } else {
            let num_coords = match *func_ctx.info[coordinate].ty.inner_with(&module.types) {
                crate::TypeInner::Scalar { .. } => 1,
                crate::TypeInner::Vector { size, .. } => size as usize,
                _ => unreachable!(),
            };
            write!(self.out, "{}{}(", kind, num_coords + extra)?;
            self.write_expr(module, coordinate, func_ctx)?;
            if let Some(expr) = array_index {
                write!(self.out, ", ")?;
                self.write_expr(module, expr, func_ctx)?;
            }
            match mip_level {
                MipLevelCoordinate::NotApplicable => {}
                MipLevelCoordinate::Zero => {
                    write!(self.out, ", 0")?;
                }
                MipLevelCoordinate::Expression(expr) => {
                    write!(self.out, ", ")?;
                    self.write_expr(module, expr, func_ctx)?;
                }
            }
            write!(self.out, ")")?;
        }
        Ok(())
    }
}
