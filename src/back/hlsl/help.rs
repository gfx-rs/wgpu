//! Helpers for the hlsl backend
//!
//! Important note about `Expression::ImageQuery`/`Expression::ArrayLength` and hlsl backend:
//!
//! Due to implementation of `GetDimensions` function in hlsl (<https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-getdimensions>)
//! backend can't work with it as an expression.
//! Instead, it generates a unique wrapped function per `Expression::ImageQuery`, based on texure info and query function.
//! See `WrappedImageQuery` struct that represents a unique function and will be generated before writing all statements and expressions.
//! This allowed to works with `Expression::ImageQuery` as expression and write wrapped function.
//!
//! For example:
//! ```wgsl
//! let dim_1d = textureDimensions(image_1d);
//! ```
//!
//! ```hlsl
//! int NagaDimensions1D(Texture1D<float4>)
//! {
//!    uint4 ret;
//!    image_1d.GetDimensions(ret.x);
//!    return ret.x;
//! }
//!
//! int dim_1d = NagaDimensions1D(image_1d);
//! ```

use super::{super::FunctionCtx, BackendResult, Error};
use crate::{arena::Handle, proc::NameKey};
use std::fmt::Write;

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WrappedArrayLength {
    pub(super) writable: bool,
}

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WrappedImageQuery {
    pub(super) dim: crate::ImageDimension,
    pub(super) arrayed: bool,
    pub(super) class: crate::ImageClass,
    pub(super) query: ImageQuery,
}

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WrappedConstructor {
    pub(super) ty: Handle<crate::Type>,
}

/// HLSL backend requires its own `ImageQuery` enum.
///
/// It is used inside `WrappedImageQuery` and should be unique per ImageQuery function.
/// IR version can't be unique per function, because it's store mipmap level as an expression.
///
/// For example:
/// ```wgsl
/// let dim_cube_array_lod = textureDimensions(image_cube_array, 1);
/// let dim_cube_array_lod2 = textureDimensions(image_cube_array, 1);
/// ```
///
/// ```ir
/// ImageQuery {
///  image: [1],
///  query: Size {
///      level: Some(
///          [1],
///      ),
///  },
/// },
/// ImageQuery {
///  image: [1],
///  query: Size {
///      level: Some(
///          [2],
///      ),
///  },
/// },
/// ```
///
/// HLSL should generate only 1 function for this case.
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
    pub(super) fn write_image_type(
        &mut self,
        dim: crate::ImageDimension,
        arrayed: bool,
        class: crate::ImageClass,
    ) -> BackendResult {
        let access_str = match class {
            crate::ImageClass::Storage { .. } => "RW",
            _ => "",
        };
        let dim_str = dim.to_hlsl_str();
        let arrayed_str = if arrayed { "Array" } else { "" };
        write!(self.out, "{}Texture{}{}", access_str, dim_str, arrayed_str)?;
        match class {
            crate::ImageClass::Depth { multi } => {
                let multi_str = if multi { "MS" } else { "" };
                write!(self.out, "{}<float>", multi_str)?
            }
            crate::ImageClass::Sampled { kind, multi } => {
                let multi_str = if multi { "MS" } else { "" };
                let scalar_kind_str = kind.to_hlsl_str(4)?;
                write!(self.out, "{}<{}4>", multi_str, scalar_kind_str)?
            }
            crate::ImageClass::Storage { format, .. } => {
                let storage_format_str = format.to_hlsl_str();
                write!(self.out, "<{}>", storage_format_str)?
            }
        }
        Ok(())
    }

    pub(super) fn write_wrapped_array_length_function_name(
        &mut self,
        query: WrappedArrayLength,
    ) -> BackendResult {
        let access_str = if query.writable { "RW" } else { "" };
        write!(self.out, "NagaBufferLength{}", access_str,)?;

        Ok(())
    }

    /// Helper function that write wrapped function for `Expression::ArrayLength`
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer-getdimensions>
    pub(super) fn write_wrapped_array_length_function(
        &mut self,
        module: &crate::Module,
        wal: WrappedArrayLength,
        expr_handle: Handle<crate::Expression>,
        func_ctx: &FunctionCtx,
    ) -> BackendResult {
        use crate::back::INDENT;

        const ARGUMENT_VARIABLE_NAME: &str = "buffer";
        const RETURN_VARIABLE_NAME: &str = "ret";

        // Write function return type and name
        let ret_ty = func_ctx.info[expr_handle].ty.inner_with(&module.types);
        self.write_value_type(module, ret_ty)?;
        write!(self.out, " ")?;
        self.write_wrapped_array_length_function_name(wal)?;

        // Write function parameters
        write!(self.out, "(")?;
        let access_str = if wal.writable { "RW" } else { "" };
        writeln!(
            self.out,
            "{}ByteAddressBuffer {})",
            access_str, ARGUMENT_VARIABLE_NAME
        )?;
        // Write function body
        writeln!(self.out, "{{")?;

        // Write `GetDimensions` function.
        writeln!(self.out, "{}uint {};", INDENT, RETURN_VARIABLE_NAME)?;
        writeln!(
            self.out,
            "{}{}.GetDimensions({});",
            INDENT, ARGUMENT_VARIABLE_NAME, RETURN_VARIABLE_NAME
        )?;

        // Write return value
        writeln!(self.out, "{}return {};", INDENT, RETURN_VARIABLE_NAME)?;

        // End of function body
        writeln!(self.out, "}}")?;
        // Write extra new line
        writeln!(self.out)?;

        Ok(())
    }

    pub(super) fn write_wrapped_image_query_function_name(
        &mut self,
        query: WrappedImageQuery,
    ) -> BackendResult {
        let dim_str = query.dim.to_hlsl_str();
        let class_str = match query.class {
            crate::ImageClass::Sampled { multi: true, .. } => "MS",
            crate::ImageClass::Depth { multi: true } => "DepthMS",
            crate::ImageClass::Depth { multi: false } => "Depth",
            crate::ImageClass::Sampled { multi: false, .. } => "",
            crate::ImageClass::Storage { .. } => "RW",
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
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-getdimensions>
    pub(super) fn write_wrapped_image_query_function(
        &mut self,
        module: &crate::Module,
        wiq: WrappedImageQuery,
        expr_handle: Handle<crate::Expression>,
        func_ctx: &FunctionCtx,
    ) -> BackendResult {
        use crate::{
            back::{COMPONENTS, INDENT},
            ImageDimension as IDim,
        };

        const ARGUMENT_VARIABLE_NAME: &str = "tex";
        const RETURN_VARIABLE_NAME: &str = "ret";
        const MIP_LEVEL_PARAM: &str = "mip_level";

        // Write function return type and name
        let ret_ty = func_ctx.info[expr_handle].ty.inner_with(&module.types);
        self.write_value_type(module, ret_ty)?;
        write!(self.out, " ")?;
        self.write_wrapped_image_query_function_name(wiq)?;

        // Write function parameters
        write!(self.out, "(")?;
        // Texture always first parameter
        self.write_image_type(wiq.dim, wiq.arrayed, wiq.class)?;
        write!(self.out, " {}", ARGUMENT_VARIABLE_NAME)?;
        // Mipmap is a second parameter if exists
        if let ImageQuery::SizeLevel = wiq.query {
            write!(self.out, ", uint {}", MIP_LEVEL_PARAM)?;
        }
        writeln!(self.out, ")")?;

        // Write function body
        writeln!(self.out, "{{")?;

        let array_coords = if wiq.arrayed { 1 } else { 0 };
        // extra parameter is the mip level count or the sample count
        let extra_coords = match wiq.class {
            crate::ImageClass::Storage { .. } => 0,
            crate::ImageClass::Sampled { .. } | crate::ImageClass::Depth { .. } => 1,
        };

        // GetDimensions Overloaded Methods
        // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-getdimensions#overloaded-methods
        let (ret_swizzle, number_of_params) = match wiq.query {
            ImageQuery::Size | ImageQuery::SizeLevel => {
                let ret = match wiq.dim {
                    IDim::D1 => "x",
                    IDim::D2 => "xy",
                    IDim::D3 => "xyz",
                    IDim::Cube => "xy",
                };
                (ret, ret.len() + array_coords + extra_coords)
            }
            ImageQuery::NumLevels | ImageQuery::NumSamples | ImageQuery::NumLayers => {
                if wiq.arrayed || wiq.dim == IDim::D3 {
                    ("w", 4)
                } else {
                    ("z", 3)
                }
            }
        };

        // Write `GetDimensions` function.
        writeln!(self.out, "{}uint4 {};", INDENT, RETURN_VARIABLE_NAME)?;
        write!(
            self.out,
            "{}{}.GetDimensions(",
            INDENT, ARGUMENT_VARIABLE_NAME
        )?;
        match wiq.query {
            ImageQuery::SizeLevel => {
                write!(self.out, "{}, ", MIP_LEVEL_PARAM)?;
            }
            _ => match wiq.class {
                crate::ImageClass::Sampled { multi: true, .. }
                | crate::ImageClass::Depth { multi: true }
                | crate::ImageClass::Storage { .. } => {}
                _ => {
                    // Write zero mipmap level for supported types
                    write!(self.out, "0, ")?;
                }
            },
        }

        for component in COMPONENTS[..number_of_params - 1].iter() {
            write!(self.out, "{}.{}, ", RETURN_VARIABLE_NAME, component)?;
        }

        // write last parameter without comma and space for last parameter
        write!(
            self.out,
            "{}.{}",
            RETURN_VARIABLE_NAME,
            COMPONENTS[number_of_params - 1]
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

        Ok(())
    }

    pub(super) fn write_wrapped_constructor_function_name(
        &mut self,
        constructor: WrappedConstructor,
    ) -> BackendResult {
        let name = &self.names[&NameKey::Type(constructor.ty)];
        write!(self.out, "Construct{}", name)?;
        Ok(())
    }

    /// Helper function that write wrapped function for `Expression::Compose` for structures.
    pub(super) fn write_wrapped_constructor_function(
        &mut self,
        module: &crate::Module,
        constructor: WrappedConstructor,
    ) -> BackendResult {
        use crate::back::INDENT;

        const ARGUMENT_VARIABLE_NAME: &str = "arg";
        const RETURN_VARIABLE_NAME: &str = "ret";

        // Write function return type and name
        let struct_name = &self.names[&NameKey::Type(constructor.ty)];
        write!(self.out, "{} ", struct_name)?;
        self.write_wrapped_constructor_function_name(constructor)?;

        // Write function parameters
        write!(self.out, "(")?;
        let members = match module.types[constructor.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => members,
            _ => return Err(Error::Unimplemented("non-struct constructor".to_string())),
        };
        for (i, member) in members.iter().enumerate() {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            self.write_type(module, member.ty)?;
            write!(self.out, " {}{}", ARGUMENT_VARIABLE_NAME, i)?;
            if let crate::TypeInner::Array { size, .. } = module.types[member.ty].inner {
                self.write_array_size(module, size)?;
            }
        }
        // Write function body
        writeln!(self.out, ") {{")?;

        let struct_name = &self.names[&NameKey::Type(constructor.ty)];
        writeln!(
            self.out,
            "{}{} {};",
            INDENT, struct_name, RETURN_VARIABLE_NAME
        )?;
        for i in 0..members.len() as u32 {
            let field_name = &self.names[&NameKey::StructMember(constructor.ty, i)];
            //TODO: handle arrays?
            writeln!(
                self.out,
                "{}{}.{} = {}{};",
                INDENT, RETURN_VARIABLE_NAME, field_name, ARGUMENT_VARIABLE_NAME, i,
            )?;
        }

        // Write return value
        writeln!(self.out, "{}return {};", INDENT, RETURN_VARIABLE_NAME)?;

        // End of function body
        writeln!(self.out, "}}")?;
        // Write extra new line
        writeln!(self.out)?;

        Ok(())
    }

    /// Helper function that write wrapped function for `Expression::ImageQuery` and `Expression::ArrayLength`
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-getdimensions>
    pub(super) fn write_wrapped_functions(
        &mut self,
        module: &crate::Module,
        func_ctx: &FunctionCtx,
    ) -> BackendResult {
        for (handle, _) in func_ctx.expressions.iter() {
            match func_ctx.expressions[handle] {
                crate::Expression::ArrayLength(expr) => {
                    let global_expr = match func_ctx.expressions[expr] {
                        crate::Expression::AccessIndex { base, index: _ } => base,
                        ref other => unreachable!("Array length of {:?}", other),
                    };
                    let global_var = match func_ctx.expressions[global_expr] {
                        crate::Expression::GlobalVariable(var_handle) => {
                            &module.global_variables[var_handle]
                        }
                        ref other => unreachable!("Array length of base {:?}", other),
                    };
                    let storage_access = match global_var.class {
                        crate::StorageClass::Storage { access } => access,
                        _ => crate::StorageAccess::default(),
                    };
                    let wal = WrappedArrayLength {
                        writable: storage_access.contains(crate::StorageAccess::STORE),
                    };

                    if !self.wrapped.array_lengths.contains(&wal) {
                        self.write_wrapped_array_length_function(module, wal, handle, func_ctx)?;
                        self.wrapped.array_lengths.insert(wal);
                    }
                }
                crate::Expression::ImageQuery { image, query } => {
                    let wiq = match *func_ctx.info[image].ty.inner_with(&module.types) {
                        crate::TypeInner::Image {
                            dim,
                            arrayed,
                            class,
                        } => WrappedImageQuery {
                            dim,
                            arrayed,
                            class,
                            query: query.into(),
                        },
                        _ => unreachable!("we only query images"),
                    };

                    if !self.wrapped.image_queries.contains(&wiq) {
                        self.write_wrapped_image_query_function(module, wiq, handle, func_ctx)?;
                        self.wrapped.image_queries.insert(wiq);
                    }
                }
                crate::Expression::Compose { ty, components: _ } => {
                    let constructor = match module.types[ty].inner {
                        crate::TypeInner::Struct { .. } => WrappedConstructor { ty },
                        _ => continue,
                    };
                    if !self.wrapped.constructors.contains(&constructor) {
                        self.write_wrapped_constructor_function(module, constructor)?;
                        self.wrapped.constructors.insert(constructor);
                    }
                }
                _ => {}
            };
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
