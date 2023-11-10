/*!
Helpers for the hlsl backend

Important note about `Expression::ImageQuery`/`Expression::ArrayLength` and hlsl backend:

Due to implementation of `GetDimensions` function in hlsl (<https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-getdimensions>)
backend can't work with it as an expression.
Instead, it generates a unique wrapped function per `Expression::ImageQuery`, based on texture info and query function.
See `WrappedImageQuery` struct that represents a unique function and will be generated before writing all statements and expressions.
This allowed to works with `Expression::ImageQuery` as expression and write wrapped function.

For example:
```wgsl
let dim_1d = textureDimensions(image_1d);
```

```hlsl
int NagaDimensions1D(Texture1D<float4>)
{
   uint4 ret;
   image_1d.GetDimensions(ret.x);
   return ret.x;
}

int dim_1d = NagaDimensions1D(image_1d);
```
*/

use super::{super::FunctionCtx, BackendResult};
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

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WrappedStructMatrixAccess {
    pub(super) ty: Handle<crate::Type>,
    pub(super) index: u32,
}

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WrappedMatCx2 {
    pub(super) columns: crate::VectorSize,
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
        write!(self.out, "{access_str}Texture{dim_str}{arrayed_str}")?;
        match class {
            crate::ImageClass::Depth { multi } => {
                let multi_str = if multi { "MS" } else { "" };
                write!(self.out, "{multi_str}<float>")?
            }
            crate::ImageClass::Sampled { kind, multi } => {
                let multi_str = if multi { "MS" } else { "" };
                let scalar_kind_str = crate::Scalar { kind, width: 4 }.to_hlsl_str()?;
                write!(self.out, "{multi_str}<{scalar_kind_str}4>")?
            }
            crate::ImageClass::Storage { format, .. } => {
                let storage_format_str = format.to_hlsl_str();
                write!(self.out, "<{storage_format_str}>")?
            }
        }
        Ok(())
    }

    pub(super) fn write_wrapped_array_length_function_name(
        &mut self,
        query: WrappedArrayLength,
    ) -> BackendResult {
        let access_str = if query.writable { "RW" } else { "" };
        write!(self.out, "NagaBufferLength{access_str}",)?;

        Ok(())
    }

    /// Helper function that write wrapped function for `Expression::ArrayLength`
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer-getdimensions>
    pub(super) fn write_wrapped_array_length_function(
        &mut self,
        wal: WrappedArrayLength,
    ) -> BackendResult {
        use crate::back::INDENT;

        const ARGUMENT_VARIABLE_NAME: &str = "buffer";
        const RETURN_VARIABLE_NAME: &str = "ret";

        // Write function return type and name
        write!(self.out, "uint ")?;
        self.write_wrapped_array_length_function_name(wal)?;

        // Write function parameters
        write!(self.out, "(")?;
        let access_str = if wal.writable { "RW" } else { "" };
        writeln!(
            self.out,
            "{access_str}ByteAddressBuffer {ARGUMENT_VARIABLE_NAME})"
        )?;
        // Write function body
        writeln!(self.out, "{{")?;

        // Write `GetDimensions` function.
        writeln!(self.out, "{INDENT}uint {RETURN_VARIABLE_NAME};")?;
        writeln!(
            self.out,
            "{INDENT}{ARGUMENT_VARIABLE_NAME}.GetDimensions({RETURN_VARIABLE_NAME});"
        )?;

        // Write return value
        writeln!(self.out, "{INDENT}return {RETURN_VARIABLE_NAME};")?;

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

        write!(self.out, "Naga{class_str}{query_str}{dim_str}{arrayed_str}")?;

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
        let ret_ty = func_ctx.resolve_type(expr_handle, &module.types);
        self.write_value_type(module, ret_ty)?;
        write!(self.out, " ")?;
        self.write_wrapped_image_query_function_name(wiq)?;

        // Write function parameters
        write!(self.out, "(")?;
        // Texture always first parameter
        self.write_image_type(wiq.dim, wiq.arrayed, wiq.class)?;
        write!(self.out, " {ARGUMENT_VARIABLE_NAME}")?;
        // Mipmap is a second parameter if exists
        if let ImageQuery::SizeLevel = wiq.query {
            write!(self.out, ", uint {MIP_LEVEL_PARAM}")?;
        }
        writeln!(self.out, ")")?;

        // Write function body
        writeln!(self.out, "{{")?;

        let array_coords = usize::from(wiq.arrayed);
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
        writeln!(self.out, "{INDENT}uint4 {RETURN_VARIABLE_NAME};")?;
        write!(self.out, "{INDENT}{ARGUMENT_VARIABLE_NAME}.GetDimensions(")?;
        match wiq.query {
            ImageQuery::SizeLevel => {
                write!(self.out, "{MIP_LEVEL_PARAM}, ")?;
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
            write!(self.out, "{RETURN_VARIABLE_NAME}.{component}, ")?;
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
            "{INDENT}return {RETURN_VARIABLE_NAME}.{ret_swizzle};"
        )?;

        // End of function body
        writeln!(self.out, "}}")?;
        // Write extra new line
        writeln!(self.out)?;

        Ok(())
    }

    pub(super) fn write_wrapped_constructor_function_name(
        &mut self,
        module: &crate::Module,
        constructor: WrappedConstructor,
    ) -> BackendResult {
        let name = crate::TypeInner::hlsl_type_id(constructor.ty, module.to_ctx(), &self.names)?;
        write!(self.out, "Construct{name}")?;
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
        if let crate::TypeInner::Array { base, size, .. } = module.types[constructor.ty].inner {
            write!(self.out, "typedef ")?;
            self.write_type(module, constructor.ty)?;
            write!(self.out, " ret_")?;
            self.write_wrapped_constructor_function_name(module, constructor)?;
            self.write_array_size(module, base, size)?;
            writeln!(self.out, ";")?;

            write!(self.out, "ret_")?;
            self.write_wrapped_constructor_function_name(module, constructor)?;
        } else {
            self.write_type(module, constructor.ty)?;
        }
        write!(self.out, " ")?;
        self.write_wrapped_constructor_function_name(module, constructor)?;

        // Write function parameters
        write!(self.out, "(")?;

        let mut write_arg = |i, ty| -> BackendResult {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            self.write_type(module, ty)?;
            write!(self.out, " {ARGUMENT_VARIABLE_NAME}{i}")?;
            if let crate::TypeInner::Array { base, size, .. } = module.types[ty].inner {
                self.write_array_size(module, base, size)?;
            }
            Ok(())
        };

        match module.types[constructor.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => {
                for (i, member) in members.iter().enumerate() {
                    write_arg(i, member.ty)?;
                }
            }
            crate::TypeInner::Array {
                base,
                size: crate::ArraySize::Constant(size),
                ..
            } => {
                for i in 0..size.get() as usize {
                    write_arg(i, base)?;
                }
            }
            _ => unreachable!(),
        };

        write!(self.out, ")")?;

        // Write function body
        writeln!(self.out, " {{")?;

        match module.types[constructor.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => {
                let struct_name = &self.names[&NameKey::Type(constructor.ty)];
                writeln!(
                    self.out,
                    "{INDENT}{struct_name} {RETURN_VARIABLE_NAME} = ({struct_name})0;"
                )?;
                for (i, member) in members.iter().enumerate() {
                    let field_name = &self.names[&NameKey::StructMember(constructor.ty, i as u32)];

                    match module.types[member.ty].inner {
                        crate::TypeInner::Matrix {
                            columns,
                            rows: crate::VectorSize::Bi,
                            ..
                        } if member.binding.is_none() => {
                            for j in 0..columns as u8 {
                                writeln!(
                                    self.out,
                                    "{INDENT}{RETURN_VARIABLE_NAME}.{field_name}_{j} = {ARGUMENT_VARIABLE_NAME}{i}[{j}];"
                                )?;
                            }
                        }
                        ref other => {
                            // We cast arrays of native HLSL `floatCx2`s to arrays of `matCx2`s
                            // (where the inner matrix is represented by a struct with C `float2` members).
                            // See the module-level block comment in mod.rs for details.
                            if let Some(super::writer::MatrixType {
                                columns,
                                rows: crate::VectorSize::Bi,
                                width: 4,
                            }) = super::writer::get_inner_matrix_data(module, member.ty)
                            {
                                write!(
                                    self.out,
                                    "{}{}.{} = (__mat{}x2",
                                    INDENT, RETURN_VARIABLE_NAME, field_name, columns as u8
                                )?;
                                if let crate::TypeInner::Array { base, size, .. } = *other {
                                    self.write_array_size(module, base, size)?;
                                }
                                writeln!(self.out, "){ARGUMENT_VARIABLE_NAME}{i};",)?;
                            } else {
                                writeln!(
                                    self.out,
                                    "{INDENT}{RETURN_VARIABLE_NAME}.{field_name} = {ARGUMENT_VARIABLE_NAME}{i};",
                                )?;
                            }
                        }
                    }
                }
            }
            crate::TypeInner::Array {
                base,
                size: crate::ArraySize::Constant(size),
                ..
            } => {
                write!(self.out, "{INDENT}")?;
                self.write_type(module, base)?;
                write!(self.out, " {RETURN_VARIABLE_NAME}")?;
                self.write_array_size(module, base, crate::ArraySize::Constant(size))?;
                write!(self.out, " = {{ ")?;
                for i in 0..size.get() {
                    if i != 0 {
                        write!(self.out, ", ")?;
                    }
                    write!(self.out, "{ARGUMENT_VARIABLE_NAME}{i}")?;
                }
                writeln!(self.out, " }};",)?;
            }
            _ => unreachable!(),
        }

        // Write return value
        writeln!(self.out, "{INDENT}return {RETURN_VARIABLE_NAME};")?;

        // End of function body
        writeln!(self.out, "}}")?;
        // Write extra new line
        writeln!(self.out)?;

        Ok(())
    }

    pub(super) fn write_wrapped_struct_matrix_get_function_name(
        &mut self,
        access: WrappedStructMatrixAccess,
    ) -> BackendResult {
        let name = &self.names[&NameKey::Type(access.ty)];
        let field_name = &self.names[&NameKey::StructMember(access.ty, access.index)];
        write!(self.out, "GetMat{field_name}On{name}")?;
        Ok(())
    }

    /// Writes a function used to get a matCx2 from within a structure.
    pub(super) fn write_wrapped_struct_matrix_get_function(
        &mut self,
        module: &crate::Module,
        access: WrappedStructMatrixAccess,
    ) -> BackendResult {
        use crate::back::INDENT;

        const STRUCT_ARGUMENT_VARIABLE_NAME: &str = "obj";

        // Write function return type and name
        let member = match module.types[access.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => &members[access.index as usize],
            _ => unreachable!(),
        };
        let ret_ty = &module.types[member.ty].inner;
        self.write_value_type(module, ret_ty)?;
        write!(self.out, " ")?;
        self.write_wrapped_struct_matrix_get_function_name(access)?;

        // Write function parameters
        write!(self.out, "(")?;
        let struct_name = &self.names[&NameKey::Type(access.ty)];
        write!(self.out, "{struct_name} {STRUCT_ARGUMENT_VARIABLE_NAME}")?;

        // Write function body
        writeln!(self.out, ") {{")?;

        // Write return value
        write!(self.out, "{INDENT}return ")?;
        self.write_value_type(module, ret_ty)?;
        write!(self.out, "(")?;
        let field_name = &self.names[&NameKey::StructMember(access.ty, access.index)];
        match module.types[member.ty].inner {
            crate::TypeInner::Matrix { columns, .. } => {
                for i in 0..columns as u8 {
                    if i != 0 {
                        write!(self.out, ", ")?;
                    }
                    write!(self.out, "{STRUCT_ARGUMENT_VARIABLE_NAME}.{field_name}_{i}")?;
                }
            }
            _ => unreachable!(),
        }
        writeln!(self.out, ");")?;

        // End of function body
        writeln!(self.out, "}}")?;
        // Write extra new line
        writeln!(self.out)?;

        Ok(())
    }

    pub(super) fn write_wrapped_struct_matrix_set_function_name(
        &mut self,
        access: WrappedStructMatrixAccess,
    ) -> BackendResult {
        let name = &self.names[&NameKey::Type(access.ty)];
        let field_name = &self.names[&NameKey::StructMember(access.ty, access.index)];
        write!(self.out, "SetMat{field_name}On{name}")?;
        Ok(())
    }

    /// Writes a function used to set a matCx2 from within a structure.
    pub(super) fn write_wrapped_struct_matrix_set_function(
        &mut self,
        module: &crate::Module,
        access: WrappedStructMatrixAccess,
    ) -> BackendResult {
        use crate::back::INDENT;

        const STRUCT_ARGUMENT_VARIABLE_NAME: &str = "obj";
        const MATRIX_ARGUMENT_VARIABLE_NAME: &str = "mat";

        // Write function return type and name
        write!(self.out, "void ")?;
        self.write_wrapped_struct_matrix_set_function_name(access)?;

        // Write function parameters
        write!(self.out, "(")?;
        let struct_name = &self.names[&NameKey::Type(access.ty)];
        write!(self.out, "{struct_name} {STRUCT_ARGUMENT_VARIABLE_NAME}, ")?;
        let member = match module.types[access.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => &members[access.index as usize],
            _ => unreachable!(),
        };
        self.write_type(module, member.ty)?;
        write!(self.out, " {MATRIX_ARGUMENT_VARIABLE_NAME}")?;
        // Write function body
        writeln!(self.out, ") {{")?;

        let field_name = &self.names[&NameKey::StructMember(access.ty, access.index)];

        match module.types[member.ty].inner {
            crate::TypeInner::Matrix { columns, .. } => {
                for i in 0..columns as u8 {
                    writeln!(
                        self.out,
                        "{INDENT}{STRUCT_ARGUMENT_VARIABLE_NAME}.{field_name}_{i} = {MATRIX_ARGUMENT_VARIABLE_NAME}[{i}];"
                    )?;
                }
            }
            _ => unreachable!(),
        }

        // End of function body
        writeln!(self.out, "}}")?;
        // Write extra new line
        writeln!(self.out)?;

        Ok(())
    }

    pub(super) fn write_wrapped_struct_matrix_set_vec_function_name(
        &mut self,
        access: WrappedStructMatrixAccess,
    ) -> BackendResult {
        let name = &self.names[&NameKey::Type(access.ty)];
        let field_name = &self.names[&NameKey::StructMember(access.ty, access.index)];
        write!(self.out, "SetMatVec{field_name}On{name}")?;
        Ok(())
    }

    /// Writes a function used to set a vec2 on a matCx2 from within a structure.
    pub(super) fn write_wrapped_struct_matrix_set_vec_function(
        &mut self,
        module: &crate::Module,
        access: WrappedStructMatrixAccess,
    ) -> BackendResult {
        use crate::back::INDENT;

        const STRUCT_ARGUMENT_VARIABLE_NAME: &str = "obj";
        const VECTOR_ARGUMENT_VARIABLE_NAME: &str = "vec";
        const MATRIX_INDEX_ARGUMENT_VARIABLE_NAME: &str = "mat_idx";

        // Write function return type and name
        write!(self.out, "void ")?;
        self.write_wrapped_struct_matrix_set_vec_function_name(access)?;

        // Write function parameters
        write!(self.out, "(")?;
        let struct_name = &self.names[&NameKey::Type(access.ty)];
        write!(self.out, "{struct_name} {STRUCT_ARGUMENT_VARIABLE_NAME}, ")?;
        let member = match module.types[access.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => &members[access.index as usize],
            _ => unreachable!(),
        };
        let vec_ty = match module.types[member.ty].inner {
            crate::TypeInner::Matrix { rows, width, .. } => crate::TypeInner::Vector {
                size: rows,
                scalar: crate::Scalar::float(width),
            },
            _ => unreachable!(),
        };
        self.write_value_type(module, &vec_ty)?;
        write!(
            self.out,
            " {VECTOR_ARGUMENT_VARIABLE_NAME}, uint {MATRIX_INDEX_ARGUMENT_VARIABLE_NAME}"
        )?;

        // Write function body
        writeln!(self.out, ") {{")?;

        writeln!(
            self.out,
            "{INDENT}switch({MATRIX_INDEX_ARGUMENT_VARIABLE_NAME}) {{"
        )?;

        let field_name = &self.names[&NameKey::StructMember(access.ty, access.index)];

        match module.types[member.ty].inner {
            crate::TypeInner::Matrix { columns, .. } => {
                for i in 0..columns as u8 {
                    writeln!(
                        self.out,
                        "{INDENT}case {i}: {{ {STRUCT_ARGUMENT_VARIABLE_NAME}.{field_name}_{i} = {VECTOR_ARGUMENT_VARIABLE_NAME}; break; }}"
                    )?;
                }
            }
            _ => unreachable!(),
        }

        writeln!(self.out, "{INDENT}}}")?;

        // End of function body
        writeln!(self.out, "}}")?;
        // Write extra new line
        writeln!(self.out)?;

        Ok(())
    }

    pub(super) fn write_wrapped_struct_matrix_set_scalar_function_name(
        &mut self,
        access: WrappedStructMatrixAccess,
    ) -> BackendResult {
        let name = &self.names[&NameKey::Type(access.ty)];
        let field_name = &self.names[&NameKey::StructMember(access.ty, access.index)];
        write!(self.out, "SetMatScalar{field_name}On{name}")?;
        Ok(())
    }

    /// Writes a function used to set a float on a matCx2 from within a structure.
    pub(super) fn write_wrapped_struct_matrix_set_scalar_function(
        &mut self,
        module: &crate::Module,
        access: WrappedStructMatrixAccess,
    ) -> BackendResult {
        use crate::back::INDENT;

        const STRUCT_ARGUMENT_VARIABLE_NAME: &str = "obj";
        const SCALAR_ARGUMENT_VARIABLE_NAME: &str = "scalar";
        const MATRIX_INDEX_ARGUMENT_VARIABLE_NAME: &str = "mat_idx";
        const VECTOR_INDEX_ARGUMENT_VARIABLE_NAME: &str = "vec_idx";

        // Write function return type and name
        write!(self.out, "void ")?;
        self.write_wrapped_struct_matrix_set_scalar_function_name(access)?;

        // Write function parameters
        write!(self.out, "(")?;
        let struct_name = &self.names[&NameKey::Type(access.ty)];
        write!(self.out, "{struct_name} {STRUCT_ARGUMENT_VARIABLE_NAME}, ")?;
        let member = match module.types[access.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => &members[access.index as usize],
            _ => unreachable!(),
        };
        let scalar_ty = match module.types[member.ty].inner {
            crate::TypeInner::Matrix { width, .. } => {
                crate::TypeInner::Scalar(crate::Scalar::float(width))
            }
            _ => unreachable!(),
        };
        self.write_value_type(module, &scalar_ty)?;
        write!(
            self.out,
            " {SCALAR_ARGUMENT_VARIABLE_NAME}, uint {MATRIX_INDEX_ARGUMENT_VARIABLE_NAME}, uint {VECTOR_INDEX_ARGUMENT_VARIABLE_NAME}"
        )?;

        // Write function body
        writeln!(self.out, ") {{")?;

        writeln!(
            self.out,
            "{INDENT}switch({MATRIX_INDEX_ARGUMENT_VARIABLE_NAME}) {{"
        )?;

        let field_name = &self.names[&NameKey::StructMember(access.ty, access.index)];

        match module.types[member.ty].inner {
            crate::TypeInner::Matrix { columns, .. } => {
                for i in 0..columns as u8 {
                    writeln!(
                        self.out,
                        "{INDENT}case {i}: {{ {STRUCT_ARGUMENT_VARIABLE_NAME}.{field_name}_{i}[{VECTOR_INDEX_ARGUMENT_VARIABLE_NAME}] = {SCALAR_ARGUMENT_VARIABLE_NAME}; break; }}"
                    )?;
                }
            }
            _ => unreachable!(),
        }

        writeln!(self.out, "{INDENT}}}")?;

        // End of function body
        writeln!(self.out, "}}")?;
        // Write extra new line
        writeln!(self.out)?;

        Ok(())
    }

    /// Write functions to create special types.
    pub(super) fn write_special_functions(&mut self, module: &crate::Module) -> BackendResult {
        for (type_key, struct_ty) in module.special_types.predeclared_types.iter() {
            match type_key {
                &crate::PredeclaredType::ModfResult { size, width }
                | &crate::PredeclaredType::FrexpResult { size, width } => {
                    let arg_type_name_owner;
                    let arg_type_name = if let Some(size) = size {
                        arg_type_name_owner = format!(
                            "{}{}",
                            if width == 8 { "double" } else { "float" },
                            size as u8
                        );
                        &arg_type_name_owner
                    } else if width == 8 {
                        "double"
                    } else {
                        "float"
                    };

                    let (defined_func_name, called_func_name, second_field_name, sign_multiplier) =
                        if matches!(type_key, &crate::PredeclaredType::ModfResult { .. }) {
                            (super::writer::MODF_FUNCTION, "modf", "whole", "")
                        } else {
                            (
                                super::writer::FREXP_FUNCTION,
                                "frexp",
                                "exp_",
                                "sign(arg) * ",
                            )
                        };

                    let struct_name = &self.names[&NameKey::Type(*struct_ty)];

                    writeln!(
                        self.out,
                        "{struct_name} {defined_func_name}({arg_type_name} arg) {{
    {arg_type_name} other;
    {struct_name} result;
    result.fract = {sign_multiplier}{called_func_name}(arg, other);
    result.{second_field_name} = other;
    return result;
}}"
                    )?;
                    writeln!(self.out)?;
                }
                &crate::PredeclaredType::AtomicCompareExchangeWeakResult { .. } => {}
            }
        }

        Ok(())
    }

    /// Helper function that writes compose wrapped functions
    pub(super) fn write_wrapped_compose_functions(
        &mut self,
        module: &crate::Module,
        expressions: &crate::Arena<crate::Expression>,
    ) -> BackendResult {
        for (handle, _) in expressions.iter() {
            if let crate::Expression::Compose { ty, .. } = expressions[handle] {
                match module.types[ty].inner {
                    crate::TypeInner::Struct { .. } | crate::TypeInner::Array { .. } => {
                        let constructor = WrappedConstructor { ty };
                        if self.wrapped.constructors.insert(constructor) {
                            self.write_wrapped_constructor_function(module, constructor)?;
                        }
                    }
                    _ => {}
                };
            }
        }
        Ok(())
    }

    /// Helper function that writes various wrapped functions
    pub(super) fn write_wrapped_functions(
        &mut self,
        module: &crate::Module,
        func_ctx: &FunctionCtx,
    ) -> BackendResult {
        self.write_wrapped_compose_functions(module, func_ctx.expressions)?;

        for (handle, _) in func_ctx.expressions.iter() {
            match func_ctx.expressions[handle] {
                crate::Expression::ArrayLength(expr) => {
                    let global_expr = match func_ctx.expressions[expr] {
                        crate::Expression::GlobalVariable(_) => expr,
                        crate::Expression::AccessIndex { base, index: _ } => base,
                        ref other => unreachable!("Array length of {:?}", other),
                    };
                    let global_var = match func_ctx.expressions[global_expr] {
                        crate::Expression::GlobalVariable(var_handle) => {
                            &module.global_variables[var_handle]
                        }
                        ref other => unreachable!("Array length of base {:?}", other),
                    };
                    let storage_access = match global_var.space {
                        crate::AddressSpace::Storage { access } => access,
                        _ => crate::StorageAccess::default(),
                    };
                    let wal = WrappedArrayLength {
                        writable: storage_access.contains(crate::StorageAccess::STORE),
                    };

                    if self.wrapped.array_lengths.insert(wal) {
                        self.write_wrapped_array_length_function(wal)?;
                    }
                }
                crate::Expression::ImageQuery { image, query } => {
                    let wiq = match *func_ctx.resolve_type(image, &module.types) {
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

                    if self.wrapped.image_queries.insert(wiq) {
                        self.write_wrapped_image_query_function(module, wiq, handle, func_ctx)?;
                    }
                }
                // Write `WrappedConstructor` for structs that are loaded from `AddressSpace::Storage`
                // since they will later be used by the fn `write_storage_load`
                crate::Expression::Load { pointer } => {
                    let pointer_space = func_ctx
                        .resolve_type(pointer, &module.types)
                        .pointer_space();

                    if let Some(crate::AddressSpace::Storage { .. }) = pointer_space {
                        if let Some(ty) = func_ctx.info[handle].ty.handle() {
                            write_wrapped_constructor(self, ty, module)?;
                        }
                    }

                    fn write_wrapped_constructor<W: Write>(
                        writer: &mut super::Writer<'_, W>,
                        ty: Handle<crate::Type>,
                        module: &crate::Module,
                    ) -> BackendResult {
                        match module.types[ty].inner {
                            crate::TypeInner::Struct { ref members, .. } => {
                                for member in members {
                                    write_wrapped_constructor(writer, member.ty, module)?;
                                }

                                let constructor = WrappedConstructor { ty };
                                if writer.wrapped.constructors.insert(constructor) {
                                    writer
                                        .write_wrapped_constructor_function(module, constructor)?;
                                }
                            }
                            crate::TypeInner::Array { base, .. } => {
                                write_wrapped_constructor(writer, base, module)?;

                                let constructor = WrappedConstructor { ty };
                                if writer.wrapped.constructors.insert(constructor) {
                                    writer
                                        .write_wrapped_constructor_function(module, constructor)?;
                                }
                            }
                            _ => {}
                        };

                        Ok(())
                    }
                }
                // We treat matrices of the form `matCx2` as a sequence of C `vec2`s
                // (see top level module docs for details).
                //
                // The functions injected here are required to get the matrix accesses working.
                crate::Expression::AccessIndex { base, index } => {
                    let base_ty_res = &func_ctx.info[base].ty;
                    let mut resolved = base_ty_res.inner_with(&module.types);
                    let base_ty_handle = match *resolved {
                        crate::TypeInner::Pointer { base, .. } => {
                            resolved = &module.types[base].inner;
                            Some(base)
                        }
                        _ => base_ty_res.handle(),
                    };
                    if let crate::TypeInner::Struct { ref members, .. } = *resolved {
                        let member = &members[index as usize];

                        match module.types[member.ty].inner {
                            crate::TypeInner::Matrix {
                                rows: crate::VectorSize::Bi,
                                ..
                            } if member.binding.is_none() => {
                                let ty = base_ty_handle.unwrap();
                                let access = WrappedStructMatrixAccess { ty, index };

                                if self.wrapped.struct_matrix_access.insert(access) {
                                    self.write_wrapped_struct_matrix_get_function(module, access)?;
                                    self.write_wrapped_struct_matrix_set_function(module, access)?;
                                    self.write_wrapped_struct_matrix_set_vec_function(
                                        module, access,
                                    )?;
                                    self.write_wrapped_struct_matrix_set_scalar_function(
                                        module, access,
                                    )?;
                                }
                            }
                            _ => {}
                        }
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
        mip_level: Option<Handle<crate::Expression>>,
        module: &crate::Module,
        func_ctx: &FunctionCtx,
    ) -> BackendResult {
        // HLSL expects the array index to be merged with the coordinate
        let extra = array_index.is_some() as usize + (mip_level.is_some()) as usize;
        if extra == 0 {
            self.write_expr(module, coordinate, func_ctx)?;
        } else {
            let num_coords = match *func_ctx.resolve_type(coordinate, &module.types) {
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
            if let Some(expr) = mip_level {
                write!(self.out, ", ")?;
                self.write_expr(module, expr, func_ctx)?;
            }
            write!(self.out, ")")?;
        }
        Ok(())
    }

    pub(super) fn write_mat_cx2_typedef_and_functions(
        &mut self,
        WrappedMatCx2 { columns }: WrappedMatCx2,
    ) -> BackendResult {
        use crate::back::INDENT;

        // typedef
        write!(self.out, "typedef struct {{ ")?;
        for i in 0..columns as u8 {
            write!(self.out, "float2 _{i}; ")?;
        }
        writeln!(self.out, "}} __mat{}x2;", columns as u8)?;

        // __get_col_of_mat
        writeln!(
            self.out,
            "float2 __get_col_of_mat{}x2(__mat{}x2 mat, uint idx) {{",
            columns as u8, columns as u8
        )?;
        writeln!(self.out, "{INDENT}switch(idx) {{")?;
        for i in 0..columns as u8 {
            writeln!(self.out, "{INDENT}case {i}: {{ return mat._{i}; }}")?;
        }
        writeln!(self.out, "{INDENT}default: {{ return (float2)0; }}")?;
        writeln!(self.out, "{INDENT}}}")?;
        writeln!(self.out, "}}")?;

        // __set_col_of_mat
        writeln!(
            self.out,
            "void __set_col_of_mat{}x2(__mat{}x2 mat, uint idx, float2 value) {{",
            columns as u8, columns as u8
        )?;
        writeln!(self.out, "{INDENT}switch(idx) {{")?;
        for i in 0..columns as u8 {
            writeln!(self.out, "{INDENT}case {i}: {{ mat._{i} = value; break; }}")?;
        }
        writeln!(self.out, "{INDENT}}}")?;
        writeln!(self.out, "}}")?;

        // __set_el_of_mat
        writeln!(
            self.out,
            "void __set_el_of_mat{}x2(__mat{}x2 mat, uint idx, uint vec_idx, float value) {{",
            columns as u8, columns as u8
        )?;
        writeln!(self.out, "{INDENT}switch(idx) {{")?;
        for i in 0..columns as u8 {
            writeln!(
                self.out,
                "{INDENT}case {i}: {{ mat._{i}[vec_idx] = value; break; }}"
            )?;
        }
        writeln!(self.out, "{INDENT}}}")?;
        writeln!(self.out, "}}")?;

        writeln!(self.out)?;

        Ok(())
    }

    pub(super) fn write_all_mat_cx2_typedefs_and_functions(
        &mut self,
        module: &crate::Module,
    ) -> BackendResult {
        for (handle, _) in module.global_variables.iter() {
            let global = &module.global_variables[handle];

            if global.space == crate::AddressSpace::Uniform {
                if let Some(super::writer::MatrixType {
                    columns,
                    rows: crate::VectorSize::Bi,
                    width: 4,
                }) = super::writer::get_inner_matrix_data(module, global.ty)
                {
                    let entry = WrappedMatCx2 { columns };
                    if self.wrapped.mat_cx2s.insert(entry) {
                        self.write_mat_cx2_typedef_and_functions(entry)?;
                    }
                }
            }
        }

        for (_, ty) in module.types.iter() {
            if let crate::TypeInner::Struct { ref members, .. } = ty.inner {
                for member in members.iter() {
                    if let crate::TypeInner::Array { .. } = module.types[member.ty].inner {
                        if let Some(super::writer::MatrixType {
                            columns,
                            rows: crate::VectorSize::Bi,
                            width: 4,
                        }) = super::writer::get_inner_matrix_data(module, member.ty)
                        {
                            let entry = WrappedMatCx2 { columns };
                            if self.wrapped.mat_cx2s.insert(entry) {
                                self.write_mat_cx2_typedef_and_functions(entry)?;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
