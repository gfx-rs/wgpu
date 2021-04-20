// TODO: temp
#![allow(dead_code)]
use super::Error;
use crate::{
    back::wgsl::keywords::RESERVED,
    proc::{EntryPointIndex, TypeResolution},
    valid::{FunctionInfo, ModuleInfo},
    Arena, BinaryOperator, Binding, Constant, Expression, Function, GlobalVariable, Handle,
    ImageClass, ImageDimension, Module, ScalarKind, ShaderStage, Statement, StorageFormat,
    StructLevel, Type, TypeInner,
};
use crate::{
    proc::{NameKey, Namer},
    StructMember,
};
use crate::{FastHashMap, VectorSize};
use std::fmt::Write;

const INDENT: &str = "    ";

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

/// WGSL attribute
/// https://gpuweb.github.io/gpuweb/wgsl/#attributes
enum Attribute {
    Binding(u32),
    Block,
    BuiltIn(crate::BuiltIn),
    Group(u32),
    Location(u32),
    Stage(ShaderStage),
    WorkGroupSize([u32; 3]),
}

/// Stores the current function type (either a regular function or an entry point)
///
/// Also stores data needed to identify it (handle for a regular function or index for an entry point)
// TODO: copy-paste from glsl-out
enum FunctionType {
    /// A regular function and it's handle
    Function(Handle<Function>),
    /// A entry point and it's index
    EntryPoint(EntryPointIndex),
}

/// Helper structure that stores data needed when writing the function
// TODO: copy-paste from glsl-out
struct FunctionCtx<'a> {
    /// The current function type being written
    ty: FunctionType,
    /// Analysis about the function
    info: &'a FunctionInfo,
    /// The expression arena of the current function being written
    expressions: &'a Arena<Expression>,
}

pub struct Writer<W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    namer: Namer,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            namer: Namer::default(),
        }
    }

    pub fn write(&mut self, module: &Module, info: &ModuleInfo) -> BackendResult {
        self.names.clear();
        self.namer.reset(module, RESERVED, &mut self.names);

        // Write all constants
        for (_, constant) in module.constants.iter() {
            if constant.name.is_some() {
                self.write_constant(&constant, true)?;
            }
        }

        // Write all globals
        for (_, global) in module.global_variables.iter() {
            if global.name.is_some() {
                self.write_global(&module, &global)?;
            }
        }

        // Write all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct {
                level, ref members, ..
            } = ty.inner
            {
                let name = &self.names[&NameKey::Type(handle)].clone();
                let block = level == StructLevel::Root;
                self.write_struct(module, name, block, members)?;
                writeln!(self.out)?;
            }
        }

        for (index, ep) in module.entry_points.iter().enumerate() {
            let attributes = match ep.stage {
                ShaderStage::Vertex | ShaderStage::Fragment => vec![Attribute::Stage(ep.stage)],
                ShaderStage::Compute => vec![
                    Attribute::Stage(ShaderStage::Compute),
                    Attribute::WorkGroupSize(ep.workgroup_size),
                ],
            };

            self.write_attributes(&attributes)?;
            // Add a newline after attribute
            writeln!(self.out)?;

            let func_ctx = FunctionCtx {
                ty: FunctionType::EntryPoint(index as u16),
                info: &info.get_entry_point(index),
                expressions: &ep.function.expressions,
            };
            self.write_function(&module, &ep.function, &func_ctx)?;
            writeln!(self.out)?;
        }

        // Add a newline at the end of file
        writeln!(self.out)?;

        Ok(())
    }

    /// Helper method used to write structs
    /// https://gpuweb.github.io/gpuweb/wgsl/#functions
    ///
    /// # Notes
    /// Ends in a newline
    fn write_function(
        &mut self,
        module: &Module,
        func: &Function,
        func_ctx: &FunctionCtx<'_>,
    ) -> BackendResult {
        // TODO: unnamed function?
        write!(self.out, "fn {}(", func.name.as_ref().unwrap())?;

        // Write function arguments
        // TODO: another function type
        if let FunctionType::EntryPoint(ep_index) = func_ctx.ty {
            for (index, arg) in func.arguments.iter().enumerate() {
                // Write argument attribute if a binding is present
                if let Some(ref binding) = arg.binding {
                    self.write_attributes(&[map_binding_to_attribute(binding)])?;
                    write!(self.out, " ")?;
                }
                // Write argument name
                write!(
                    self.out,
                    "{}: ",
                    &self.names[&NameKey::EntryPointArgument(ep_index, index as u32)]
                )?;
                // Write argument type
                self.write_type(module, arg.ty)?;
                if index < func.arguments.len() - 1 {
                    // Add a separator between args
                    write!(self.out, ", ")?;
                }
            }
            write!(self.out, ")")?;
        }

        // Write function return type
        if let Some(ref result) = func.result {
            if let Some(ref binding) = result.binding {
                write!(self.out, " -> ")?;
                self.write_attributes(&[map_binding_to_attribute(binding)])?;
                write!(self.out, " ")?;
                self.write_type(module, result.ty)?;
                // Extra space only for readability
                write!(self.out, " ")?;
            } else {
                let struct_name = &self.names[&NameKey::Type(result.ty)].clone();
                write!(self.out, " -> {} ", struct_name)?;
            }
        }

        write!(self.out, "{{")?;
        writeln!(self.out)?;

        // Write the function body (statement list)
        for sta in func.body.iter() {
            // The indentation should always be 1 when writing the function body
            self.write_stmt(&module, sta, &func_ctx, 1)?;
        }

        writeln!(self.out, "}}")?;
        Ok(())
    }

    /// Helper method to write a attribute
    ///
    /// # Notes
    /// Adds no leading or trailing whitespace
    fn write_attributes(&mut self, attributes: &[Attribute]) -> BackendResult {
        write!(self.out, "[[")?;
        for (index, attribute) in attributes.iter().enumerate() {
            match *attribute {
                Attribute::Block => {
                    write!(self.out, "block")?;
                }
                Attribute::Location(id) => write!(self.out, "location({})", id)?,
                Attribute::BuiltIn(builtin_attrib) => {
                    let builtin_str = builtin_str(builtin_attrib);
                    if let Some(builtin) = builtin_str {
                        write!(self.out, "builtin({})", builtin)?
                    } else {
                        log::warn!("Unsupported builtin attribute: {:?}", builtin_attrib);
                    }
                }
                Attribute::Stage(shader_stage) => match shader_stage {
                    ShaderStage::Vertex => write!(self.out, "stage(vertex)")?,
                    ShaderStage::Fragment => write!(self.out, "stage(fragment)")?,
                    ShaderStage::Compute => write!(self.out, "stage(compute)")?,
                },
                Attribute::WorkGroupSize(size) => {
                    write!(
                        self.out,
                        "{}",
                        format!("workgroup_size({}, {}, {})", size[0], size[1], size[2])
                    )?;
                }
                Attribute::Binding(id) => write!(self.out, "binding({})", id)?,
                Attribute::Group(id) => write!(self.out, "group({})", id)?,
            };
            if index < attributes.len() - 1 {
                // Add a separator between args
                write!(self.out, ", ")?;
            }
        }
        write!(self.out, "]]")?;

        Ok(())
    }

    /// Helper method used to write structs
    ///
    /// # Notes
    /// Ends in a newline
    fn write_struct(
        &mut self,
        module: &Module,
        name: &str,
        block: bool,
        members: &[StructMember],
    ) -> BackendResult {
        if block {
            self.write_attributes(&[Attribute::Block])?;
            writeln!(self.out)?;
        }
        write!(self.out, "struct {} {{", name)?;
        writeln!(self.out)?;
        for (_, member) in members.iter().enumerate() {
            // The indentation is only for readability
            write!(self.out, "{}", INDENT)?;
            if let Some(ref binding) = member.binding {
                self.write_attributes(&[map_binding_to_attribute(binding)])?;
                write!(self.out, " ")?;
            }
            // Write struct member name and type
            write!(self.out, "{}: ", member.name.as_ref().unwrap())?;
            self.write_type(module, member.ty)?;
            write!(self.out, ";")?;
            writeln!(self.out)?;
        }
        write!(self.out, "}};")?;

        writeln!(self.out)?;

        Ok(())
    }

    /// Helper method used to write non image/sampler types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_type(&mut self, module: &Module, ty: Handle<Type>) -> BackendResult {
        let inner = &module.types[ty].inner;
        match *inner {
            TypeInner::Struct { .. } => {
                // Get the struct name
                let name = &self.names[&NameKey::Type(ty)];
                write!(self.out, "{}", name)?;
                return Ok(());
            }
            ref other => self.write_value_type(module, other)?,
        }

        Ok(())
    }

    /// Helper method used to write value types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_value_type(&mut self, _module: &Module, inner: &TypeInner) -> BackendResult {
        match *inner {
            TypeInner::Vector { size, kind, .. } => write!(
                self.out,
                "{}",
                format!("vec{}<{}>", vector_size_str(size), scalar_kind_str(kind),)
            )?,
            TypeInner::Sampler { comparison: false } => {
                write!(self.out, "sampler")?;
            }
            TypeInner::Sampler { comparison: true } => {
                write!(self.out, "sampler_comparison")?;
            }
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                // More about texture types: https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
                let dim_str = image_dimension_str(dim);
                let arrayed_str = if arrayed { "_array" } else { "" };
                let (class_str, multisampled_str, scalar_str) = match class {
                    ImageClass::Sampled { kind, multi } => (
                        "",
                        if multi { "multisampled" } else { "" },
                        format!("<{}>", scalar_kind_str(kind)),
                    ),
                    ImageClass::Depth => ("depth", "", String::from("")),
                    ImageClass::Storage(storage_format) => (
                        "storage",
                        "",
                        format!("<{}>", storage_format_str(storage_format)),
                    ),
                };
                let ty_str = format!(
                    "texture_{}{}{}{}{}",
                    class_str, multisampled_str, dim_str, arrayed_str, scalar_str
                );
                write!(self.out, "{}", ty_str)?;
            }
            _ => todo!("write_value_type {:?}", inner),
        }

        Ok(())
    }
    /// Helper method used to write statements
    ///
    /// # Notes
    /// Always adds a newline
    fn write_stmt(
        &mut self,
        module: &Module,
        stmt: &Statement,
        func_ctx: &FunctionCtx<'_>,
        indent: usize,
    ) -> BackendResult {
        match *stmt {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let min_ref_count = func_ctx.expressions[handle].bake_ref_count();
                    if min_ref_count <= func_ctx.info[handle].ref_count {
                        match func_ctx.info[handle].ty {
                            TypeResolution::Handle(ty_handle) => {
                                write!(self.out, "{}", INDENT.repeat(indent))?;
                                self.write_type(module, ty_handle)?
                            }
                            TypeResolution::Value(ref _inner) => {
                                //TODO:
                                //write!(self.out, "{}", INDENT.repeat(indent))?;
                                //self.write_value_type(module, inner)?
                            }
                        }
                    }
                }
            }
            // TODO: copy-paste from glsl-out
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                write!(self.out, "if (")?;
                self.write_expr(module, condition, func_ctx)?;
                writeln!(self.out, ") {{")?;

                for sta in accept {
                    // Increase indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, indent + 1)?;
                }

                // If there are no statements in the reject block we skip writing it
                // This is only for readability
                if !reject.is_empty() {
                    writeln!(self.out, "{}}} else {{", INDENT.repeat(indent))?;

                    for sta in reject {
                        // Increase indentation to help with readability
                        self.write_stmt(module, sta, func_ctx, indent + 1)?;
                    }
                }

                writeln!(self.out, "{}}}", INDENT.repeat(indent))?
            }
            Statement::Return { value } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                if let Some(return_value) = value {
                    write!(self.out, "return ")?;
                    self.write_expr(module, return_value, &func_ctx)?;
                    writeln!(self.out, ";")?;
                } else {
                    writeln!(self.out, "return;")?;
                }
            }
            // TODO: copy-paste from glsl-out
            Statement::Kill => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                writeln!(self.out, "discard;")?
            }
            _ => todo!("write_stmt {:?}", stmt),
        }

        Ok(())
    }

    /// Helper method to write expressions
    ///
    /// # Notes
    /// Doesn't add any newlines or leading/trailing spaces
    fn write_expr(
        &mut self,
        module: &Module,
        expr: Handle<Expression>,
        func_ctx: &FunctionCtx<'_>,
    ) -> BackendResult {
        let expression = &func_ctx.expressions[expr];
        match *expression {
            Expression::Constant(constant) => {
                self.write_constant(&module.constants[constant], false)?
            }
            Expression::Compose { ty, ref components } => {
                self.write_type(&module, ty)?;
                write!(self.out, "(")?;
                self.write_slice(components, |this, _, arg| {
                    this.write_expr(&module, *arg, func_ctx)
                })?;
                write!(self.out, ")")?
            }
            Expression::FunctionArgument(pos) => {
                let name_key = match func_ctx.ty {
                    FunctionType::Function(handle) => NameKey::FunctionArgument(handle, pos),
                    FunctionType::EntryPoint(ep_index) => {
                        NameKey::EntryPointArgument(ep_index, pos)
                    }
                };
                let name = &self.names[&name_key];
                write!(self.out, "{}", name)?;
            }
            Expression::Binary { op, left, right } => {
                self.write_expr(module, left, func_ctx)?;

                // TODO: glsl-out copy-paste
                write!(
                    self.out,
                    " {} ",
                    match op {
                        BinaryOperator::Add => "+",
                        BinaryOperator::Subtract => "-",
                        BinaryOperator::Multiply => "*",
                        BinaryOperator::Divide => "/",
                        BinaryOperator::Modulo => "%",
                        BinaryOperator::Equal => "==",
                        BinaryOperator::NotEqual => "!=",
                        BinaryOperator::Less => "<",
                        BinaryOperator::LessEqual => "<=",
                        BinaryOperator::Greater => ">",
                        BinaryOperator::GreaterEqual => ">=",
                        BinaryOperator::And => "&",
                        BinaryOperator::ExclusiveOr => "^",
                        BinaryOperator::InclusiveOr => "|",
                        BinaryOperator::LogicalAnd => "&&",
                        BinaryOperator::LogicalOr => "||",
                        BinaryOperator::ShiftLeft => "<<",
                        BinaryOperator::ShiftRight => ">>",
                    }
                )?;

                self.write_expr(module, right, func_ctx)?;
            }
            // TODO: copy-paste from glsl-out
            Expression::Access { base, index } => {
                self.write_expr(module, base, func_ctx)?;
                write!(self.out, "[")?;
                self.write_expr(module, index, func_ctx)?;
                write!(self.out, "]")?
            }
            // TODO: copy-paste from glsl-out
            Expression::AccessIndex { base, index } => {
                self.write_expr(module, base, func_ctx)?;

                let base_ty_res = &func_ctx.info[base].ty;
                let mut resolved = base_ty_res.inner_with(&module.types);
                let base_ty_handle = match *resolved {
                    TypeInner::Pointer { base, class: _ } => {
                        resolved = &module.types[base].inner;
                        Some(base)
                    }
                    _ => base_ty_res.handle(),
                };

                match *resolved {
                    TypeInner::Vector { .. }
                    | TypeInner::Matrix { .. }
                    | TypeInner::Array { .. }
                    | TypeInner::ValuePointer { .. } => write!(self.out, "[{}]", index)?,
                    TypeInner::Struct { .. } => {
                        // This will never panic in case the type is a `Struct`, this is not true
                        // for other types so we can only check while inside this match arm
                        let ty = base_ty_handle.unwrap();

                        write!(
                            self.out,
                            ".{}",
                            &self.names[&NameKey::StructMember(ty, index)]
                        )?
                    }
                    ref other => return Err(Error::Custom(format!("Cannot index {:?}", other))),
                }
            }
            Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index: _,
                offset: _,
                level,
                depth_ref: _,
            } => {
                // TODO: other texture functions
                // TODO: comments
                let fun_name = match level {
                    crate::SampleLevel::Auto => "textureSample",
                    _ => todo!("expression_imagesample_level {:?}", level),
                };
                write!(self.out, "{}(", fun_name)?;
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, sampler, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;
                write!(self.out, ")")?;
            }
            // TODO: copy-paste from msl-out
            Expression::GlobalVariable(handle) => {
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(self.out, "{}", name)?;
            }
            _ => todo!("write_expr {:?}", expression),
        }

        Ok(())
    }

    /// Helper method that writes a list of comma separated `T` with a writer function `F`
    ///
    /// The writer function `F` receives a mutable reference to `self` that if needed won't cause
    /// borrow checker issues (using for example a closure with `self` will cause issues), the
    /// second argument is the 0 based index of the element on the list, and the last element is
    /// a reference to the element `T` being written
    ///
    /// # Notes
    /// - Adds no newlines or leading/trailing whitespace
    /// - The last element won't have a trailing `,`
    // TODO: copy-paste from glsl-out
    fn write_slice<T, F: FnMut(&mut Self, u32, &T) -> BackendResult>(
        &mut self,
        data: &[T],
        mut f: F,
    ) -> BackendResult {
        // Loop trough `data` invoking `f` for each element
        for (i, item) in data.iter().enumerate() {
            f(self, i as u32, item)?;

            // Only write a comma if isn't the last element
            if i != data.len().saturating_sub(1) {
                // The leading space is for readability only
                write!(self.out, ", ")?;
            }
        }

        Ok(())
    }

    /// Helper method used to write global variables
    fn write_global(&mut self, module: &Module, global: &GlobalVariable) -> BackendResult {
        if let Some(ref binding) = global.binding {
            self.write_attributes(&[
                Attribute::Group(binding.group),
                Attribute::Binding(binding.binding),
            ])?;
            write!(self.out, " ")?;
        }

        if let Some(ref name) = global.name {
            // First write only global name
            write!(self.out, "var {}: ", name)?;
            // Write global type
            self.write_type(module, global.ty)?;
            // End with semicolon and extra newline for readability
            writeln!(self.out, ";")?;
            writeln!(self.out)?;
        }

        Ok(())
    }

    /// Helper method used to write constants
    ///
    /// # Notes
    /// Adds newlines for global constants
    fn write_constant(&mut self, constant: &Constant, global: bool) -> BackendResult {
        match constant.inner {
            crate::ConstantInner::Scalar {
                width: _,
                ref value,
            } => {
                if let Some(ref name) = constant.name {
                    if global {
                        // First write only constant name
                        write!(self.out, "let {}: ", name)?;
                        // Next write constant type and value
                        match *value {
                            crate::ScalarValue::Sint(value) => {
                                write!(self.out, "i32 = {}", value)?;
                            }
                            crate::ScalarValue::Uint(value) => {
                                write!(self.out, "u32 = {}", value)?;
                            }
                            crate::ScalarValue::Float(value) => {
                                write!(self.out, "f32 = {}", value)?;
                            }
                            crate::ScalarValue::Bool(value) => {
                                write!(self.out, "bool = {}", value)?;
                            }
                        };
                        // End with semicolon and extra newline for readability
                        writeln!(self.out, ";")?;
                        writeln!(self.out)?;
                    } else {
                        write!(self.out, "{}", name)?;
                    }
                } else {
                    match *value {
                        crate::ScalarValue::Sint(value) => {
                            write!(self.out, "{}", value)?;
                        }
                        crate::ScalarValue::Uint(value) => {
                            write!(self.out, "{}", value)?;
                        }
                        // TODO: fix float
                        crate::ScalarValue::Float(value) => {
                            write!(self.out, "{:.1}", value)?;
                        }
                        crate::ScalarValue::Bool(value) => {
                            write!(self.out, "{}", value)?;
                        }
                    };
                }
            }
            _ => todo!("write_constant {:?}", constant.inner),
        }

        Ok(())
    }

    pub fn finish(self) -> W {
        self.out
    }
}

fn builtin_str(built_in: crate::BuiltIn) -> Option<&'static str> {
    use crate::BuiltIn;
    match built_in {
        BuiltIn::VertexIndex => Some("vertex_index"),
        BuiltIn::InstanceIndex => Some("instance_index"),
        BuiltIn::Position => Some("position"),
        BuiltIn::FrontFacing => Some("front_facing"),
        BuiltIn::FragDepth => Some("frag_depth"),
        BuiltIn::LocalInvocationId => Some("local_invocation_id"),
        BuiltIn::LocalInvocationIndex => Some("local_invocation_index"),
        BuiltIn::GlobalInvocationId => Some("global_invocation_id"),
        BuiltIn::WorkGroupId => Some("workgroup_id"),
        BuiltIn::WorkGroupSize => Some("workgroup_size"),
        BuiltIn::SampleIndex => Some("sample_index"),
        BuiltIn::SampleMask => Some("sample_mask"),
        _ => None,
    }
}

//TODO: copy-paste from msl-out
fn vector_size_str(size: VectorSize) -> &'static str {
    match size {
        VectorSize::Bi => "2",
        VectorSize::Tri => "3",
        VectorSize::Quad => "4",
    }
}

//TODO: copy-paste from msl-out
fn image_dimension_str(dim: ImageDimension) -> &'static str {
    match dim {
        ImageDimension::D1 => "1d",
        ImageDimension::D2 => "2d",
        ImageDimension::D3 => "3d",
        ImageDimension::Cube => "cube",
    }
}

fn scalar_kind_str(kind: ScalarKind) -> &'static str {
    match kind {
        crate::ScalarKind::Float => "f32",
        crate::ScalarKind::Sint => "i32",
        crate::ScalarKind::Uint => "u32",
        crate::ScalarKind::Bool => "bool",
    }
}

fn storage_format_str(format: StorageFormat) -> &'static str {
    match format {
        StorageFormat::R8Unorm => "r8unorm",
        StorageFormat::R8Snorm => "r8snorm",
        StorageFormat::R8Uint => "r8uint",
        StorageFormat::R8Sint => "r8sint",
        StorageFormat::R16Uint => "r16uint",
        StorageFormat::R16Sint => "r16sint",
        StorageFormat::R16Float => "r16float",
        StorageFormat::Rg8Unorm => "rg8unorm",
        StorageFormat::Rg8Snorm => "rg8snorm",
        StorageFormat::Rg8Uint => "rg8uint",
        StorageFormat::Rg8Sint => "rg8sint",
        StorageFormat::R32Uint => "r32uint",
        StorageFormat::R32Sint => "r32sint",
        StorageFormat::R32Float => "r32float",
        StorageFormat::Rg16Uint => "rg16uint",
        StorageFormat::Rg16Sint => "rg16sint",
        StorageFormat::Rg16Float => "rg16float",
        StorageFormat::Rgba8Unorm => "rgba8unorm",
        StorageFormat::Rgba8Snorm => "rgba8snorm",
        StorageFormat::Rgba8Uint => "rgba8uint",
        StorageFormat::Rgba8Sint => "rgba8sint",
        StorageFormat::Rgb10a2Unorm => "rgb10a2unorm",
        StorageFormat::Rg11b10Float => "rg11b10float",
        StorageFormat::Rg32Uint => "rg32uint",
        StorageFormat::Rg32Sint => "rg32sint",
        StorageFormat::Rg32Float => "rg32float",
        StorageFormat::Rgba16Uint => "rgba16uint",
        StorageFormat::Rgba16Sint => "rgba16sint",
        StorageFormat::Rgba16Float => "rgba16float",
        StorageFormat::Rgba32Uint => "rgba32uint",
        StorageFormat::Rgba32Sint => "rgba32sint",
        StorageFormat::Rgba32Float => "rgba32float",
    }
}

fn map_binding_to_attribute(binding: &Binding) -> Attribute {
    match *binding {
        Binding::BuiltIn(built_in) => Attribute::BuiltIn(built_in),
        //TODO: Interpolation
        Binding::Location { location, .. } => Attribute::Location(location),
    }
}
