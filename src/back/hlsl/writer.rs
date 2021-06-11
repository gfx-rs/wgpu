//TODO: temp
#![allow(dead_code)]
use super::{Error, ShaderModel};
use crate::back::hlsl::keywords::RESERVED;
use crate::proc::{EntryPointIndex, NameKey, Namer};
use crate::valid::{FunctionInfo, ModuleInfo};
use crate::{
    Arena, Bytes, Constant, Expression, FastHashMap, Function, Handle, ImageDimension,
    LocalVariable, Module, ScalarKind, ShaderStage, Statement, Type, TypeInner,
};
use std::fmt::Write;

const INDENT: &str = "    ";

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

/// Stores the current function type (either a regular function or an entry point)
///
/// Also stores data needed to identify it (handle for a regular function or index for an entry point)
// TODO: copy-paste from glsl-out, wgsl-out
enum FunctionType {
    /// A regular function and it's handle
    Function(Handle<Function>),
    /// A entry point and it's index
    EntryPoint(EntryPointIndex),
}

/// Helper structure that stores data needed when writing the function
// TODO: copy-paste from glsl-out, wgsl-out
struct FunctionCtx<'a> {
    /// The current function type being written
    ty: FunctionType,
    /// Analysis about the function
    info: &'a FunctionInfo,
    /// The expression arena of the current function being written
    expressions: &'a Arena<Expression>,
    /// Map of expressions that have associated variable names
    named_expressions: &'a crate::NamedExpressions,
}

impl<'a> FunctionCtx<'_> {
    /// Helper method that generates a [`NameKey`](crate::proc::NameKey) for a local in the current function
    fn name_key(&self, local: Handle<LocalVariable>) -> NameKey {
        match self.ty {
            FunctionType::Function(handle) => NameKey::FunctionLocal(handle, local),
            FunctionType::EntryPoint(idx) => NameKey::EntryPointLocal(idx, local),
        }
    }
}

pub struct Writer<W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    namer: Namer,
    shader_model: ShaderModel,
    named_expressions: crate::NamedExpressions,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W, shader_model: ShaderModel) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            namer: Namer::default(),
            shader_model,
            named_expressions: crate::NamedExpressions::default(),
        }
    }

    fn reset(&mut self, module: &Module) {
        self.names.clear();
        self.namer.reset(module, RESERVED, &[], &mut self.names);
        self.named_expressions.clear();
    }

    pub fn write(&mut self, module: &Module, info: &ModuleInfo) -> BackendResult {
        if self.shader_model < ShaderModel(50) {
            return Err(Error::UnsupportedShaderModel(self.shader_model));
        }

        self.reset(module);

        // Write all entry points
        for (index, ep) in module.entry_points.iter().enumerate() {
            let func_ctx = FunctionCtx {
                ty: FunctionType::EntryPoint(index as u16),
                info: info.get_entry_point(index),
                expressions: &ep.function.expressions,
                named_expressions: &ep.function.named_expressions,
            };
            if ep.stage == ShaderStage::Compute {
                // HLSL is calling workgroup size, num threads
                let num_threads = ep.workgroup_size;
                writeln!(
                    self.out,
                    "[numthreads({}, {}, {})]",
                    num_threads[0], num_threads[1], num_threads[2]
                )?;
            }
            self.write_function(module, &ep.function, &func_ctx)?;

            if index < module.entry_points.len() - 1 {
                writeln!(self.out)?;
            }
        }

        Ok(())
    }

    /// Helper method used to write non image/sampler types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_type(&mut self, module: &Module, ty: Handle<Type>) -> BackendResult {
        let inner = &module.types[ty].inner;
        match *inner {
            TypeInner::Struct { .. } => write!(self.out, "{}", self.names[&NameKey::Type(ty)])?,
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
            TypeInner::Scalar { kind, width } => {
                write!(self.out, "{}", scalar_kind_str(kind, width)?)?;
            }
            _ => {
                return Err(Error::Unimplemented(format!(
                    "write_value_type {:?}",
                    inner
                )))
            }
        }

        Ok(())
    }

    /// Helper method used to write structs
    /// # Notes
    /// Ends in a newline
    fn write_function(
        &mut self,
        module: &Module,
        func: &Function,
        func_ctx: &FunctionCtx<'_>,
    ) -> BackendResult {
        // Function Declaration Syntax - https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-function-syntax
        if let Some(ref result) = func.result {
            self.write_type(module, result.ty)?;
        } else {
            write!(self.out, "void")?;
        }

        let func_name = match func_ctx.ty {
            FunctionType::EntryPoint(index) => self.names[&NameKey::EntryPoint(index)].clone(),
            FunctionType::Function(handle) => self.names[&NameKey::Function(handle)].clone(),
        };

        // Write function name
        write!(self.out, " {}(", func_name)?;

        // Write function arguments
        for (index, arg) in func.arguments.iter().enumerate() {
            // Write argument type
            self.write_type(module, arg.ty)?;

            let argument_name = match func_ctx.ty {
                FunctionType::Function(handle) => {
                    self.names[&NameKey::FunctionArgument(handle, index as u32)].clone()
                }
                FunctionType::EntryPoint(ep_index) => {
                    self.names[&NameKey::EntryPointArgument(ep_index, index as u32)].clone()
                }
            };

            // Write argument name. Space is important.
            write!(self.out, " {}", argument_name)?;
            if index < func.arguments.len() - 1 {
                // Add a separator between args
                write!(self.out, ", ")?;
            }
        }
        // Ends of arguments
        write!(self.out, ")")?;

        // Write semantic if it present
        if let Some(ref result) = func.result {
            if let Some(ref binding) = result.binding {
                match *binding {
                    crate::Binding::BuiltIn(builtin) => {
                        write!(self.out, " : {}", builtin_str(builtin)?)?
                    }
                    // TODO: Is this reachable ?
                    crate::Binding::Location { .. } => {
                        return Err(Error::Unimplemented(format!(
                            "write_function semantic {:?}",
                            binding
                        )))
                    }
                }
            }
        }

        // Function body start
        writeln!(self.out)?;
        writeln!(self.out, "{{")?;
        // Write function local variables
        for (handle, local) in func.local_variables.iter() {
            // Write indentation (only for readability)
            write!(self.out, "{}", INDENT)?;

            // Write the local name
            // The leading space is important
            write!(self.out, "var {}: ", self.names[&func_ctx.name_key(handle)])?;

            // Write the local type
            self.write_type(module, local.ty)?;

            // Write the local initializer if needed
            if let Some(init) = local.init {
                // Put the equal signal only if there's a initializer
                // The leading and trailing spaces aren't needed but help with readability
                write!(self.out, " = ")?;

                // Write the constant
                // `write_constant` adds no trailing or leading space/newline
                self.write_constant(module, init)?;
            }

            // Finish the local with `;` and add a newline (only for readability)
            writeln!(self.out, ";")?
        }

        if !func.local_variables.is_empty() {
            writeln!(self.out)?;
        }

        // Write the function body (statement list)
        for sta in func.body.iter() {
            // The indentation should always be 1 when writing the function body
            self.write_stmt(module, sta, func_ctx, 1)?;
        }

        writeln!(self.out, "}}")?;

        self.named_expressions.clear();

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
            Statement::Return { value } => {
                write!(self.out, "{}return", INDENT.repeat(indent))?;
                if let Some(return_value) = value {
                    // The leading space is important
                    write!(self.out, " ")?;
                    self.write_expr(module, return_value, func_ctx)?;
                }
                writeln!(self.out, ";")?;
            }
            _ => return Err(Error::Unimplemented(format!("write_stmt {:?}", stmt))),
        }

        Ok(())
    }

    /// Helper method to write expressions
    ///
    /// # Notes
    /// Doesn't add any newlines or leading/trailing spaces
    fn write_expr(
        &mut self,
        _module: &Module,
        expr: Handle<Expression>,
        func_ctx: &FunctionCtx<'_>,
    ) -> BackendResult {
        if let Some(name) = self.named_expressions.get(&expr) {
            write!(self.out, "{}", name)?;
            return Ok(());
        }

        let expression = &func_ctx.expressions[expr];

        #[allow(clippy::match_single_binding)]
        match *expression {
            _ => return Err(Error::Unimplemented(format!("write_expr {:?}", expression))),
        }
    }

    /// Helper method used to write constants
    ///
    /// # Notes
    /// Doesn't add any newlines or leading/trailing spaces
    fn write_constant(&mut self, module: &Module, handle: Handle<Constant>) -> BackendResult {
        let constant = &module.constants[handle];
        #[allow(clippy::match_single_binding)]
        match constant.inner {
            _ => {
                return Err(Error::Unimplemented(format!(
                    "write_constant {:?}",
                    constant
                )))
            }
        }
    }

    pub fn finish(self) -> W {
        self.out
    }
}

fn image_dimension_str(dim: ImageDimension) -> &'static str {
    match dim {
        ImageDimension::D1 => "1D",
        ImageDimension::D2 => "2D",
        ImageDimension::D3 => "3D",
        ImageDimension::Cube => "Cube",
    }
}

fn builtin_str(built_in: crate::BuiltIn) -> Result<&'static str, Error> {
    use crate::BuiltIn;
    match built_in {
        BuiltIn::Position => Ok("SV_Position"),
        BuiltIn::PointSize => Err(Error::UnsupportedBuiltIn(built_in)),
        _ => Err(Error::Unimplemented(format!("builtin_str {:?}", built_in))),
    }
}

/// Helper function that returns scalar related strings
/// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-scalar
fn scalar_kind_str(kind: ScalarKind, width: Bytes) -> Result<&'static str, Error> {
    match kind {
        ScalarKind::Sint => Ok("int"),
        ScalarKind::Uint => Ok("uint"),
        ScalarKind::Float => match width {
            2 => Ok("half"),
            4 => Ok("float"),
            8 => Ok("double"),
            _ => Err(Error::UnsupportedScalar(kind, width)),
        },
        ScalarKind::Bool => Ok("bool"),
    }
}
