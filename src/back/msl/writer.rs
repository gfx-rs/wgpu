use super::{
    keywords::RESERVED, sampler as sm, Error, LocationMode, Options, PipelineOptions,
    TranslationInfo,
};
use crate::{
    arena::{Arena, Handle},
    back::vector_size_str,
    proc::{EntryPointIndex, NameKey, Namer, TypeResolution},
    valid::{Capabilities, FunctionInfo, GlobalUse, ModuleInfo},
    FastHashMap,
};
use bit_set::BitSet;
use std::{
    fmt::{Display, Error as FmtError, Formatter, Write},
    iter,
};

const NAMESPACE: &str = "metal";
const INDENT: &str = "    ";
const BAKE_PREFIX: &str = "_e";
const WRAPPED_ARRAY_FIELD: &str = "inner";

#[derive(Clone)]
struct Level(usize);
impl Level {
    fn next(&self) -> Self {
        Level(self.0 + 1)
    }
}
impl Display for Level {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        (0..self.0).try_for_each(|_| formatter.write_str(INDENT))
    }
}

struct TypeContext<'a> {
    handle: Handle<crate::Type>,
    arena: &'a Arena<crate::Type>,
    names: &'a FastHashMap<NameKey, String>,
    access: crate::StorageAccess,
    first_time: bool,
}

impl<'a> Display for TypeContext<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        let ty = &self.arena[self.handle];
        if ty.needs_alias() && !self.first_time {
            let name = &self.names[&NameKey::Type(self.handle)];
            return write!(out, "{}", name);
        }

        match ty.inner {
            // work around Metal toolchain bug with `uint` typedef
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Uint,
                ..
            } => {
                write!(out, "metal::uint")
            }
            crate::TypeInner::Scalar { kind, .. } => {
                write!(out, "{}", scalar_kind_string(kind))
            }
            crate::TypeInner::Vector { size, kind, .. } => {
                write!(
                    out,
                    "{}::{}{}",
                    NAMESPACE,
                    scalar_kind_string(kind),
                    vector_size_str(size),
                )
            }
            crate::TypeInner::Matrix { columns, rows, .. } => {
                write!(
                    out,
                    "{}::{}{}x{}",
                    NAMESPACE,
                    scalar_kind_string(crate::ScalarKind::Float),
                    vector_size_str(columns),
                    vector_size_str(rows),
                )
            }
            crate::TypeInner::Pointer { base, class } => {
                let sub = Self {
                    handle: base,
                    first_time: false,
                    ..*self
                };
                let class_name = match class.get_name(self.access) {
                    Some(name) => name,
                    None => return Ok(()),
                };
                write!(out, "{} {}&", class_name, sub)
            }
            crate::TypeInner::ValuePointer {
                size: None,
                kind,
                width: _,
                class,
            } => {
                let class_name = match class.get_name(self.access) {
                    Some(name) => name,
                    None => return Ok(()),
                };
                write!(out, "{} {}&", class_name, scalar_kind_string(kind),)
            }
            crate::TypeInner::ValuePointer {
                size: Some(size),
                kind,
                width: _,
                class,
            } => {
                let class_name = match class.get_name(self.access) {
                    Some(name) => name,
                    None => return Ok(()),
                };
                write!(
                    out,
                    "{} {}::{}{}&",
                    class_name,
                    NAMESPACE,
                    scalar_kind_string(kind),
                    vector_size_str(size),
                )
            }
            crate::TypeInner::Array { base, .. } => {
                let sub = Self {
                    handle: base,
                    first_time: false,
                    ..*self
                };
                // Array lengths go at the end of the type definition,
                // so just print the element type here.
                write!(out, "{}", sub)
            }
            crate::TypeInner::Struct { .. } => unreachable!(),
            crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                let dim_str = match dim {
                    crate::ImageDimension::D1 => "1d",
                    crate::ImageDimension::D2 => "2d",
                    crate::ImageDimension::D3 => "3d",
                    crate::ImageDimension::Cube => "cube",
                };
                let (texture_str, msaa_str, kind, access) = match class {
                    crate::ImageClass::Sampled { kind, multi } => {
                        let (msaa_str, access) = if multi {
                            ("_ms", "read")
                        } else {
                            ("", "sample")
                        };
                        ("texture", msaa_str, kind, access)
                    }
                    crate::ImageClass::Depth => ("depth", "", crate::ScalarKind::Float, "sample"),
                    crate::ImageClass::Storage(format) => {
                        let access = if self
                            .access
                            .contains(crate::StorageAccess::LOAD | crate::StorageAccess::STORE)
                        {
                            "read_write"
                        } else if self.access.contains(crate::StorageAccess::STORE) {
                            "write"
                        } else if self.access.contains(crate::StorageAccess::LOAD) {
                            "read"
                        } else {
                            unreachable!("module is not valid")
                        };
                        ("texture", "", format.into(), access)
                    }
                };
                let base_name = scalar_kind_string(kind);
                let array_str = if arrayed { "_array" } else { "" };
                write!(
                    out,
                    "{}::{}{}{}{}<{}, {}::access::{}>",
                    NAMESPACE,
                    texture_str,
                    dim_str,
                    msaa_str,
                    array_str,
                    base_name,
                    NAMESPACE,
                    access,
                )
            }
            crate::TypeInner::Sampler { comparison: _ } => {
                write!(out, "{}::sampler", NAMESPACE)
            }
        }
    }
}

struct TypedGlobalVariable<'a> {
    module: &'a crate::Module,
    names: &'a FastHashMap<NameKey, String>,
    handle: Handle<crate::GlobalVariable>,
    usage: GlobalUse,
    reference: bool,
}

impl<'a> TypedGlobalVariable<'a> {
    fn try_fmt<W: Write>(&self, out: &mut W) -> Result<(), Error> {
        let var = &self.module.global_variables[self.handle];
        let name = &self.names[&NameKey::GlobalVariable(self.handle)];
        let ty_name = TypeContext {
            handle: var.ty,
            arena: &self.module.types,
            names: self.names,
            access: var.storage_access,
            first_time: false,
        };

        let (space, access, reference) = match var.class.get_name(var.storage_access) {
            Some(space) if self.reference => {
                let access = match var.class {
                    crate::StorageClass::Private | crate::StorageClass::WorkGroup
                        if !self.usage.contains(GlobalUse::WRITE) =>
                    {
                        "const"
                    }
                    _ => "",
                };
                (space, access, "&")
            }
            _ => ("", "", ""),
        };

        Ok(write!(
            out,
            "{}{}{}{}{}{} {}",
            space,
            if space.is_empty() { "" } else { " " },
            ty_name,
            if access.is_empty() { "" } else { " " },
            access,
            reference,
            name,
        )?)
    }
}

struct ConstantContext<'a> {
    handle: Handle<crate::Constant>,
    arena: &'a Arena<crate::Constant>,
    names: &'a FastHashMap<NameKey, String>,
    first_time: bool,
}

impl<'a> Display for ConstantContext<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        let con = &self.arena[self.handle];
        if con.needs_alias() && !self.first_time {
            let name = &self.names[&NameKey::Constant(self.handle)];
            return write!(out, "{}", name);
        }

        match con.inner {
            crate::ConstantInner::Scalar { value, width: _ } => match value {
                crate::ScalarValue::Sint(value) => {
                    write!(out, "{}", value)
                }
                crate::ScalarValue::Uint(value) => {
                    write!(out, "{}u", value)
                }
                crate::ScalarValue::Float(value) => {
                    if value.is_infinite() {
                        let sign = if value.is_sign_negative() { "-" } else { "" };
                        write!(out, "{}INFINITY", sign)
                    } else if value.is_nan() {
                        write!(out, "NAN")
                    } else {
                        let suffix = if value.fract() == 0.0 { ".0" } else { "" };

                        write!(out, "{}{}", value, suffix)
                    }
                }
                crate::ScalarValue::Bool(value) => {
                    write!(out, "{}", value)
                }
            },
            crate::ConstantInner::Composite { .. } => unreachable!("should be aliased"),
        }
    }
}

pub struct Writer<W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    named_expressions: BitSet,
    namer: Namer,
    runtime_sized_buffers: FastHashMap<Handle<crate::GlobalVariable>, usize>,
    #[cfg(test)]
    put_expression_stack_pointers: crate::FastHashSet<*const ()>,
    #[cfg(test)]
    put_block_stack_pointers: crate::FastHashSet<*const ()>,
}

fn scalar_kind_string(kind: crate::ScalarKind) -> &'static str {
    match kind {
        crate::ScalarKind::Float => "float",
        crate::ScalarKind::Sint => "int",
        crate::ScalarKind::Uint => "uint",
        crate::ScalarKind::Bool => "bool",
    }
}

const COMPONENTS: &[char] = &['x', 'y', 'z', 'w'];

fn separate(need_separator: bool) -> &'static str {
    if need_separator {
        ","
    } else {
        ""
    }
}

fn should_pack_struct_member(
    members: &[crate::StructMember],
    span: u32,
    index: usize,
    module: &crate::Module,
) -> Option<crate::ScalarKind> {
    let member = &members[index];
    let ty_inner = &module.types[member.ty].inner;

    let last_offset = member.offset + ty_inner.span(&module.constants);
    let next_offset = match members.get(index + 1) {
        Some(next) => next.offset,
        None => span,
    };
    let is_tight = next_offset == last_offset;

    match *ty_inner {
        crate::TypeInner::Vector {
            size: crate::VectorSize::Tri,
            width: 4,
            kind,
        } if member.offset & 0xF != 0 || is_tight => Some(kind),
        _ => None,
    }
}

fn needs_array_length(ty: Handle<crate::Type>, arena: &Arena<crate::Type>) -> bool {
    if let crate::TypeInner::Struct { ref members, .. } = arena[ty].inner {
        if let Some(member) = members.last() {
            if let crate::TypeInner::Array {
                size: crate::ArraySize::Dynamic,
                ..
            } = arena[member.ty].inner
            {
                return true;
            }
        }
    }
    false
}

impl crate::StorageClass {
    /// Returns true for storage classes, for which the global
    /// variables are passed in function arguments.
    /// These arguments need to be passed through any functions
    /// called from the entry point.
    fn needs_pass_through(&self) -> bool {
        match *self {
            crate::StorageClass::Uniform
            | crate::StorageClass::Storage
            | crate::StorageClass::Private
            | crate::StorageClass::PushConstant
            | crate::StorageClass::Handle => true,
            _ => false,
        }
    }

    fn get_name(&self, access: crate::StorageAccess) -> Option<&'static str> {
        match *self {
            Self::Handle => None,
            Self::Uniform | Self::PushConstant => Some("constant"),
            //TODO: should still be "constant" for read-only buffers
            Self::Storage => Some(if access.contains(crate::StorageAccess::STORE) {
                "device"
            } else {
                "constant"
            }),
            Self::Private | Self::Function => Some("thread"),
            Self::WorkGroup => Some("threadgroup"),
        }
    }
}

impl crate::Type {
    // Returns `true` if we need to emit an alias for this type.
    fn needs_alias(&self) -> bool {
        use crate::TypeInner as Ti;
        match self.inner {
            // value types are concise enough, we only alias them if they are named
            Ti::Scalar { .. }
            | Ti::Vector { .. }
            | Ti::Matrix { .. }
            | Ti::Pointer { .. }
            | Ti::ValuePointer { .. } => self.name.is_some(),
            // composite types are better to be aliased, regardless of the name
            Ti::Struct { .. } | Ti::Array { .. } => true,
            // handle types may be different, depending on the global var access, so we always inline them
            Ti::Image { .. } | Ti::Sampler { .. } => false,
        }
    }
}

impl crate::Constant {
    // Returns `true` if we need to emit an alias for this constant.
    fn needs_alias(&self) -> bool {
        match self.inner {
            crate::ConstantInner::Scalar { .. } => self.name.is_some(),
            crate::ConstantInner::Composite { .. } => true,
        }
    }
}

enum FunctionOrigin {
    Handle(Handle<crate::Function>),
    EntryPoint(EntryPointIndex),
}

struct ExpressionContext<'a> {
    function: &'a crate::Function,
    origin: FunctionOrigin,
    info: &'a FunctionInfo,
    module: &'a crate::Module,
    pipeline_options: &'a PipelineOptions,
}

impl<'a> ExpressionContext<'a> {
    fn resolve_type(&self, handle: Handle<crate::Expression>) -> &'a crate::TypeInner {
        self.info[handle].ty.inner_with(&self.module.types)
    }
}

struct StatementContext<'a> {
    expression: ExpressionContext<'a>,
    mod_info: &'a ModuleInfo,
    result_struct: Option<&'a str>,
}

impl<W: Write> Writer<W> {
    /// Creates a new `Writer` instance.
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            named_expressions: BitSet::new(),
            namer: Namer::default(),
            runtime_sized_buffers: FastHashMap::default(),
            #[cfg(test)]
            put_expression_stack_pointers: Default::default(),
            #[cfg(test)]
            put_block_stack_pointers: Default::default(),
        }
    }

    /// Finishes writing and returns the output.
    pub fn finish(self) -> W {
        self.out
    }

    fn put_call_parameters(
        &mut self,
        parameters: impl Iterator<Item = Handle<crate::Expression>>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        write!(self.out, "(")?;
        for (i, handle) in parameters.enumerate() {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            self.put_expression(handle, context, true)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_image_query(
        &mut self,
        image: Handle<crate::Expression>,
        query: &str,
        level: Option<Handle<crate::Expression>>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        self.put_expression(image, context, false)?;
        write!(self.out, ".get_{}(", query)?;
        if let Some(expr) = level {
            self.put_expression(expr, context, true)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_image_size_query(
        &mut self,
        image: Handle<crate::Expression>,
        level: Option<Handle<crate::Expression>>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        //Note: MSL only has separate width/height/depth queries,
        // so compose the result of them.
        let dim = match *context.resolve_type(image) {
            crate::TypeInner::Image { dim, .. } => dim,
            ref other => unreachable!("Unexpected type {:?}", other),
        };
        match dim {
            crate::ImageDimension::D1 => {
                write!(self.out, "int(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ")")?;
            }
            crate::ImageDimension::D2 => {
                write!(self.out, "int2(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "height", level, context)?;
                write!(self.out, ")")?;
            }
            crate::ImageDimension::D3 => {
                write!(self.out, "int3(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "height", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "depth", level, context)?;
                write!(self.out, ")")?;
            }
            crate::ImageDimension::Cube => {
                write!(self.out, "int3(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ")")?;
            }
        }
        Ok(())
    }

    fn put_storage_image_coordinate(
        &mut self,
        expr: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        // coordinates in IR are int, but Metal expects uint
        let size_str = match *context.info[expr].ty.inner_with(&context.module.types) {
            crate::TypeInner::Scalar { .. } => "",
            crate::TypeInner::Vector { size, .. } => vector_size_str(size),
            _ => return Err(Error::Validation),
        };
        write!(self.out, "{}::uint{}(", NAMESPACE, size_str)?;
        self.put_expression(expr, context, true)?;
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_image_sample_level(
        &mut self,
        image: Handle<crate::Expression>,
        level: crate::SampleLevel,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        let has_levels = match *context.resolve_type(image) {
            crate::TypeInner::Image {
                dim: crate::ImageDimension::D1,
                ..
            } => false,
            _ => true,
        };
        match level {
            crate::SampleLevel::Auto => {}
            crate::SampleLevel::Zero => {
                //TODO: do we support Zero on `Sampled` image classes?
            }
            _ if !has_levels => {
                log::warn!("1D image can't be sampled with level {:?}", level);
            }
            crate::SampleLevel::Exact(h) => {
                write!(self.out, ", {}::level(", NAMESPACE)?;
                self.put_expression(h, context, true)?;
                write!(self.out, ")")?;
            }
            crate::SampleLevel::Bias(h) => {
                write!(self.out, ", {}::bias(", NAMESPACE)?;
                self.put_expression(h, context, true)?;
                write!(self.out, ")")?;
            }
            crate::SampleLevel::Gradient { x, y } => {
                write!(self.out, ", {}::gradient(", NAMESPACE)?;
                self.put_expression(x, context, true)?;
                write!(self.out, ", ")?;
                self.put_expression(y, context, true)?;
                write!(self.out, ")")?;
            }
        }
        Ok(())
    }

    fn put_compose(
        &mut self,
        ty: Handle<crate::Type>,
        components: &[Handle<crate::Expression>],
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        match context.module.types[ty].inner {
            crate::TypeInner::Scalar { width: 4, kind } if components.len() == 1 => {
                write!(self.out, "{}", scalar_kind_string(kind))?;
                self.put_call_parameters(components.iter().cloned(), context)?;
            }
            crate::TypeInner::Vector { size, kind, .. } => {
                write!(
                    self.out,
                    "{}::{}{}",
                    NAMESPACE,
                    scalar_kind_string(kind),
                    vector_size_str(size)
                )?;
                self.put_call_parameters(components.iter().cloned(), context)?;
            }
            crate::TypeInner::Matrix { columns, rows, .. } => {
                let kind = crate::ScalarKind::Float;
                write!(
                    self.out,
                    "{}::{}{}x{}",
                    NAMESPACE,
                    scalar_kind_string(kind),
                    vector_size_str(columns),
                    vector_size_str(rows)
                )?;
                self.put_call_parameters(components.iter().cloned(), context)?;
            }
            crate::TypeInner::Array { .. } | crate::TypeInner::Struct { .. } => {
                write!(self.out, "{} {{", &self.names[&NameKey::Type(ty)])?;
                for (i, &component) in components.iter().enumerate() {
                    if i != 0 {
                        write!(self.out, ", ")?;
                    }
                    self.put_expression(component, context, true)?;
                }
                write!(self.out, "}}")?;
            }
            _ => return Err(Error::UnsupportedCompose(ty)),
        }
        Ok(())
    }

    fn put_array_length(
        &mut self,
        expr: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        let handle = match context.function.expressions[expr] {
            crate::Expression::AccessIndex { base, .. } => {
                match context.function.expressions[base] {
                    crate::Expression::GlobalVariable(handle) => handle,
                    _ => return Err(Error::Validation),
                }
            }
            _ => return Err(Error::Validation),
        };

        let global = &context.module.global_variables[handle];
        if let crate::TypeInner::Struct { ref members, .. } = context.module.types[global.ty].inner
        {
            if let Some(&crate::StructMember {
                offset,
                ty: array_ty,
                ..
            }) = members.last()
            {
                let (span, stride) = match context.module.types[array_ty].inner {
                    crate::TypeInner::Array { base, stride, .. } => (
                        context.module.types[base]
                            .inner
                            .span(&context.module.constants),
                        stride,
                    ),
                    _ => return Err(Error::Validation),
                };

                let buffer_idx = self.runtime_sized_buffers[&handle];
                write!(
                    self.out,
                    "(1 + (_buffer_sizes.size{idx} - {offset} - {span}) / {stride})",
                    idx = buffer_idx,
                    offset = offset,
                    span = span,
                    stride = stride,
                )?;
                Ok(())
            } else {
                Err(Error::Validation)
            }
        } else {
            Err(Error::Validation)
        }
    }

    fn put_expression(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        context: &ExpressionContext,
        is_scoped: bool,
    ) -> Result<(), Error> {
        // Add to the set in order to track the stack size.
        #[cfg(test)]
        #[allow(trivial_casts)]
        self.put_expression_stack_pointers
            .insert(&expr_handle as *const _ as *const ());

        if self.named_expressions.contains(expr_handle.index()) {
            write!(self.out, "{}{}", BAKE_PREFIX, expr_handle.index())?;
            return Ok(());
        }

        let expression = &context.function.expressions[expr_handle];
        log::trace!("expression {:?} = {:?}", expr_handle, expression);
        match *expression {
            crate::Expression::Access { base, index } => {
                let accessing_wrapped_array =
                    match *context.info[base].ty.inner_with(&context.module.types) {
                        crate::TypeInner::Array { .. } => true,
                        crate::TypeInner::Pointer {
                            base: pointer_base, ..
                        } => match context.module.types[pointer_base].inner {
                            crate::TypeInner::Array {
                                size: crate::ArraySize::Constant(_),
                                ..
                            } => true,
                            _ => false,
                        },
                        _ => false,
                    };

                self.put_expression(base, context, false)?;
                if accessing_wrapped_array {
                    write!(self.out, ".{}", WRAPPED_ARRAY_FIELD)?;
                }
                write!(self.out, "[")?;
                self.put_expression(index, context, true)?;
                write!(self.out, "]")?;
            }
            crate::Expression::AccessIndex { base, index } => {
                self.put_expression(base, context, false)?;
                let base_res = &context.info[base].ty;
                let mut resolved = base_res.inner_with(&context.module.types);
                let base_ty_handle = match *resolved {
                    crate::TypeInner::Pointer { base, class: _ } => {
                        resolved = &context.module.types[base].inner;
                        Some(base)
                    }
                    _ => base_res.handle(),
                };
                match *resolved {
                    crate::TypeInner::Struct { .. } => {
                        let base_ty = base_ty_handle.unwrap();
                        let name = &self.names[&NameKey::StructMember(base_ty, index)];
                        write!(self.out, ".{}", name)?;
                    }
                    crate::TypeInner::ValuePointer { .. } | crate::TypeInner::Vector { .. } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                    }
                    crate::TypeInner::Matrix { .. } => {
                        write!(self.out, "[{}]", index)?;
                    }
                    crate::TypeInner::Array { .. } => {
                        write!(self.out, "[{}]", index)?;
                    }
                    _ => {
                        // unexpected indexing, should fail validation
                    }
                }
            }
            crate::Expression::Constant(handle) => {
                let coco = ConstantContext {
                    handle,
                    arena: &context.module.constants,
                    names: &self.names,
                    first_time: false,
                };
                write!(self.out, "{}", coco)?;
            }
            crate::Expression::Splat { size, value } => {
                let scalar_kind = match *context.resolve_type(value) {
                    crate::TypeInner::Scalar { kind, .. } => kind,
                    _ => return Err(Error::Validation),
                };
                let scalar = scalar_kind_string(scalar_kind);
                let size = vector_size_str(size);

                write!(self.out, "{}::{}{}(", NAMESPACE, scalar, size)?;
                self.put_expression(value, context, true)?;
                write!(self.out, ")")?;
            }
            crate::Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                self.put_expression(vector, context, false)?;
                write!(self.out, ".")?;
                for &sc in pattern[..size as usize].iter() {
                    write!(self.out, "{}", COMPONENTS[sc as usize])?;
                }
            }
            crate::Expression::Compose { ty, ref components } => {
                self.put_compose(ty, components, context)?;
            }
            crate::Expression::FunctionArgument(index) => {
                let name_key = match context.origin {
                    FunctionOrigin::Handle(handle) => NameKey::FunctionArgument(handle, index),
                    FunctionOrigin::EntryPoint(ep_index) => {
                        NameKey::EntryPointArgument(ep_index, index)
                    }
                };
                let name = &self.names[&name_key];
                write!(self.out, "{}", name)?;
            }
            crate::Expression::GlobalVariable(handle) => {
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(self.out, "{}", name)?;
            }
            crate::Expression::LocalVariable(handle) => {
                let name_key = match context.origin {
                    FunctionOrigin::Handle(fun_handle) => {
                        NameKey::FunctionLocal(fun_handle, handle)
                    }
                    FunctionOrigin::EntryPoint(ep_index) => {
                        NameKey::EntryPointLocal(ep_index, handle)
                    }
                };
                let name = &self.names[&name_key];
                write!(self.out, "{}", name)?;
            }
            crate::Expression::Load { pointer } => {
                // Because packed vectors such as `packed_float3` cannot be directly multipied by
                // matrices, we wrap them with `float3` on load.
                let wrap_packed_vec_scalar_kind = match context.function.expressions[pointer] {
                    crate::Expression::AccessIndex { base, index } => {
                        let ty = match context.resolve_type(base) {
                            &crate::TypeInner::Pointer { base, .. } => {
                                &context.module.types[base].inner
                            }
                            // This path is unexpected and shouldn't happen, but it's easier
                            // to leave in.
                            ty => ty,
                        };
                        match *ty {
                            crate::TypeInner::Struct {
                                ref members, span, ..
                            } => should_pack_struct_member(
                                members,
                                span,
                                index as usize,
                                &context.module,
                            ),
                            _ => None,
                        }
                    }
                    _ => None,
                };

                if let Some(scalar_kind) = wrap_packed_vec_scalar_kind {
                    write!(
                        self.out,
                        "{}::{}3(",
                        NAMESPACE,
                        scalar_kind_string(scalar_kind)
                    )?;
                    self.put_expression(pointer, context, true)?;
                    write!(self.out, ")")?;
                } else {
                    // We don't do any dereferencing with `*` here as pointer arguments to functions
                    // are done by `&` references and not `*` pointers. These do not need to be
                    // dereferenced.
                    self.put_expression(pointer, context, is_scoped)?;
                }
            }
            crate::Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                let op = match depth_ref {
                    Some(_) => "sample_compare",
                    None => "sample",
                };
                self.put_expression(image, context, false)?;
                write!(self.out, ".{}(", op)?;
                self.put_expression(sampler, context, true)?;
                write!(self.out, ", ")?;
                self.put_expression(coordinate, context, true)?;
                if let Some(expr) = array_index {
                    write!(self.out, ", ")?;
                    self.put_expression(expr, context, true)?;
                }
                if let Some(dref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.put_expression(dref, context, true)?;
                }

                self.put_image_sample_level(image, level, context)?;

                if let Some(constant) = offset {
                    let coco = ConstantContext {
                        handle: constant,
                        arena: &context.module.constants,
                        names: &self.names,
                        first_time: false,
                    };
                    write!(self.out, ", {}", coco)?;
                }
                write!(self.out, ")")?;
            }
            crate::Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                index,
            } => {
                self.put_expression(image, context, false)?;
                write!(self.out, ".read(")?;
                self.put_storage_image_coordinate(coordinate, context)?;
                if let Some(expr) = array_index {
                    write!(self.out, ", ")?;
                    self.put_expression(expr, context, true)?;
                }
                if let Some(index) = index {
                    write!(self.out, ", ")?;
                    self.put_expression(index, context, true)?;
                }
                write!(self.out, ")")?;
            }
            //Note: for all the queries, the signed integers are expected,
            // so a conversion is needed.
            crate::Expression::ImageQuery { image, query } => match query {
                crate::ImageQuery::Size { level } => {
                    self.put_image_size_query(image, level, context)?;
                }
                crate::ImageQuery::NumLevels => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_num_mip_levels())")?;
                }
                crate::ImageQuery::NumLayers => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_array_size())")?;
                }
                crate::ImageQuery::NumSamples => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_num_samples())")?;
                }
            },
            crate::Expression::Unary { op, expr } => {
                let op_str = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::Not => "!",
                };
                write!(self.out, "{}", op_str)?;
                self.put_expression(expr, context, false)?;
            }
            crate::Expression::Binary { op, left, right } => {
                let op_str = crate::back::binary_operation_str(op);
                let kind = context
                    .resolve_type(left)
                    .scalar_kind()
                    .ok_or(Error::UnsupportedBinaryOp(op))?;
                if op == crate::BinaryOperator::Modulo && kind == crate::ScalarKind::Float {
                    write!(self.out, "{}::fmod(", NAMESPACE)?;
                    self.put_expression(left, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(right, context, true)?;
                    write!(self.out, ")")?;
                } else {
                    if !is_scoped {
                        write!(self.out, "(")?;
                    }
                    self.put_expression(left, context, false)?;
                    write!(self.out, " {} ", op_str)?;
                    self.put_expression(right, context, false)?;
                    if !is_scoped {
                        write!(self.out, ")")?;
                    }
                }
            }
            crate::Expression::Select {
                condition,
                accept,
                reject,
            } => match *context.resolve_type(condition) {
                crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Bool,
                    ..
                } => {
                    if !is_scoped {
                        write!(self.out, "(")?;
                    }
                    self.put_expression(condition, context, false)?;
                    write!(self.out, " ? ")?;
                    self.put_expression(accept, context, false)?;
                    write!(self.out, " : ")?;
                    self.put_expression(reject, context, false)?;
                    if !is_scoped {
                        write!(self.out, ")")?;
                    }
                }
                crate::TypeInner::Vector {
                    kind: crate::ScalarKind::Bool,
                    ..
                } => {
                    write!(self.out, "{}::select(", NAMESPACE)?;
                    self.put_expression(reject, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(accept, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(condition, context, true)?;
                    write!(self.out, ")")?;
                }
                _ => return Err(Error::Validation),
            },
            crate::Expression::Derivative { axis, expr } => {
                let op = match axis {
                    crate::DerivativeAxis::X => "dfdx",
                    crate::DerivativeAxis::Y => "dfdy",
                    crate::DerivativeAxis::Width => "fwidth",
                };
                write!(self.out, "{}::{}", NAMESPACE, op)?;
                self.put_call_parameters(iter::once(expr), context)?;
            }
            crate::Expression::Relational { fun, argument } => {
                let op = match fun {
                    crate::RelationalFunction::Any => "any",
                    crate::RelationalFunction::All => "all",
                    crate::RelationalFunction::IsNan => "isnan",
                    crate::RelationalFunction::IsInf => "isinf",
                    crate::RelationalFunction::IsFinite => "isfinite",
                    crate::RelationalFunction::IsNormal => "isnormal",
                };
                write!(self.out, "{}::{}", NAMESPACE, op)?;
                self.put_call_parameters(iter::once(argument), context)?;
            }
            crate::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
            } => {
                use crate::MathFunction as Mf;

                let scalar_argument = match *context.resolve_type(arg) {
                    crate::TypeInner::Scalar { .. } => true,
                    _ => false,
                };

                let fun_name = match fun {
                    // comparison
                    Mf::Abs => "abs",
                    Mf::Min => "min",
                    Mf::Max => "max",
                    Mf::Clamp => "clamp",
                    // trigonometry
                    Mf::Cos => "cos",
                    Mf::Cosh => "cosh",
                    Mf::Sin => "sin",
                    Mf::Sinh => "sinh",
                    Mf::Tan => "tan",
                    Mf::Tanh => "tanh",
                    Mf::Acos => "acos",
                    Mf::Asin => "asin",
                    Mf::Atan => "atan",
                    Mf::Atan2 => "atan2",
                    // decomposition
                    Mf::Ceil => "ceil",
                    Mf::Floor => "floor",
                    Mf::Round => "round",
                    Mf::Fract => "fract",
                    Mf::Trunc => "trunc",
                    Mf::Modf => "modf",
                    Mf::Frexp => "frexp",
                    Mf::Ldexp => "ldexp",
                    // exponent
                    Mf::Exp => "exp",
                    Mf::Exp2 => "exp2",
                    Mf::Log => "log",
                    Mf::Log2 => "log2",
                    Mf::Pow => "pow",
                    // geometry
                    Mf::Dot => "dot",
                    Mf::Outer => return Err(Error::UnsupportedCall(format!("{:?}", fun))),
                    Mf::Cross => "cross",
                    Mf::Distance => "distance",
                    Mf::Length if scalar_argument => "abs",
                    Mf::Length => "length",
                    Mf::Normalize => "normalize",
                    Mf::FaceForward => "faceforward",
                    Mf::Reflect => "reflect",
                    Mf::Refract => "refract",
                    // computational
                    Mf::Sign => "sign",
                    Mf::Fma => "fma",
                    Mf::Mix => "mix",
                    Mf::Step => "step",
                    Mf::SmoothStep => "smoothstep",
                    Mf::Sqrt => "sqrt",
                    Mf::InverseSqrt => "rsqrt",
                    Mf::Inverse => return Err(Error::UnsupportedCall(format!("{:?}", fun))),
                    Mf::Transpose => "transpose",
                    Mf::Determinant => "determinant",
                    // bits
                    Mf::CountOneBits => "popcount",
                    Mf::ReverseBits => "reverse_bits",
                };

                if fun == Mf::Distance && scalar_argument {
                    write!(self.out, "{}::abs(", NAMESPACE)?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, " - ")?;
                    self.put_expression(arg1.unwrap(), context, false)?;
                    write!(self.out, ")")?;
                } else {
                    write!(self.out, "{}::{}", NAMESPACE, fun_name)?;
                    self.put_call_parameters(iter::once(arg).chain(arg1).chain(arg2), context)?;
                }
            }
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                let scalar = scalar_kind_string(kind);
                let (size, width) = match *context.resolve_type(expr) {
                    crate::TypeInner::Scalar { width, .. } => ("", width),
                    crate::TypeInner::Vector { size, width, .. } => (vector_size_str(size), width),
                    _ => return Err(Error::Validation),
                };
                let op = match convert {
                    Some(w) if w == width => "static_cast",
                    Some(8) if kind == crate::ScalarKind::Float => {
                        return Err(Error::CapabilityNotSupported(Capabilities::FLOAT64))
                    }
                    Some(_) => return Err(Error::Validation),
                    None => "as_type",
                };
                write!(self.out, "{}<{}{}>(", op, scalar, size)?;
                self.put_expression(expr, context, true)?;
                write!(self.out, ")")?;
            }
            // has to be a named expression
            crate::Expression::Call(_) => unreachable!(),
            crate::Expression::ArrayLength(expr) => {
                self.put_array_length(expr, context)?;
            }
        }
        Ok(())
    }

    fn put_return_value(
        &mut self,
        level: Level,
        expr_handle: Handle<crate::Expression>,
        result_struct: Option<&str>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        match result_struct {
            Some(struct_name) => {
                let result_ty = context.function.result.as_ref().unwrap().ty;
                match context.module.types[result_ty].inner {
                    crate::TypeInner::Struct { ref members, .. } => {
                        let tmp = "_tmp";
                        write!(self.out, "{}const auto {} = ", level, tmp)?;
                        self.put_expression(expr_handle, context, true)?;
                        writeln!(self.out, ";")?;
                        write!(self.out, "{}return {} {{", level, struct_name)?;
                        let mut is_first = true;
                        for (index, member) in members.iter().enumerate() {
                            if !context.pipeline_options.allow_point_size
                                && member.binding
                                    == Some(crate::Binding::BuiltIn(crate::BuiltIn::PointSize))
                            {
                                continue;
                            }
                            if member.binding
                                == Some(crate::Binding::BuiltIn(crate::BuiltIn::CullDistance))
                            {
                                log::warn!("Ignoring CullDistance BuiltIn");
                                continue;
                            }
                            let comma = if is_first { "" } else { "," };
                            is_first = false;
                            let name = &self.names[&NameKey::StructMember(result_ty, index as u32)];
                            // HACK: we are forcefully deduplicating the expression here
                            // to convert from a wrapped struct to a raw array, e.g.
                            // `float gl_ClipDistance1 [[clip_distance]] [1];`.
                            if let crate::TypeInner::Array {
                                size: crate::ArraySize::Constant(const_handle),
                                ..
                            } = context.module.types[member.ty].inner
                            {
                                let size = context.module.constants[const_handle]
                                    .to_array_length()
                                    .unwrap();
                                write!(self.out, "{} {{", comma)?;
                                for j in 0..size {
                                    if j != 0 {
                                        write!(self.out, ",")?;
                                    }
                                    write!(
                                        self.out,
                                        "{}.{}.{}[{}]",
                                        tmp, name, WRAPPED_ARRAY_FIELD, j
                                    )?;
                                }
                                write!(self.out, "}}")?;
                            } else {
                                write!(self.out, "{} {}.{}", comma, tmp, name)?;
                            }
                        }
                    }
                    _ => {
                        write!(self.out, "{}return {} {{ ", level, struct_name)?;
                        self.put_expression(expr_handle, context, true)?;
                    }
                }
                write!(self.out, " }}")?;
            }
            None => {
                write!(self.out, "{}return ", level)?;
                self.put_expression(expr_handle, context, true)?;
            }
        }
        writeln!(self.out, ";")?;
        Ok(())
    }

    fn start_baking_expression(
        &mut self,
        handle: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        match context.info[handle].ty {
            TypeResolution::Handle(ty_handle) => {
                let ty_name = TypeContext {
                    handle: ty_handle,
                    arena: &context.module.types,
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    first_time: false,
                };
                write!(self.out, "{}", ty_name)?;
            }
            TypeResolution::Value(crate::TypeInner::Scalar { kind, .. }) => {
                write!(self.out, "{}", scalar_kind_string(kind))?;
            }
            TypeResolution::Value(crate::TypeInner::Vector { size, kind, .. }) => {
                write!(
                    self.out,
                    "{}::{}{}",
                    NAMESPACE,
                    scalar_kind_string(kind),
                    vector_size_str(size)
                )?;
            }
            TypeResolution::Value(crate::TypeInner::Matrix { columns, rows, .. }) => {
                write!(
                    self.out,
                    "{}::{}{}x{}",
                    NAMESPACE,
                    scalar_kind_string(crate::ScalarKind::Float),
                    vector_size_str(columns),
                    vector_size_str(rows),
                )?;
            }
            TypeResolution::Value(ref other) => {
                log::error!("Type {:?} isn't a known local", other);
                return Err(Error::FeatureNotImplemented("weird local type".to_string()));
            }
        }

        //TODO: figure out the naming scheme that wouldn't collide with user names.
        write!(self.out, " {}{} = ", BAKE_PREFIX, handle.index())?;
        Ok(())
    }

    fn put_block(
        &mut self,
        level: Level,
        statements: &[crate::Statement],
        context: &StatementContext,
    ) -> Result<(), Error> {
        // Add to the set in order to track the stack size.
        #[cfg(test)]
        #[allow(trivial_casts)]
        self.put_block_stack_pointers
            .insert(&level as *const _ as *const ());

        for statement in statements {
            log::trace!("statement[{}] {:?}", level.0, statement);
            match *statement {
                crate::Statement::Emit(ref range) => {
                    for handle in range.clone() {
                        let min_ref_count =
                            context.expression.function.expressions[handle].bake_ref_count();
                        if min_ref_count <= context.expression.info[handle].ref_count {
                            write!(self.out, "{}", level)?;
                            self.start_baking_expression(handle, &context.expression)?;
                            self.put_expression(handle, &context.expression, true)?;
                            writeln!(self.out, ";")?;
                            self.named_expressions.insert(handle.index());
                        }
                    }
                }
                crate::Statement::Block(ref block) => {
                    if !block.is_empty() {
                        writeln!(self.out, "{}{{", level)?;
                        self.put_block(level.next(), block, context)?;
                        writeln!(self.out, "{}}}", level)?;
                    }
                }
                crate::Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    write!(self.out, "{}if (", level)?;
                    self.put_expression(condition, &context.expression, true)?;
                    writeln!(self.out, ") {{")?;
                    self.put_block(level.next(), accept, context)?;
                    if !reject.is_empty() {
                        writeln!(self.out, "{}}} else {{", level)?;
                        self.put_block(level.next(), reject, context)?;
                    }
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    write!(self.out, "{}switch(", level)?;
                    self.put_expression(selector, &context.expression, true)?;
                    writeln!(self.out, ") {{")?;
                    let lcase = level.next();
                    for case in cases.iter() {
                        writeln!(self.out, "{}case {}: {{", lcase, case.value)?;
                        self.put_block(lcase.next(), &case.body, context)?;
                        if !case.fall_through {
                            writeln!(self.out, "{}break;", lcase.next())?;
                        }
                        writeln!(self.out, "{}}}", lcase)?;
                    }
                    writeln!(self.out, "{}default: {{", lcase)?;
                    self.put_block(lcase.next(), default, context)?;
                    writeln!(self.out, "{}}}", lcase)?;
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Loop {
                    ref body,
                    ref continuing,
                } => {
                    if !continuing.is_empty() {
                        let gate_name = self.namer.call("loop_init");
                        writeln!(self.out, "{}bool {} = true;", level, gate_name)?;
                        writeln!(self.out, "{}while(true) {{", level)?;
                        let lif = level.next();
                        writeln!(self.out, "{}if (!{}) {{", lif, gate_name)?;
                        self.put_block(lif.next(), continuing, context)?;
                        writeln!(self.out, "{}}}", lif)?;
                        writeln!(self.out, "{}{} = false;", lif, gate_name)?;
                    } else {
                        writeln!(self.out, "{}while(true) {{", level)?;
                    }
                    self.put_block(level.next(), body, context)?;
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Break => {
                    writeln!(self.out, "{}break;", level)?;
                }
                crate::Statement::Continue => {
                    writeln!(self.out, "{}continue;", level)?;
                }
                crate::Statement::Return {
                    value: Some(expr_handle),
                } => {
                    self.put_return_value(
                        level.clone(),
                        expr_handle,
                        context.result_struct,
                        &context.expression,
                    )?;
                }
                crate::Statement::Return { value: None } => {
                    writeln!(self.out, "{}return;", level)?;
                }
                crate::Statement::Kill => {
                    writeln!(self.out, "{}{}::discard_fragment();", level, NAMESPACE)?;
                }
                crate::Statement::Barrier(flags) => {
                    //Note: OR-ring bitflags requires `__HAVE_MEMFLAG_OPERATORS__`,
                    // so we try to avoid it here.
                    if flags.is_empty() {
                        writeln!(
                            self.out,
                            "{}{}::threadgroup_barrier({}::mem_flags::mem_none);",
                            level, NAMESPACE, NAMESPACE,
                        )?;
                    }
                    if flags.contains(crate::Barrier::STORAGE) {
                        writeln!(
                            self.out,
                            "{}{}::threadgroup_barrier({}::mem_flags::mem_device);",
                            level, NAMESPACE, NAMESPACE,
                        )?;
                    }
                    if flags.contains(crate::Barrier::WORK_GROUP) {
                        writeln!(
                            self.out,
                            "{}{}::threadgroup_barrier({}::mem_flags::mem_threadgroup);",
                            level, NAMESPACE, NAMESPACE,
                        )?;
                    }
                }
                crate::Statement::Store { pointer, value } => {
                    // we can't assign fixed-size arrays
                    let pointer_info = &context.expression.info[pointer];
                    let array_size =
                        match *pointer_info.ty.inner_with(&context.expression.module.types) {
                            crate::TypeInner::Pointer { base, .. } => {
                                match context.expression.module.types[base].inner {
                                    crate::TypeInner::Array {
                                        size: crate::ArraySize::Constant(ch),
                                        ..
                                    } => Some(ch),
                                    _ => None,
                                }
                            }
                            _ => None,
                        };
                    match array_size {
                        Some(const_handle) => {
                            let size = context.expression.module.constants[const_handle]
                                .to_array_length()
                                .unwrap();
                            write!(self.out, "{}for(int _i=0; _i<{}; ++_i) ", level, size)?;
                            self.put_expression(pointer, &context.expression, true)?;
                            write!(self.out, ".{}[_i] = ", WRAPPED_ARRAY_FIELD)?;
                            self.put_expression(value, &context.expression, true)?;
                            writeln!(self.out, ".{}[_i];", WRAPPED_ARRAY_FIELD)?;
                        }
                        None => {
                            write!(self.out, "{}", level)?;
                            self.put_expression(pointer, &context.expression, true)?;
                            write!(self.out, " = ")?;
                            self.put_expression(value, &context.expression, true)?;
                            writeln!(self.out, ";")?;
                        }
                    }
                }
                crate::Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    write!(self.out, "{}", level)?;
                    self.put_expression(image, &context.expression, false)?;
                    write!(self.out, ".write(")?;
                    self.put_expression(value, &context.expression, true)?;
                    write!(self.out, ", ")?;
                    self.put_storage_image_coordinate(coordinate, &context.expression)?;
                    if let Some(expr) = array_index {
                        write!(self.out, ", ")?;
                        self.put_expression(expr, &context.expression, true)?;
                    }
                    writeln!(self.out, ");")?;
                }
                crate::Statement::Call {
                    function,
                    ref arguments,
                    result,
                } => {
                    write!(self.out, "{}", level)?;
                    if let Some(expr) = result {
                        self.start_baking_expression(expr, &context.expression)?;
                        self.named_expressions.insert(expr.index());
                    }
                    let fun_name = &self.names[&NameKey::Function(function)];
                    write!(self.out, "{}(", fun_name)?;
                    // first, write down the actual arguments
                    for (i, &handle) in arguments.iter().enumerate() {
                        if i != 0 {
                            write!(self.out, ", ")?;
                        }
                        self.put_expression(handle, &context.expression, true)?;
                    }
                    // follow-up with any global resources used
                    let mut separate = !arguments.is_empty();
                    let fun_info = &context.mod_info[function];
                    let mut supports_array_length = false;
                    for (handle, var) in context.expression.module.global_variables.iter() {
                        if fun_info[handle].is_empty() {
                            continue;
                        }
                        if var.class.needs_pass_through() {
                            let name = &self.names[&NameKey::GlobalVariable(handle)];
                            if separate {
                                write!(self.out, ", ")?;
                            } else {
                                separate = true;
                            }
                            write!(self.out, "{}", name)?;
                        }
                        supports_array_length |=
                            needs_array_length(var.ty, &context.expression.module.types);
                    }
                    if supports_array_length {
                        if separate {
                            write!(self.out, ", ")?;
                        }
                        write!(self.out, "_buffer_sizes")?;
                    }

                    // done
                    writeln!(self.out, ");")?;
                }
            }
        }

        // un-emit expressions
        //TODO: take care of loop/continuing?
        for statement in statements {
            if let crate::Statement::Emit(ref range) = *statement {
                for handle in range.clone() {
                    self.named_expressions.remove(handle.index());
                }
            }
        }
        Ok(())
    }

    pub fn write(
        &mut self,
        module: &crate::Module,
        info: &ModuleInfo,
        options: &Options,
        pipeline_options: &PipelineOptions,
    ) -> Result<TranslationInfo, Error> {
        self.names.clear();
        self.namer.reset(module, RESERVED, &[], &mut self.names);
        self.runtime_sized_buffers.clear();

        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out)?;

        {
            let mut indices = vec![];
            for (handle, var) in module.global_variables.iter() {
                if needs_array_length(var.ty, &module.types) {
                    let idx = handle.index();
                    self.runtime_sized_buffers.insert(handle, idx);
                    indices.push(idx);
                }
            }

            if !indices.is_empty() {
                writeln!(self.out, "struct _mslBufferSizes {{")?;

                for idx in indices {
                    writeln!(self.out, "{}{}::uint size{};", INDENT, NAMESPACE, idx)?;
                }

                writeln!(self.out, "}};")?;
                writeln!(self.out)?;
            }
        };

        self.write_scalar_constants(module)?;
        self.write_type_defs(module)?;
        self.write_composite_constants(module)?;
        self.write_functions(module, info, options, pipeline_options)
    }

    fn write_type_defs(&mut self, module: &crate::Module) -> Result<(), Error> {
        for (handle, ty) in module.types.iter() {
            if !ty.needs_alias() {
                continue;
            }
            let name = &self.names[&NameKey::Type(handle)];
            match ty.inner {
                crate::TypeInner::Array {
                    base,
                    size,
                    stride: _,
                } => {
                    let base_name = TypeContext {
                        handle: base,
                        arena: &module.types,
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        first_time: false,
                    };

                    match size {
                        crate::ArraySize::Constant(const_handle) => {
                            let coco = ConstantContext {
                                handle: const_handle,
                                arena: &module.constants,
                                names: &self.names,
                                first_time: false,
                            };

                            writeln!(self.out, "struct {} {{", name)?;
                            writeln!(
                                self.out,
                                "{}{} {}[{}];",
                                INDENT, base_name, WRAPPED_ARRAY_FIELD, coco
                            )?;
                            writeln!(self.out, "}};")?;
                        }
                        crate::ArraySize::Dynamic => {
                            writeln!(self.out, "typedef {} {}[1];", base_name, name)?;
                        }
                    }
                }
                crate::TypeInner::Struct {
                    ref members, span, ..
                } => {
                    writeln!(self.out, "struct {} {{", name)?;
                    let mut last_offset = 0;
                    for (index, member) in members.iter().enumerate() {
                        // quick and dirty way to figure out if we need this...
                        if member.binding.is_none() && member.offset > last_offset {
                            let pad = member.offset - last_offset;
                            //TODO: adjust the struct initializers
                            writeln!(self.out, "{}char _pad{}[{}];", INDENT, index, pad)?;
                        }
                        let ty_inner = &module.types[member.ty].inner;
                        last_offset = member.offset + ty_inner.span(&module.constants);

                        let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];

                        // If the member should be packed (as is the case for a misaligned vec3) issue a packed vector
                        match should_pack_struct_member(members, span, index, module) {
                            Some(kind) => {
                                writeln!(
                                    self.out,
                                    "{}packed_{}3 {};",
                                    INDENT,
                                    scalar_kind_string(kind),
                                    member_name
                                )?;
                            }
                            None => {
                                let base_name = TypeContext {
                                    handle: member.ty,
                                    arena: &module.types,
                                    names: &self.names,
                                    access: crate::StorageAccess::empty(),
                                    first_time: false,
                                };
                                writeln!(self.out, "{}{} {};", INDENT, base_name, member_name)?;

                                // for 3-component vectors, add one component
                                if let crate::TypeInner::Vector {
                                    size: crate::VectorSize::Tri,
                                    kind: _,
                                    width,
                                } = *ty_inner
                                {
                                    last_offset += width as u32;
                                }
                            }
                        }
                    }
                    writeln!(self.out, "}};")?;
                }
                _ => {
                    let ty_name = TypeContext {
                        handle,
                        arena: &module.types,
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        first_time: true,
                    };
                    writeln!(self.out, "typedef {} {};", ty_name, name)?;
                }
            }
        }
        Ok(())
    }

    fn write_scalar_constants(&mut self, module: &crate::Module) -> Result<(), Error> {
        for (handle, constant) in module.constants.iter() {
            match constant.inner {
                crate::ConstantInner::Scalar {
                    width: _,
                    ref value,
                } if constant.name.is_some() => {
                    debug_assert!(constant.needs_alias());
                    write!(self.out, "constexpr constant ")?;
                    match *value {
                        crate::ScalarValue::Sint(_) => {
                            write!(self.out, "int")?;
                        }
                        crate::ScalarValue::Uint(_) => {
                            write!(self.out, "unsigned")?;
                        }
                        crate::ScalarValue::Float(_) => {
                            write!(self.out, "float")?;
                        }
                        crate::ScalarValue::Bool(_) => {
                            write!(self.out, "bool")?;
                        }
                    }
                    let name = &self.names[&NameKey::Constant(handle)];
                    let coco = ConstantContext {
                        handle,
                        arena: &module.constants,
                        names: &self.names,
                        first_time: true,
                    };
                    writeln!(self.out, " {} = {};", name, coco)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn write_composite_constants(&mut self, module: &crate::Module) -> Result<(), Error> {
        for (handle, constant) in module.constants.iter() {
            match constant.inner {
                crate::ConstantInner::Scalar { .. } => {}
                crate::ConstantInner::Composite { ty, ref components } => {
                    debug_assert!(constant.needs_alias());
                    let name = &self.names[&NameKey::Constant(handle)];
                    let ty_name = TypeContext {
                        handle: ty,
                        arena: &module.types,
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        first_time: false,
                    };
                    write!(self.out, "constant {} {} = {{", ty_name, name,)?;
                    for (i, &sub_handle) in components.iter().enumerate() {
                        let separator = if i != 0 { ", " } else { "" };
                        let coco = ConstantContext {
                            handle: sub_handle,
                            arena: &module.constants,
                            names: &self.names,
                            first_time: false,
                        };
                        write!(self.out, "{}{}", separator, coco)?;
                    }
                    writeln!(self.out, "}};")?;
                }
            }
        }
        Ok(())
    }

    fn put_inline_sampler_properties(
        &mut self,
        level: Level,
        sampler: &sm::InlineSampler,
    ) -> Result<(), Error> {
        for (&letter, address) in ['s', 't', 'r'].iter().zip(sampler.address.iter()) {
            writeln!(
                self.out,
                "{}{}::{}_address::{},",
                level,
                NAMESPACE,
                letter,
                address.as_str(),
            )?;
        }
        writeln!(
            self.out,
            "{}{}::mag_filter::{},",
            level,
            NAMESPACE,
            sampler.mag_filter.as_str(),
        )?;
        writeln!(
            self.out,
            "{}{}::min_filter::{},",
            level,
            NAMESPACE,
            sampler.min_filter.as_str(),
        )?;
        if let Some(filter) = sampler.mip_filter {
            writeln!(
                self.out,
                "{}{}::mip_filter::{},",
                level,
                NAMESPACE,
                filter.as_str(),
            )?;
        }
        // avoid setting it on platforms that don't support it
        if sampler.border_color != sm::BorderColor::TransparentBlack {
            writeln!(
                self.out,
                "{}{}::border_color::{},",
                level,
                NAMESPACE,
                sampler.border_color.as_str(),
            )?;
        }
        //TODO: I'm not able to feed this in a way that MSL likes:
        //>error: use of undeclared identifier 'lod_clamp'
        //>error: no member named 'max_anisotropy' in namespace 'metal'
        if false {
            if let Some(ref lod) = sampler.lod_clamp {
                writeln!(self.out, "{}lod_clamp({},{}),", level, lod.start, lod.end,)?;
            }
            if let Some(aniso) = sampler.max_anisotropy {
                writeln!(self.out, "{}max_anisotropy({}),", level, aniso.get(),)?;
            }
        }
        if sampler.compare_func != sm::CompareFunc::Never {
            writeln!(
                self.out,
                "{}{}::compare_func::{},",
                level,
                NAMESPACE,
                sampler.compare_func.as_str(),
            )?;
        }
        writeln!(
            self.out,
            "{}{}::coord::{}",
            level,
            NAMESPACE,
            sampler.coord.as_str()
        )?;
        Ok(())
    }

    // Returns the array of mapped entry point names.
    fn write_functions(
        &mut self,
        module: &crate::Module,
        mod_info: &ModuleInfo,
        options: &Options,
        pipeline_options: &PipelineOptions,
    ) -> Result<TranslationInfo, Error> {
        let mut pass_through_globals = Vec::new();
        for (fun_handle, fun) in module.functions.iter() {
            let fun_info = &mod_info[fun_handle];
            pass_through_globals.clear();
            let mut supports_array_length = false;
            for (handle, var) in module.global_variables.iter() {
                if !fun_info[handle].is_empty() {
                    if var.class.needs_pass_through() {
                        pass_through_globals.push(handle);
                    }
                    supports_array_length |= needs_array_length(var.ty, &module.types);
                }
            }

            writeln!(self.out)?;
            let fun_name = &self.names[&NameKey::Function(fun_handle)];
            match fun.result {
                Some(ref result) => {
                    let ty_name = TypeContext {
                        handle: result.ty,
                        arena: &module.types,
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        first_time: false,
                    };
                    write!(self.out, "{}", ty_name)?;
                }
                None => {
                    write!(self.out, "void")?;
                }
            }
            writeln!(self.out, " {}(", fun_name)?;

            for (index, arg) in fun.arguments.iter().enumerate() {
                let name = &self.names[&NameKey::FunctionArgument(fun_handle, index as u32)];
                let param_type_name = TypeContext {
                    handle: arg.ty,
                    arena: &module.types,
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    first_time: false,
                };
                let separator = separate(
                    !pass_through_globals.is_empty()
                        || index + 1 != fun.arguments.len()
                        || supports_array_length,
                );
                writeln!(
                    self.out,
                    "{}{} {}{}",
                    INDENT, param_type_name, name, separator
                )?;
            }
            for (index, &handle) in pass_through_globals.iter().enumerate() {
                let tyvar = TypedGlobalVariable {
                    module,
                    names: &self.names,
                    handle,
                    usage: fun_info[handle],
                    reference: true,
                };
                let separator =
                    separate(index + 1 != pass_through_globals.len() || supports_array_length);
                write!(self.out, "{}", INDENT)?;
                tyvar.try_fmt(&mut self.out)?;
                writeln!(self.out, "{}", separator)?;
            }

            if supports_array_length {
                writeln!(
                    self.out,
                    "{}constant _mslBufferSizes& _buffer_sizes",
                    INDENT
                )?;
            }

            writeln!(self.out, ") {{")?;

            for (local_handle, local) in fun.local_variables.iter() {
                let ty_name = TypeContext {
                    handle: local.ty,
                    arena: &module.types,
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    first_time: false,
                };
                let local_name = &self.names[&NameKey::FunctionLocal(fun_handle, local_handle)];
                write!(self.out, "{}{} {}", INDENT, ty_name, local_name)?;
                if let Some(value) = local.init {
                    let coco = ConstantContext {
                        handle: value,
                        arena: &module.constants,
                        names: &self.names,
                        first_time: false,
                    };
                    write!(self.out, " = {}", coco)?;
                }
                writeln!(self.out, ";")?;
            }

            let context = StatementContext {
                expression: ExpressionContext {
                    function: fun,
                    origin: FunctionOrigin::Handle(fun_handle),
                    info: fun_info,
                    module,
                    pipeline_options,
                },
                mod_info,
                result_struct: None,
            };
            self.named_expressions.clear();
            self.put_block(Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
        }

        let mut info = TranslationInfo {
            entry_point_names: Vec::with_capacity(module.entry_points.len()),
        };
        for (ep_index, ep) in module.entry_points.iter().enumerate() {
            let fun = &ep.function;
            let fun_info = mod_info.get_entry_point(ep_index);
            let mut ep_error = None;
            let mut supports_array_length = false;

            // skip this entry point if any global bindings are missing
            if !options.fake_missing_bindings {
                for (var_handle, var) in module.global_variables.iter() {
                    if fun_info[var_handle].is_empty() {
                        continue;
                    }
                    if let Some(ref br) = var.binding {
                        if let Err(e) = options.resolve_resource_binding(ep.stage, br) {
                            ep_error = Some(e);
                            break;
                        }
                    }
                    if var.class == crate::StorageClass::PushConstant {
                        if let Err(e) = options.resolve_push_constants(ep.stage) {
                            ep_error = Some(e);
                            break;
                        }
                    }
                    supports_array_length |= needs_array_length(var.ty, &module.types);
                }
                if supports_array_length {
                    if let Err(err) = options.resolve_sizes_buffer(ep.stage) {
                        ep_error = Some(err);
                    }
                }
            }

            if let Some(err) = ep_error {
                info.entry_point_names.push(Err(err));
                continue;
            }
            let fun_name = &self.names[&NameKey::EntryPoint(ep_index as _)];
            info.entry_point_names.push(Ok(fun_name.clone()));

            writeln!(self.out)?;

            let stage_out_name = format!("{}Output", fun_name);
            let stage_in_name = format!("{}Input", fun_name);

            let (em_str, in_mode, out_mode) = match ep.stage {
                crate::ShaderStage::Vertex => (
                    "vertex",
                    LocationMode::VertexInput,
                    LocationMode::Intermediate,
                ),
                crate::ShaderStage::Fragment { .. } => (
                    "fragment",
                    LocationMode::Intermediate,
                    LocationMode::FragmentOutput,
                ),
                crate::ShaderStage::Compute { .. } => {
                    ("kernel", LocationMode::Uniform, LocationMode::Uniform)
                }
            };

            let mut argument_members = Vec::new();
            for (arg_index, arg) in fun.arguments.iter().enumerate() {
                match module.types[arg.ty].inner {
                    crate::TypeInner::Struct { ref members, .. } => {
                        for (member_index, member) in members.iter().enumerate() {
                            argument_members.push((
                                NameKey::StructMember(arg.ty, member_index as u32),
                                member.ty,
                                member.binding.as_ref(),
                            ))
                        }
                    }
                    _ => argument_members.push((
                        NameKey::EntryPointArgument(ep_index as _, arg_index as u32),
                        arg.ty,
                        arg.binding.as_ref(),
                    )),
                }
            }
            let varyings_member_name = self.namer.call("varyings");
            let mut varying_count = 0;
            if !argument_members.is_empty() {
                writeln!(self.out, "struct {} {{", stage_in_name)?;
                for &(ref name_key, ty, binding) in argument_members.iter() {
                    let binding = match binding {
                        Some(ref binding @ &crate::Binding::Location { .. }) => binding,
                        _ => continue,
                    };
                    varying_count += 1;
                    let name = &self.names[&name_key];
                    let ty_name = TypeContext {
                        handle: ty,
                        arena: &module.types,
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        first_time: false,
                    };
                    let resolved = options.resolve_local_binding(binding, in_mode)?;
                    write!(self.out, "{}{} {}", INDENT, ty_name, name)?;
                    resolved.try_fmt_decorated(&mut self.out, "")?;
                    writeln!(self.out, ";")?;
                }
                writeln!(self.out, "}};")?;
            }

            let result_member_name = self.namer.call("member");
            let result_type_name = match fun.result {
                Some(ref result) => {
                    let mut result_members = Vec::new();
                    if let crate::TypeInner::Struct { ref members, .. } =
                        module.types[result.ty].inner
                    {
                        for (member_index, member) in members.iter().enumerate() {
                            result_members.push((
                                &self.names[&NameKey::StructMember(result.ty, member_index as u32)],
                                member.ty,
                                member.binding.as_ref(),
                            ));
                        }
                    } else {
                        result_members.push((
                            &result_member_name,
                            result.ty,
                            result.binding.as_ref(),
                        ));
                    }

                    writeln!(self.out, "struct {} {{", stage_out_name)?;
                    for (name, ty, binding) in result_members {
                        let ty_name = TypeContext {
                            handle: ty,
                            arena: &module.types,
                            names: &self.names,
                            access: crate::StorageAccess::empty(),
                            first_time: true,
                        };
                        let binding = binding.ok_or(Error::Validation)?;
                        // Cull Distance is not supported in Metal.
                        // But we can't return UnsupportedBuiltIn error to user.
                        // Because otherwise we can't generate msl shader from any glslang SPIR-V shaders.
                        // glslang generates gl_PerVertex struct with gl_CullDistance builtin inside by default.
                        if *binding == crate::Binding::BuiltIn(crate::BuiltIn::CullDistance) {
                            log::warn!("Ignoring CullDistance BuiltIn");
                            continue;
                        }
                        if !pipeline_options.allow_point_size
                            && *binding == crate::Binding::BuiltIn(crate::BuiltIn::PointSize)
                        {
                            continue;
                        }
                        let array_len = match module.types[ty].inner {
                            crate::TypeInner::Array {
                                size: crate::ArraySize::Constant(handle),
                                ..
                            } => module.constants[handle].to_array_length(),
                            _ => None,
                        };
                        let resolved = options.resolve_local_binding(binding, out_mode)?;
                        write!(self.out, "{}{} {}", INDENT, ty_name, name)?;
                        resolved.try_fmt_decorated(&mut self.out, "")?;
                        if let Some(array_len) = array_len {
                            write!(self.out, " [{}]", array_len)?;
                        }
                        writeln!(self.out, ";")?;
                    }
                    writeln!(self.out, "}};")?;
                    &stage_out_name
                }
                None => "void",
            };
            writeln!(self.out, "{} {} {}(", em_str, result_type_name, fun_name)?;

            let mut is_first_argument = true;
            if varying_count != 0 {
                writeln!(
                    self.out,
                    "  {} {} [[stage_in]]",
                    stage_in_name, varyings_member_name
                )?;
                is_first_argument = false;
            }
            for &(ref name_key, ty, binding) in argument_members.iter() {
                let binding = match binding {
                    Some(ref binding @ &crate::Binding::BuiltIn(..)) => binding,
                    _ => continue,
                };
                let name = &self.names[&name_key];
                let ty_name = TypeContext {
                    handle: ty,
                    arena: &module.types,
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    first_time: false,
                };
                let resolved = options.resolve_local_binding(binding, in_mode)?;
                let separator = if is_first_argument {
                    is_first_argument = false;
                    ' '
                } else {
                    ','
                };
                write!(self.out, "{} {} {}", separator, ty_name, name)?;
                resolved.try_fmt_decorated(&mut self.out, "\n")?;
            }
            for (handle, var) in module.global_variables.iter() {
                let usage = fun_info[handle];
                if usage.is_empty() || var.class == crate::StorageClass::Private {
                    continue;
                }
                // the resolves have already been checked for `!fake_missing_bindings` case
                let resolved = match var.class {
                    crate::StorageClass::PushConstant => {
                        options.resolve_push_constants(ep.stage).ok()
                    }
                    crate::StorageClass::WorkGroup => None,
                    _ => options
                        .resolve_resource_binding(ep.stage, var.binding.as_ref().unwrap())
                        .ok(),
                };
                if let Some(ref resolved) = resolved {
                    // Inline samplers are be defined in the EP body
                    if resolved.as_inline_sampler(options).is_some() {
                        continue;
                    }
                }

                let tyvar = TypedGlobalVariable {
                    module,
                    names: &self.names,
                    handle,
                    usage,
                    reference: true,
                };
                let separator = if is_first_argument {
                    is_first_argument = false;
                    ' '
                } else {
                    ','
                };
                write!(self.out, "{} ", separator)?;
                tyvar.try_fmt(&mut self.out)?;
                if let Some(resolved) = resolved {
                    resolved.try_fmt_decorated(&mut self.out, "")?;
                }
                if let Some(value) = var.init {
                    let coco = ConstantContext {
                        handle: value,
                        arena: &module.constants,
                        names: &self.names,
                        first_time: false,
                    };
                    write!(self.out, " = {}", coco)?;
                }
                writeln!(self.out)?;
            }

            if supports_array_length {
                // this is checked earlier
                let resolved = options.resolve_sizes_buffer(ep.stage).unwrap();
                let separator = if module.global_variables.is_empty() {
                    ' '
                } else {
                    ','
                };
                write!(
                    self.out,
                    "{} constant _mslBufferSizes& _buffer_sizes",
                    separator,
                )?;
                resolved.try_fmt_decorated(&mut self.out, "\n")?;
            }

            // end of the entry point argument list
            writeln!(self.out, ") {{")?;

            // Metal doesn't support private mutable variables outside of functions,
            // so we put them here, just like the locals.
            for (handle, var) in module.global_variables.iter() {
                let usage = fun_info[handle];
                if usage.is_empty() {
                    continue;
                }
                if var.class == crate::StorageClass::Private {
                    let tyvar = TypedGlobalVariable {
                        module,
                        names: &self.names,
                        handle,
                        usage,
                        reference: false,
                    };
                    write!(self.out, "{}", INDENT)?;
                    tyvar.try_fmt(&mut self.out)?;
                    match var.init {
                        Some(value) => {
                            let coco = ConstantContext {
                                handle: value,
                                arena: &module.constants,
                                names: &self.names,
                                first_time: false,
                            };
                            writeln!(self.out, " = {};", coco)?;
                        }
                        None => {
                            writeln!(self.out, " = {{}};")?;
                        }
                    };
                } else if let Some(ref binding) = var.binding {
                    // write an inline sampler
                    let resolved = options.resolve_resource_binding(ep.stage, binding).unwrap();
                    if let Some(sampler) = resolved.as_inline_sampler(options) {
                        let name = &self.names[&NameKey::GlobalVariable(handle)];
                        writeln!(
                            self.out,
                            "{}constexpr {}::sampler {}(",
                            INDENT, NAMESPACE, name
                        )?;
                        self.put_inline_sampler_properties(Level(2), sampler)?;
                        writeln!(self.out, "{});", INDENT)?;
                    }
                }
            }

            // Now refactor the inputs in a way that the rest of the code expects
            for (arg_index, arg) in fun.arguments.iter().enumerate() {
                let arg_name =
                    &self.names[&NameKey::EntryPointArgument(ep_index as _, arg_index as u32)];
                match module.types[arg.ty].inner {
                    crate::TypeInner::Struct { ref members, .. } => {
                        let struct_name = &self.names[&NameKey::Type(arg.ty)];
                        write!(
                            self.out,
                            "{}const {} {} = {{ ",
                            INDENT, struct_name, arg_name
                        )?;
                        for (member_index, member) in members.iter().enumerate() {
                            let name =
                                &self.names[&NameKey::StructMember(arg.ty, member_index as u32)];
                            if member_index != 0 {
                                write!(self.out, ", ")?;
                            }
                            if let Some(crate::Binding::Location { .. }) = member.binding {
                                write!(self.out, "{}.", varyings_member_name)?;
                            }
                            write!(self.out, "{}", name)?;
                        }
                        writeln!(self.out, " }};")?;
                    }
                    _ => {
                        if let Some(crate::Binding::Location { .. }) = arg.binding {
                            writeln!(
                                self.out,
                                "{}const auto {} = {}.{};",
                                INDENT, arg_name, varyings_member_name, arg_name
                            )?;
                        }
                    }
                }
            }

            // Finally, declare all the local variables that we need
            //TODO: we can postpone this till the relevant expressions are emitted
            for (local_handle, local) in fun.local_variables.iter() {
                let name = &self.names[&NameKey::EntryPointLocal(ep_index as _, local_handle)];
                let ty_name = TypeContext {
                    handle: local.ty,
                    arena: &module.types,
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    first_time: false,
                };
                write!(self.out, "{}{} {}", INDENT, ty_name, name)?;
                if let Some(value) = local.init {
                    let coco = ConstantContext {
                        handle: value,
                        arena: &module.constants,
                        names: &self.names,
                        first_time: false,
                    };
                    write!(self.out, " = {}", coco)?;
                }
                writeln!(self.out, ";")?;
            }

            let context = StatementContext {
                expression: ExpressionContext {
                    function: fun,
                    origin: FunctionOrigin::EntryPoint(ep_index as _),
                    info: fun_info,
                    module,
                    pipeline_options,
                },
                mod_info,
                result_struct: Some(&stage_out_name),
            };
            self.named_expressions.clear();
            self.put_block(Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
            if ep_index + 1 != module.entry_points.len() {
                writeln!(self.out)?;
            }
        }

        Ok(info)
    }
}

#[test]
fn test_stack_size() {
    use crate::valid::{Capabilities, ValidationFlags};
    // create a module with at least one expression nested
    let mut module = crate::Module::default();
    let constant = module.constants.append(crate::Constant {
        name: None,
        specialization: None,
        inner: crate::ConstantInner::Scalar {
            value: crate::ScalarValue::Float(1.0),
            width: 4,
        },
    });
    let mut fun = crate::Function::default();
    let const_expr = fun
        .expressions
        .append(crate::Expression::Constant(constant));
    let nested_expr = fun.expressions.append(crate::Expression::Unary {
        op: crate::UnaryOperator::Negate,
        expr: const_expr,
    });
    fun.body
        .push(crate::Statement::Emit(fun.expressions.range_from(1)));
    fun.body.push(crate::Statement::If {
        condition: nested_expr,
        accept: Vec::new(),
        reject: Vec::new(),
    });
    let _ = module.functions.append(fun);
    // analyse the module
    let info = crate::valid::Validator::new(ValidationFlags::empty(), Capabilities::empty())
        .validate(&module)
        .unwrap();
    // process the module
    let mut writer = Writer::new(String::new());
    writer
        .write(&module, &info, &Default::default(), &Default::default())
        .unwrap();

    {
        // check expression stack
        let mut addresses = !0usize..0usize;
        for pointer in writer.put_expression_stack_pointers {
            addresses.start = addresses.start.min(pointer as usize);
            addresses.end = addresses.end.max(pointer as usize);
        }
        let stack_size = addresses.end - addresses.start;
        // check the size (in debug only)
        // last observed macOS value: 17664
        if stack_size < 17000 || stack_size > 19000 {
            panic!("`put_expression` stack size {} has changed!", stack_size);
        }
    }

    {
        // check block stack
        let mut addresses = !0usize..0usize;
        for pointer in writer.put_block_stack_pointers {
            addresses.start = addresses.start.min(pointer as usize);
            addresses.end = addresses.end.max(pointer as usize);
        }
        let stack_size = addresses.end - addresses.start;
        // check the size (in debug only)
        // last observed macOS value: 13600
        if stack_size < 12000 || stack_size > 14500 {
            panic!("`put_block` stack size {} has changed!", stack_size);
        }
    }
}
