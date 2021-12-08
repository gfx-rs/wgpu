use super::{sampler as sm, Error, LocationMode, Options, PipelineOptions, TranslationInfo};
use crate::{
    arena::Handle,
    back,
    proc::index,
    proc::{self, NameKey, TypeResolution},
    valid, FastHashMap, FastHashSet,
};
use bit_set::BitSet;
use std::{
    fmt::{Display, Error as FmtError, Formatter, Write},
    iter,
};

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

const NAMESPACE: &str = "metal";
// The name of the array member of the Metal struct types we generate to
// represent Naga `Array` types. See the comments in `Writer::write_type_defs`
// for details.
const WRAPPED_ARRAY_FIELD: &str = "inner";
// This is a hack: we need to pass a pointer to an atomic,
// but generally the backend isn't putting "&" in front of every pointer.
// Some more general handling of pointers is needed to be implemented here.
const ATOMIC_REFERENCE: &str = "&";

struct TypeContext<'a> {
    handle: Handle<crate::Type>,
    arena: &'a crate::UniqueArena<crate::Type>,
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
            crate::TypeInner::Scalar { kind, .. } => {
                match kind {
                    // work around Metal toolchain bug with `uint` typedef
                    crate::ScalarKind::Uint => write!(out, "{}::uint", NAMESPACE),
                    _ => {
                        let kind_str = scalar_kind_string(kind);
                        write!(out, "{}", kind_str)
                    }
                }
            }
            crate::TypeInner::Atomic { kind, .. } => {
                write!(out, "{}::atomic_{}", NAMESPACE, scalar_kind_string(kind))
            }
            crate::TypeInner::Vector { size, kind, .. } => {
                write!(
                    out,
                    "{}::{}{}",
                    NAMESPACE,
                    scalar_kind_string(kind),
                    back::vector_size_str(size),
                )
            }
            crate::TypeInner::Matrix { columns, rows, .. } => {
                write!(
                    out,
                    "{}::{}{}x{}",
                    NAMESPACE,
                    scalar_kind_string(crate::ScalarKind::Float),
                    back::vector_size_str(columns),
                    back::vector_size_str(rows),
                )
            }
            crate::TypeInner::Pointer { base, class } => {
                let sub = Self {
                    handle: base,
                    first_time: false,
                    ..*self
                };
                let class_name = match class.to_msl_name() {
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
                let class_name = match class.to_msl_name() {
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
                let class_name = match class.to_msl_name() {
                    Some(name) => name,
                    None => return Ok(()),
                };
                write!(
                    out,
                    "{} {}::{}{}&",
                    class_name,
                    NAMESPACE,
                    scalar_kind_string(kind),
                    back::vector_size_str(size),
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
                    crate::ImageClass::Depth { multi } => {
                        let (msaa_str, access) = if multi {
                            ("_ms", "read")
                        } else {
                            ("", "sample")
                        };
                        ("depth", msaa_str, crate::ScalarKind::Float, access)
                    }
                    crate::ImageClass::Storage { format, .. } => {
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
                            log::warn!(
                                "Storage access for {:?} (name '{}'): {:?}",
                                self.handle,
                                ty.name.as_deref().unwrap_or_default(),
                                self.access
                            );
                            unreachable!("module is not valid");
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
    usage: valid::GlobalUse,
    reference: bool,
}

impl<'a> TypedGlobalVariable<'a> {
    fn try_fmt<W: Write>(&self, out: &mut W) -> BackendResult {
        let var = &self.module.global_variables[self.handle];
        let name = &self.names[&NameKey::GlobalVariable(self.handle)];

        let storage_access = match var.class {
            crate::StorageClass::Storage { access } => access,
            _ => match self.module.types[var.ty].inner {
                crate::TypeInner::Image {
                    class: crate::ImageClass::Storage { access, .. },
                    ..
                } => access,
                _ => crate::StorageAccess::default(),
            },
        };
        let ty_name = TypeContext {
            handle: var.ty,
            arena: &self.module.types,
            names: self.names,
            access: storage_access,
            first_time: false,
        };

        let (space, access, reference) = match var.class.to_msl_name() {
            Some(space) if self.reference => {
                let access = match var.class {
                    crate::StorageClass::Private | crate::StorageClass::WorkGroup
                        if !self.usage.contains(valid::GlobalUse::WRITE) =>
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
    arena: &'a crate::Arena<crate::Constant>,
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
    named_expressions: crate::NamedExpressions,
    namer: proc::Namer,
    #[cfg(test)]
    put_expression_stack_pointers: FastHashSet<*const ()>,
    #[cfg(test)]
    put_block_stack_pointers: FastHashSet<*const ()>,
    struct_member_pads: FastHashSet<(Handle<crate::Type>, u32)>,
}

fn scalar_kind_string(kind: crate::ScalarKind) -> &'static str {
    match kind {
        crate::ScalarKind::Float => "float",
        crate::ScalarKind::Sint => "int",
        crate::ScalarKind::Uint => "uint",
        crate::ScalarKind::Bool => "bool",
    }
}

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
    //Note: this is imperfect - the same structure can be used for host-shared
    // things, where packed float would matter.
    if member.binding.is_some() {
        return None;
    }

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

fn needs_array_length(ty: Handle<crate::Type>, arena: &crate::UniqueArena<crate::Type>) -> bool {
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
            | crate::StorageClass::Storage { .. }
            | crate::StorageClass::Private
            | crate::StorageClass::WorkGroup
            | crate::StorageClass::PushConstant
            | crate::StorageClass::Handle => true,
            crate::StorageClass::Function => false,
        }
    }

    fn to_msl_name(self) -> Option<&'static str> {
        match self {
            Self::Handle => None,
            Self::Uniform | Self::PushConstant => Some("constant"),
            Self::Storage { access } if access.contains(crate::StorageAccess::STORE) => {
                Some("device")
            }
            Self::Storage { .. } => Some("constant"),
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
            | Ti::Atomic { .. }
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
    EntryPoint(proc::EntryPointIndex),
}

struct ExpressionContext<'a> {
    function: &'a crate::Function,
    origin: FunctionOrigin,
    info: &'a valid::FunctionInfo,
    module: &'a crate::Module,
    pipeline_options: &'a PipelineOptions,
    policies: index::BoundsCheckPolicies,

    /// A bitset containing the `Expression` handle indexes of expressions used
    /// as indices in `ReadZeroSkipWrite`-policy accesses. These may need to be
    /// cached in temporary variables. See `index::find_checked_indexes` for
    /// details.
    guarded_indices: BitSet,
}

impl<'a> ExpressionContext<'a> {
    fn resolve_type(&self, handle: Handle<crate::Expression>) -> &'a crate::TypeInner {
        self.info[handle].ty.inner_with(&self.module.types)
    }

    fn choose_bounds_check_policy(
        &self,
        pointer: Handle<crate::Expression>,
    ) -> index::BoundsCheckPolicy {
        self.policies
            .choose_policy(pointer, &self.module.types, self.info)
    }

    fn access_needs_check(
        &self,
        base: Handle<crate::Expression>,
        index: index::GuardedIndex,
    ) -> Option<index::IndexableLength> {
        index::access_needs_check(base, index, self.module, self.function, self.info)
    }
}

struct StatementContext<'a> {
    expression: ExpressionContext<'a>,
    mod_info: &'a valid::ModuleInfo,
    result_struct: Option<&'a str>,
}

impl<W: Write> Writer<W> {
    /// Creates a new `Writer` instance.
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            named_expressions: crate::NamedExpressions::default(),
            namer: proc::Namer::default(),
            #[cfg(test)]
            put_expression_stack_pointers: Default::default(),
            #[cfg(test)]
            put_block_stack_pointers: Default::default(),
            struct_member_pads: FastHashSet::default(),
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
    ) -> BackendResult {
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
    ) -> BackendResult {
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
    ) -> BackendResult {
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
                write!(self.out, "int2(")?;
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
    ) -> BackendResult {
        // coordinates in IR are int, but Metal expects uint
        let size_str = match *context.info[expr].ty.inner_with(&context.module.types) {
            crate::TypeInner::Scalar { .. } => "",
            crate::TypeInner::Vector { size, .. } => back::vector_size_str(size),
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
    ) -> BackendResult {
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
                write!(self.out, ", {}::gradient2d(", NAMESPACE)?;
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
    ) -> BackendResult {
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
                    back::vector_size_str(size)
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
                    back::vector_size_str(columns),
                    back::vector_size_str(rows)
                )?;
                self.put_call_parameters(components.iter().cloned(), context)?;
            }
            crate::TypeInner::Array { .. } | crate::TypeInner::Struct { .. } => {
                write!(self.out, "{} {{", &self.names[&NameKey::Type(ty)])?;
                for (index, &component) in components.iter().enumerate() {
                    if index != 0 {
                        write!(self.out, ", ")?;
                    }
                    // insert padding initialization, if needed
                    if self.struct_member_pads.contains(&(ty, index as u32)) {
                        write!(self.out, "{{}}, ")?;
                    }
                    self.put_expression(component, context, true)?;
                }
                write!(self.out, "}}")?;
            }
            _ => return Err(Error::UnsupportedCompose(ty)),
        }
        Ok(())
    }

    /// Write the maximum valid index of the dynamically sized array at the end of `handle`.
    ///
    /// The 'maximum valid index' is simply one less than the array's length.
    ///
    /// This emits an expression of the form `a / b`, so the caller must
    /// parenthesize its output if it will be applying operators of higher
    /// precedence.
    ///
    /// `handle` must be the handle of a global variable whose final member is a
    /// dynamically sized array.
    fn put_dynamic_array_max_index(
        &mut self,
        handle: Handle<crate::GlobalVariable>,
        context: &ExpressionContext,
    ) -> BackendResult {
        let global = &context.module.global_variables[handle];
        let members = match context.module.types[global.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => members,
            _ => return Err(Error::Validation),
        };

        let (offset, array_ty) = match members.last() {
            Some(&crate::StructMember { offset, ty, .. }) => (offset, ty),
            None => return Err(Error::Validation),
        };

        let (span, stride) = match context.module.types[array_ty].inner {
            crate::TypeInner::Array { base, stride, .. } => (
                context.module.types[base]
                    .inner
                    .span(&context.module.constants),
                stride,
            ),
            _ => return Err(Error::Validation),
        };

        // When the stride length is larger than the span, the final element's stride of
        // bytes would have padding following the value. But the buffer size in
        // `buffer_sizes.sizeN` may not include this padding - it only needs to be large
        // enough to hold the actual values' bytes.
        //
        // So subtract off the span to get a byte size that falls at the start or within
        // the final element. Then divide by the stride size, to get one less than the
        // length, and then add one. This works even if the buffer size does include the
        // stride padding, since division rounds towards zero (MSL 2.4 §6.1). It will fail
        // if there are zero elements in the array, but the WebGPU `validating shader binding`
        // rules, together with draw-time validation when `minBindingSize` is zero,
        // prevent that.
        write!(
            self.out,
            "(_buffer_sizes.size{idx} - {offset} - {span}) / {stride}",
            idx = handle.index(),
            offset = offset,
            span = span,
            stride = stride,
        )?;
        Ok(())
    }

    fn put_atomic_fetch(
        &mut self,
        pointer: Handle<crate::Expression>,
        key: &str,
        value: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> BackendResult {
        write!(
            self.out,
            "{}::atomic_fetch_{}_explicit({}",
            NAMESPACE, key, ATOMIC_REFERENCE
        )?;
        self.put_expression(pointer, context, true)?;
        write!(self.out, ", ")?;
        self.put_expression(value, context, true)?;
        write!(self.out, ", {}::memory_order_relaxed)", NAMESPACE)?;
        Ok(())
    }

    /// Emit code for the expression `expr_handle`.
    ///
    /// The `is_scoped` argument is true if the surrounding operators have the
    /// precedence of the comma operator, or lower. So, for example:
    ///
    /// - Pass `true` for `is_scoped` when writing function arguments, an
    ///   expression statement, an initializer expression, or anything already
    ///   wrapped in parenthesis.
    ///
    /// - Pass `false` if it is an operand of a `?:` operator, a `[]`, or really
    ///   almost anything else.
    fn put_expression(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        context: &ExpressionContext,
        is_scoped: bool,
    ) -> BackendResult {
        // Add to the set in order to track the stack size.
        #[cfg(test)]
        #[allow(trivial_casts)]
        self.put_expression_stack_pointers
            .insert(&expr_handle as *const _ as *const ());

        if let Some(name) = self.named_expressions.get(&expr_handle) {
            write!(self.out, "{}", name)?;
            return Ok(());
        }

        let expression = &context.function.expressions[expr_handle];
        log::trace!("expression {:?} = {:?}", expr_handle, expression);
        match *expression {
            crate::Expression::Access { .. } | crate::Expression::AccessIndex { .. } => {
                // This is an acceptable place to generate a `ReadZeroSkipWrite` check.
                // Since `put_bounds_checks` and `put_access_chain` handle an entire
                // access chain at a time, recursing back through `put_expression` only
                // for index expressions and the base object, we will never see intermediate
                // `Access` or `AccessIndex` expressions here.
                let policy = context.choose_bounds_check_policy(expr_handle);
                if policy == index::BoundsCheckPolicy::ReadZeroSkipWrite
                    && self.put_bounds_checks(
                        expr_handle,
                        context,
                        back::Level(0),
                        if is_scoped { "" } else { "(" },
                    )?
                {
                    write!(self.out, " ? ")?;
                    self.put_access_chain(expr_handle, policy, context)?;
                    write!(self.out, " : DefaultConstructible()")?;

                    if !is_scoped {
                        write!(self.out, ")")?;
                    }
                } else {
                    self.put_access_chain(expr_handle, policy, context)?;
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
                let size = back::vector_size_str(size);

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
                    write!(self.out, "{}", back::COMPONENTS[sc as usize])?;
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
            crate::Expression::Load { pointer } => self.put_load(pointer, context, is_scoped)?,
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
                arg3,
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
                    Mf::Asinh => "asinh",
                    Mf::Acosh => "acosh",
                    Mf::Atanh => "atanh",
                    // decomposition
                    Mf::Ceil => "ceil",
                    Mf::Floor => "floor",
                    Mf::Round => "rint",
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
                    Mf::ExtractBits => "extract_bits",
                    Mf::InsertBits => "insert_bits",
                    // data packing
                    Mf::Pack4x8snorm => "pack_float_to_unorm4x8",
                    Mf::Pack4x8unorm => "pack_float_to_snorm4x8",
                    Mf::Pack2x16snorm => "pack_float_to_unorm2x16",
                    Mf::Pack2x16unorm => "pack_float_to_snorm2x16",
                    Mf::Pack2x16float => "",
                    // data unpacking
                    Mf::Unpack4x8snorm => "unpack_snorm4x8_to_float",
                    Mf::Unpack4x8unorm => "unpack_unorm4x8_to_float",
                    Mf::Unpack2x16snorm => "unpack_snorm2x16_to_float",
                    Mf::Unpack2x16unorm => "unpack_unorm2x16_to_float",
                    Mf::Unpack2x16float => "",
                };

                if fun == Mf::Distance && scalar_argument {
                    write!(self.out, "{}::abs(", NAMESPACE)?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, " - ")?;
                    self.put_expression(arg1.unwrap(), context, false)?;
                    write!(self.out, ")")?;
                } else if fun == Mf::Unpack2x16float {
                    write!(self.out, "float2(as_type<half2>(")?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, "))")?;
                } else if fun == Mf::Pack2x16float {
                    write!(self.out, "as_type<uint>(half2(")?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, "))")?;
                } else {
                    write!(self.out, "{}::{}", NAMESPACE, fun_name)?;
                    self.put_call_parameters(
                        iter::once(arg).chain(arg1).chain(arg2).chain(arg3),
                        context,
                    )?;
                }
            }
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                let scalar = scalar_kind_string(kind);
                let (src_kind, src_width) = match *context.resolve_type(expr) {
                    crate::TypeInner::Scalar { kind, width }
                    | crate::TypeInner::Vector { kind, width, .. } => (kind, width),
                    _ => return Err(Error::Validation),
                };
                let is_bool_cast =
                    kind == crate::ScalarKind::Bool || src_kind == crate::ScalarKind::Bool;
                let op = match convert {
                    Some(w) if w == src_width || is_bool_cast => "static_cast",
                    Some(8) if kind == crate::ScalarKind::Float => {
                        return Err(Error::CapabilityNotSupported(valid::Capabilities::FLOAT64))
                    }
                    Some(_) => return Err(Error::Validation),
                    None => "as_type",
                };
                write!(self.out, "{}<", op)?;
                match *context.resolve_type(expr) {
                    crate::TypeInner::Vector { size, .. } => {
                        write!(
                            self.out,
                            "{}::{}{}",
                            NAMESPACE,
                            scalar,
                            back::vector_size_str(size)
                        )?;
                    }
                    _ => {
                        write!(self.out, "{}", scalar)?;
                    }
                }
                write!(self.out, ">(")?;
                self.put_expression(expr, context, true)?;
                write!(self.out, ")")?;
            }
            // has to be a named expression
            crate::Expression::CallResult(_) | crate::Expression::AtomicResult { .. } => {
                unreachable!()
            }
            crate::Expression::ArrayLength(expr) => {
                // Find the global to which the array belongs.
                let global = match context.function.expressions[expr] {
                    crate::Expression::AccessIndex { base, .. } => {
                        match context.function.expressions[base] {
                            crate::Expression::GlobalVariable(handle) => handle,
                            _ => return Err(Error::Validation),
                        }
                    }
                    _ => return Err(Error::Validation),
                };

                if !is_scoped {
                    write!(self.out, "(")?;
                }
                write!(self.out, "1 + ")?;
                self.put_dynamic_array_max_index(global, context)?;
                if !is_scoped {
                    write!(self.out, ")")?;
                }
            }
        }
        Ok(())
    }

    /// Write a `GuardedIndex` as a Metal expression.
    fn put_index(
        &mut self,
        index: index::GuardedIndex,
        context: &ExpressionContext,
        is_scoped: bool,
    ) -> BackendResult {
        match index {
            index::GuardedIndex::Expression(expr) => {
                self.put_expression(expr, context, is_scoped)?
            }
            index::GuardedIndex::Known(value) => write!(self.out, "{}", value)?,
        }
        Ok(())
    }

    /// Emit an index bounds check condition for `chain`, if required.
    ///
    /// `chain` is a subtree of `Access` and `AccessIndex` expressions,
    /// operating either on a pointer to a value, or on a value directly. If we cannot
    /// statically determine that all indexing operations in `chain` are within
    /// bounds, then write a conditional expression to check them dynamically,
    /// and return true. All accesses in the chain are checked by the generated
    /// expression.
    ///
    /// This assumes that the [`BoundsCheckPolicy`] for `chain` is [`ReadZeroSkipWrite`].
    ///
    /// The text written is of the form:
    ///
    /// ```ignore
    /// {level}{prefix}metal::uint(i) < 4 && metal::uint(j) < 10
    /// ```
    ///
    /// where `{level}` and `{prefix}` are the arguments to this function. For [`Store`]
    /// statements, presumably these arguments start an indented `if` statement; for
    /// [`Load`] expressions, the caller is probably building up a ternary `?:`
    /// expression. In either case, what is written is not a complete syntactic structure
    /// in its own right, and the caller will have to finish it off if we return `true`.
    ///
    /// If no expression is written, return false.
    ///
    /// [`BoundsCheckPolicy`]: index::BoundsCheckPolicy
    /// [`ReadZeroSkipWrite`]: index::BoundsCheckPolicy::ReadZeroSkipWrite
    /// [`Store`]: crate::Statement::Store
    /// [`Load`]: crate::Expression::Load
    #[allow(unused_variables)]
    fn put_bounds_checks(
        &mut self,
        mut chain: Handle<crate::Expression>,
        context: &ExpressionContext,
        level: back::Level,
        prefix: &'static str,
    ) -> Result<bool, Error> {
        let mut check_written = false;

        // Iterate over the access chain, handling each expression.
        loop {
            // Produce a `GuardedIndex`, so we can shared code between the
            // `Access` and `AccessIndex` cases.
            let (base, guarded_index) = match context.function.expressions[chain] {
                crate::Expression::Access { base, index } => {
                    (base, Some(index::GuardedIndex::Expression(index)))
                }
                crate::Expression::AccessIndex { base, index } => {
                    // Don't try to check indices into structs. Validation already took
                    // care of them, and index::needs_guard doesn't handle that case.
                    let mut base_inner = context.info[base].ty.inner_with(&context.module.types);
                    if let crate::TypeInner::Pointer { base, .. } = *base_inner {
                        base_inner = &context.module.types[base].inner;
                    }
                    match *base_inner {
                        crate::TypeInner::Struct { .. } => (base, None),
                        _ => (base, Some(index::GuardedIndex::Known(index))),
                    }
                }
                _ => break,
            };

            if let Some(index) = guarded_index {
                if let Some(length) = context.access_needs_check(base, index) {
                    if check_written {
                        write!(self.out, " && ")?;
                    } else {
                        write!(self.out, "{}{}", level, prefix)?;
                        check_written = true;
                    }

                    // Check that the index falls within bounds. Do this with a single
                    // comparison, by casting the index to `uint` first, so that negative
                    // indices become large positive values.
                    write!(self.out, "{}::uint(", NAMESPACE)?;
                    self.put_index(index, context, true)?;
                    self.out.write_str(") < ")?;
                    match length {
                        index::IndexableLength::Known(value) => write!(self.out, "{}", value)?,
                        index::IndexableLength::Dynamic => {
                            let global = context
                                .function
                                .originating_global(base)
                                .ok_or(Error::Validation)?;
                            write!(self.out, "1 + ")?;
                            self.put_dynamic_array_max_index(global, context)?
                        }
                    }
                }
            }

            chain = base
        }

        Ok(check_written)
    }

    /// Write the access chain `chain`.
    ///
    /// `chain` is a subtree of [`Access`] and [`AccessIndex`] expressions,
    /// operating either on a pointer to a value, or on a value directly.
    ///
    /// Generate bounds checks code only if `policy` is [`Restrict`]. The
    /// [`ReadZeroSkipWrite`] policy requires checks before any accesses take place, so
    /// that must be handled in the caller.
    ///
    /// Handle the entire chain, recursing back into `put_expression` only for index
    /// expressions and the base expression that originates the pointer or composite value
    /// being accessed. This allows `put_expression` to assume that any `Access` or
    /// `AccessIndex` expressions it sees are the top of a chain, so it can emit
    /// `ReadZeroSkipWrite` checks.
    ///
    /// [`Access`]: crate::Expression::Access
    /// [`AccessIndex`]: crate::Expression::AccessIndex
    /// [`Restrict`]: crate::proc::index::BoundsCheckPolicy::Restrict
    /// [`ReadZeroSkipWrite`]: crate::proc::index::BoundsCheckPolicy::ReadZeroSkipWrite
    fn put_access_chain(
        &mut self,
        chain: Handle<crate::Expression>,
        policy: index::BoundsCheckPolicy,
        context: &ExpressionContext,
    ) -> BackendResult {
        match context.function.expressions[chain] {
            crate::Expression::Access { base, index } => {
                let mut base_ty = context.info[base].ty.inner_with(&context.module.types);

                // Look through any pointers to see what we're really indexing.
                if let crate::TypeInner::Pointer { base, class: _ } = *base_ty {
                    base_ty = &context.module.types[base].inner;
                }

                self.put_subscripted_access_chain(
                    base,
                    base_ty,
                    index::GuardedIndex::Expression(index),
                    policy,
                    context,
                )?;
            }
            crate::Expression::AccessIndex { base, index } => {
                let base_resolution = &context.info[base].ty;
                let mut base_ty = base_resolution.inner_with(&context.module.types);
                let mut base_ty_handle = base_resolution.handle();

                // Look through any pointers to see what we're really indexing.
                if let crate::TypeInner::Pointer { base, class: _ } = *base_ty {
                    base_ty = &context.module.types[base].inner;
                    base_ty_handle = Some(base);
                }

                // Handle structs and anything else that can use `.x` syntax here, so
                // `put_subscripted_access_chain` won't have to handle the absurd case of
                // indexing a struct with an expression.
                match *base_ty {
                    crate::TypeInner::Struct { .. } => {
                        let base_ty = base_ty_handle.unwrap();
                        self.put_access_chain(base, policy, context)?;
                        let name = &self.names[&NameKey::StructMember(base_ty, index)];
                        write!(self.out, ".{}", name)?;
                    }
                    crate::TypeInner::ValuePointer { .. } | crate::TypeInner::Vector { .. } => {
                        self.put_access_chain(base, policy, context)?;
                        write!(self.out, ".{}", back::COMPONENTS[index as usize])?;
                    }
                    _ => {
                        self.put_subscripted_access_chain(
                            base,
                            base_ty,
                            index::GuardedIndex::Known(index),
                            policy,
                            context,
                        )?;
                    }
                }
            }
            _ => self.put_expression(chain, context, false)?,
        }

        Ok(())
    }

    /// Write a `[]`-style access of `base` by `index`.
    ///
    /// If `policy` is [`Restrict`], then generate code as needed to force all index
    /// values within bounds.
    ///
    /// The `base_ty` argument must be the type we are actually indexing, like [`Array`] or
    /// [`Vector`]. In other words, it's `base`'s type with any surrounding [`Pointer`]
    /// removed. Our callers often already have this handy.
    ///
    /// This only emits `[]` expressions; it doesn't handle struct member accesses or
    /// referencing vector components by name.
    ///
    /// [`Restrict`]: crate::proc::index::BoundsCheckPolicy::Restrict
    /// [`Array`]: crate::TypeInner::Array
    /// [`Vector`]: crate::TypeInner::Vector
    /// [`Pointer`]: crate::TypeInner::Pointer
    fn put_subscripted_access_chain(
        &mut self,
        base: Handle<crate::Expression>,
        base_ty: &crate::TypeInner,
        index: index::GuardedIndex,
        policy: index::BoundsCheckPolicy,
        context: &ExpressionContext,
    ) -> BackendResult {
        let accessing_wrapped_array = match *base_ty {
            crate::TypeInner::Array {
                size: crate::ArraySize::Constant(_),
                ..
            } => true,
            _ => false,
        };

        self.put_access_chain(base, policy, context)?;
        if accessing_wrapped_array {
            write!(self.out, ".{}", WRAPPED_ARRAY_FIELD)?;
        }
        write!(self.out, "[")?;

        // Decide whether this index needs to be clamped to fall within range.
        let restriction_needed = if policy == index::BoundsCheckPolicy::Restrict {
            context.access_needs_check(base, index)
        } else {
            None
        };
        if let Some(limit) = restriction_needed {
            write!(self.out, "{}::min(unsigned(", NAMESPACE)?;
            self.put_index(index, context, true)?;
            write!(self.out, "), ")?;
            match limit {
                index::IndexableLength::Known(limit) => {
                    write!(self.out, "{}u", limit - 1)?;
                }
                index::IndexableLength::Dynamic => {
                    let global = context
                        .function
                        .originating_global(base)
                        .ok_or(Error::Validation)?;
                    self.put_dynamic_array_max_index(global, context)?;
                }
            }
            write!(self.out, ")")?;
        } else {
            self.put_index(index, context, true)?;
        }

        write!(self.out, "]")?;

        Ok(())
    }

    fn put_load(
        &mut self,
        pointer: Handle<crate::Expression>,
        context: &ExpressionContext,
        is_scoped: bool,
    ) -> BackendResult {
        // Since access chains never cross storage classes, we can just check the index
        // bounds check policy once at the top.
        let policy = context.choose_bounds_check_policy(pointer);
        if policy == index::BoundsCheckPolicy::ReadZeroSkipWrite
            && self.put_bounds_checks(
                pointer,
                context,
                back::Level(0),
                if is_scoped { "" } else { "(" },
            )?
        {
            write!(self.out, " ? ")?;
            self.put_unchecked_load(pointer, policy, context)?;
            write!(self.out, " : DefaultConstructible()")?;

            if !is_scoped {
                write!(self.out, ")")?;
            }
        } else {
            self.put_unchecked_load(pointer, policy, context)?;
        }

        Ok(())
    }

    fn put_unchecked_load(
        &mut self,
        pointer: Handle<crate::Expression>,
        policy: index::BoundsCheckPolicy,
        context: &ExpressionContext,
    ) -> BackendResult {
        // Because packed vectors such as `packed_float3` cannot be directly multipied by
        // matrices, we convert them to unpacked vectors like `float3` on load.
        let wrap_packed_vec_scalar_kind = match context.function.expressions[pointer] {
            crate::Expression::AccessIndex { base, index } => {
                let ty = match *context.resolve_type(base) {
                    crate::TypeInner::Pointer { base, .. } => &context.module.types[base].inner,
                    ref ty => ty,
                };
                match *ty {
                    crate::TypeInner::Struct {
                        ref members, span, ..
                    } => should_pack_struct_member(members, span, index as usize, context.module),
                    _ => None,
                }
            }
            _ => None,
        };
        let is_atomic = match *context.resolve_type(pointer) {
            crate::TypeInner::Pointer { base, .. } => match context.module.types[base].inner {
                crate::TypeInner::Atomic { .. } => true,
                _ => false,
            },
            _ => false,
        };

        if let Some(scalar_kind) = wrap_packed_vec_scalar_kind {
            write!(
                self.out,
                "{}::{}3(",
                NAMESPACE,
                scalar_kind_string(scalar_kind)
            )?;
            self.put_access_chain(pointer, policy, context)?;
            write!(self.out, ")")?;
        } else if is_atomic {
            write!(
                self.out,
                "{}::atomic_load_explicit({}",
                NAMESPACE, ATOMIC_REFERENCE
            )?;
            self.put_access_chain(pointer, policy, context)?;
            write!(self.out, ", {}::memory_order_relaxed)", NAMESPACE)?;
        } else {
            // We don't do any dereferencing with `*` here as pointer arguments to functions
            // are done by `&` references and not `*` pointers. These do not need to be
            // dereferenced.
            self.put_access_chain(pointer, policy, context)?;
        }

        Ok(())
    }

    fn put_return_value(
        &mut self,
        level: back::Level,
        expr_handle: Handle<crate::Expression>,
        result_struct: Option<&str>,
        context: &ExpressionContext,
    ) -> BackendResult {
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
        name: &str,
    ) -> BackendResult {
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
                    back::vector_size_str(size)
                )?;
            }
            TypeResolution::Value(crate::TypeInner::Matrix { columns, rows, .. }) => {
                write!(
                    self.out,
                    "{}::{}{}x{}",
                    NAMESPACE,
                    scalar_kind_string(crate::ScalarKind::Float),
                    back::vector_size_str(columns),
                    back::vector_size_str(rows),
                )?;
            }
            TypeResolution::Value(ref other) => {
                log::warn!("Type {:?} isn't a known local", other); //TEMP!
                return Err(Error::FeatureNotImplemented("weird local type".to_string()));
            }
        }

        //TODO: figure out the naming scheme that wouldn't collide with user names.
        write!(self.out, " {} = ", name)?;

        Ok(())
    }

    fn put_block(
        &mut self,
        level: back::Level,
        statements: &[crate::Statement],
        context: &StatementContext,
    ) -> BackendResult {
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
                        let info = &context.expression.info[handle];
                        let ptr_class = info
                            .ty
                            .inner_with(&context.expression.module.types)
                            .pointer_class();
                        let expr_name = if ptr_class.is_some() {
                            None // don't bake pointer expressions (just yet)
                        } else if let Some(name) =
                            context.expression.function.named_expressions.get(&handle)
                        {
                            // The `crate::Function::named_expressions` table holds
                            // expressions that should be saved in temporaries once they
                            // are `Emit`ted. We only add them to `self.named_expressions`
                            // when we reach the `Emit` that covers them, so that we don't
                            // try to use their names before we've actually initialized
                            // the temporary that holds them.
                            //
                            // Don't assume the names in `named_expressions` are unique,
                            // or even valid. Use the `Namer`.
                            Some(self.namer.call(name))
                        } else {
                            // If this expression is an index that we're going to first compare
                            // against a limit, and then actually use as an index, then we may
                            // want to cache it in a temporary, to avoid evaluating it twice.
                            let bake =
                                if context.expression.guarded_indices.contains(handle.index()) {
                                    true
                                } else {
                                    // Expressions whose reference count is above the
                                    // threshold should always be stored in temporaries.
                                    let min_ref_count = context.expression.function.expressions
                                        [handle]
                                        .bake_ref_count();
                                    min_ref_count <= info.ref_count
                                };

                            if bake {
                                Some(format!("{}{}", back::BAKE_PREFIX, handle.index()))
                            } else {
                                None
                            }
                        };

                        if let Some(name) = expr_name {
                            write!(self.out, "{}", level)?;
                            self.start_baking_expression(handle, &context.expression, &name)?;
                            self.put_expression(handle, &context.expression, true)?;
                            self.named_expressions.insert(handle, name);
                            writeln!(self.out, ";")?;
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
                } => {
                    write!(self.out, "{}switch(", level)?;
                    self.put_expression(selector, &context.expression, true)?;
                    let type_postfix = match *context.expression.resolve_type(selector) {
                        crate::TypeInner::Scalar {
                            kind: crate::ScalarKind::Uint,
                            ..
                        } => "u",
                        _ => "",
                    };
                    writeln!(self.out, ") {{")?;
                    let lcase = level.next();
                    for case in cases.iter() {
                        match case.value {
                            crate::SwitchValue::Integer(value) => {
                                writeln!(self.out, "{}case {}{}: {{", lcase, value, type_postfix)?;
                            }
                            crate::SwitchValue::Default => {
                                writeln!(self.out, "{}default: {{", lcase)?;
                            }
                        }
                        self.put_block(lcase.next(), &case.body, context)?;
                        if !case.fall_through
                            && case.body.last().map_or(true, |s| !s.is_terminator())
                        {
                            writeln!(self.out, "{}break;", lcase.next())?;
                        }
                        writeln!(self.out, "{}}}", lcase)?;
                    }
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
                        level,
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
                    self.put_store(pointer, value, level, context)?
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
                        let name = format!("{}{}", back::BAKE_PREFIX, expr.index());
                        self.start_baking_expression(expr, &context.expression, &name)?;
                        self.named_expressions.insert(expr, name);
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
                crate::Statement::Atomic {
                    pointer,
                    ref fun,
                    value,
                    result,
                } => {
                    write!(self.out, "{}", level)?;
                    let res_name = format!("{}{}", back::BAKE_PREFIX, result.index());
                    self.start_baking_expression(result, &context.expression, &res_name)?;
                    self.named_expressions.insert(result, res_name);
                    match *fun {
                        crate::AtomicFunction::Add => {
                            self.put_atomic_fetch(pointer, "add", value, &context.expression)?;
                        }
                        crate::AtomicFunction::Subtract => {
                            self.put_atomic_fetch(pointer, "sub", value, &context.expression)?;
                        }
                        crate::AtomicFunction::And => {
                            self.put_atomic_fetch(pointer, "and", value, &context.expression)?;
                        }
                        crate::AtomicFunction::InclusiveOr => {
                            self.put_atomic_fetch(pointer, "or", value, &context.expression)?;
                        }
                        crate::AtomicFunction::ExclusiveOr => {
                            self.put_atomic_fetch(pointer, "xor", value, &context.expression)?;
                        }
                        crate::AtomicFunction::Min => {
                            self.put_atomic_fetch(pointer, "min", value, &context.expression)?;
                        }
                        crate::AtomicFunction::Max => {
                            self.put_atomic_fetch(pointer, "max", value, &context.expression)?;
                        }
                        crate::AtomicFunction::Exchange { compare: None } => {
                            write!(
                                self.out,
                                "{}::atomic_exchange_explicit({}",
                                NAMESPACE, ATOMIC_REFERENCE,
                            )?;
                            self.put_expression(pointer, &context.expression, true)?;
                            write!(self.out, ", ")?;
                            self.put_expression(value, &context.expression, true)?;
                            write!(self.out, ", {}::memory_order_relaxed)", NAMESPACE)?;
                        }
                        crate::AtomicFunction::Exchange { .. } => {
                            return Err(Error::FeatureNotImplemented(
                                "atomic CompareExchange".to_string(),
                            ));
                        }
                    }
                    // done
                    writeln!(self.out, ";")?;
                }
            }
        }

        // un-emit expressions
        //TODO: take care of loop/continuing?
        for statement in statements {
            if let crate::Statement::Emit(ref range) = *statement {
                for handle in range.clone() {
                    self.named_expressions.remove(&handle);
                }
            }
        }
        Ok(())
    }

    fn put_store(
        &mut self,
        pointer: Handle<crate::Expression>,
        value: Handle<crate::Expression>,
        level: back::Level,
        context: &StatementContext,
    ) -> BackendResult {
        let policy = context.expression.choose_bounds_check_policy(pointer);
        if policy == index::BoundsCheckPolicy::ReadZeroSkipWrite
            && self.put_bounds_checks(pointer, &context.expression, level, "if (")?
        {
            writeln!(self.out, ") {{")?;
            self.put_unchecked_store(pointer, value, policy, level.next(), context)?;
            writeln!(self.out, "{}}}", level)?;
        } else {
            self.put_unchecked_store(pointer, value, policy, level, context)?;
        }

        Ok(())
    }

    fn put_unchecked_store(
        &mut self,
        pointer: Handle<crate::Expression>,
        value: Handle<crate::Expression>,
        policy: index::BoundsCheckPolicy,
        level: back::Level,
        context: &StatementContext,
    ) -> BackendResult {
        let pointer_info = &context.expression.info[pointer];
        let (array_size, is_atomic) =
            match *pointer_info.ty.inner_with(&context.expression.module.types) {
                crate::TypeInner::Pointer { base, .. } => {
                    match context.expression.module.types[base].inner {
                        crate::TypeInner::Array {
                            size: crate::ArraySize::Constant(ch),
                            ..
                        } => (Some(ch), false),
                        crate::TypeInner::Atomic { .. } => (None, true),
                        _ => (None, false),
                    }
                }
                _ => (None, false),
            };

        // we can't assign fixed-size arrays
        if let Some(const_handle) = array_size {
            let size = context.expression.module.constants[const_handle]
                .to_array_length()
                .unwrap();
            write!(self.out, "{}for(int _i=0; _i<{}; ++_i) ", level, size)?;
            self.put_access_chain(pointer, policy, &context.expression)?;
            write!(self.out, ".{}[_i] = ", WRAPPED_ARRAY_FIELD)?;
            self.put_expression(value, &context.expression, true)?;
            writeln!(self.out, ".{}[_i];", WRAPPED_ARRAY_FIELD)?;
        } else if is_atomic {
            write!(
                self.out,
                "{}{}::atomic_store_explicit({}",
                level, NAMESPACE, ATOMIC_REFERENCE
            )?;
            self.put_access_chain(pointer, policy, &context.expression)?;
            write!(self.out, ", ")?;
            self.put_expression(value, &context.expression, true)?;
            writeln!(self.out, ", {}::memory_order_relaxed);", NAMESPACE)?;
        } else {
            write!(self.out, "{}", level)?;
            self.put_access_chain(pointer, policy, &context.expression)?;
            write!(self.out, " = ")?;
            self.put_expression(value, &context.expression, true)?;
            writeln!(self.out, ";")?;
        }

        Ok(())
    }

    pub fn write(
        &mut self,
        module: &crate::Module,
        info: &valid::ModuleInfo,
        options: &Options,
        pipeline_options: &PipelineOptions,
    ) -> Result<TranslationInfo, Error> {
        self.names.clear();
        self.namer
            .reset(module, super::keywords::RESERVED, &[], &mut self.names);
        self.struct_member_pads.clear();

        writeln!(
            self.out,
            "// language: metal{}.{}",
            options.lang_version.0, options.lang_version.1
        )?;
        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out)?;

        if options
            .bounds_check_policies
            .contains(index::BoundsCheckPolicy::ReadZeroSkipWrite)
        {
            self.put_default_constructible()?;
        }

        {
            let mut indices = vec![];
            for (handle, var) in module.global_variables.iter() {
                if needs_array_length(var.ty, &module.types) {
                    let idx = handle.index();
                    indices.push(idx);
                }
            }

            if !indices.is_empty() {
                writeln!(self.out, "struct _mslBufferSizes {{")?;

                for idx in indices {
                    writeln!(self.out, "{}{}::uint size{};", back::INDENT, NAMESPACE, idx)?;
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

    /// Write the definition for the `DefaultConstructible` class.
    ///
    /// The [`ReadZeroSkipWrite`] bounds check policy requires us to be able to
    /// produce 'zero' values for any type, including structs, arrays, and so
    /// on. We could do this by emitting default constructor applications, but
    /// that would entail printing the name of the type, which is more trouble
    /// than you'd think. Instead, we just construct this magic C++14 class that
    /// can be converted to any type that can be default constructed, using
    /// template parameter inference to detect which type is needed, so we don't
    /// have to figure out the name.
    ///
    /// [`ReadZeroSkipWrite`]: index::BoundsCheckPolicy::ReadZeroSkipWrite
    fn put_default_constructible(&mut self) -> BackendResult {
        writeln!(self.out, "struct DefaultConstructible {{")?;
        writeln!(self.out, "    template<typename T>")?;
        writeln!(self.out, "    operator T() && {{")?;
        writeln!(self.out, "        return T {{}};")?;
        writeln!(self.out, "    }}")?;
        writeln!(self.out, "}};")?;
        Ok(())
    }

    fn write_type_defs(&mut self, module: &crate::Module) -> BackendResult {
        for (handle, ty) in module.types.iter() {
            if !ty.needs_alias() {
                continue;
            }
            let name = &self.names[&NameKey::Type(handle)];
            match ty.inner {
                // Naga IR can pass around arrays by value, but Metal, following
                // C++, performs an array-to-pointer conversion (C++ [conv.array])
                // on expressions of array type, so assigning the array by value
                // isn't possible. However, Metal *does* assign structs by
                // value. So in our Metal output, we wrap all array types in
                // synthetic struct types:
                //
                //     struct type1 {
                //         float inner[10]
                //     };
                //
                // Then we carefully include `.inner` (`WRAPPED_ARRAY_FIELD`) in
                // any expression that actually wants access to the array.
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
                                back::INDENT,
                                base_name,
                                WRAPPED_ARRAY_FIELD,
                                coco
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
                            self.struct_member_pads.insert((handle, index as u32));
                            let pad = member.offset - last_offset;
                            writeln!(self.out, "{}char _pad{}[{}];", back::INDENT, index, pad)?;
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
                                    back::INDENT,
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
                                writeln!(
                                    self.out,
                                    "{}{} {};",
                                    back::INDENT,
                                    base_name,
                                    member_name
                                )?;

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

    fn write_scalar_constants(&mut self, module: &crate::Module) -> BackendResult {
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

    fn write_composite_constants(&mut self, module: &crate::Module) -> BackendResult {
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
        level: back::Level,
        sampler: &sm::InlineSampler,
    ) -> BackendResult {
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
        mod_info: &valid::ModuleInfo,
        options: &Options,
        pipeline_options: &PipelineOptions,
    ) -> Result<TranslationInfo, Error> {
        let mut pass_through_globals = Vec::new();
        for (fun_handle, fun) in module.functions.iter() {
            log::trace!(
                "function {:?}, handle {:?}",
                fun.name.as_deref().unwrap_or("(anonymous)"),
                fun_handle
            );

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
                    back::INDENT,
                    param_type_name,
                    name,
                    separator
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
                write!(self.out, "{}", back::INDENT)?;
                tyvar.try_fmt(&mut self.out)?;
                writeln!(self.out, "{}", separator)?;
            }

            if supports_array_length {
                writeln!(
                    self.out,
                    "{}constant _mslBufferSizes& _buffer_sizes",
                    back::INDENT
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
                write!(self.out, "{}{} {}", back::INDENT, ty_name, local_name)?;
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

            let guarded_indices =
                index::find_checked_indexes(module, fun, fun_info, options.bounds_check_policies);

            let context = StatementContext {
                expression: ExpressionContext {
                    function: fun,
                    origin: FunctionOrigin::Handle(fun_handle),
                    info: fun_info,
                    policies: options.bounds_check_policies,
                    guarded_indices,
                    module,
                    pipeline_options,
                },
                mod_info,
                result_struct: None,
            };
            self.named_expressions.clear();
            self.put_block(back::Level(1), &fun.body, &context)?;
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

            log::trace!(
                "entry point {:?}, index {:?}",
                fun.name.as_deref().unwrap_or("(anonymous)"),
                ep_index
            );

            // skip this entry point if any global bindings are missing,
            // or their types are incompatible.
            if !options.fake_missing_bindings {
                for (var_handle, var) in module.global_variables.iter() {
                    if fun_info[var_handle].is_empty() {
                        continue;
                    }
                    if let Some(ref br) = var.binding {
                        let good = match options.per_stage_map[ep.stage].resources.get(br) {
                            Some(target) => match module.types[var.ty].inner {
                                crate::TypeInner::Struct { .. } => target.buffer.is_some(),
                                crate::TypeInner::Image { .. } => target.texture.is_some(),
                                crate::TypeInner::Sampler { .. } => target.sampler.is_some(),
                                _ => false,
                            },
                            None => false,
                        };
                        if !good {
                            ep_error = Some(super::EntryPointError::MissingBinding(br.clone()));
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
                    let name = &self.names[name_key];
                    let ty_name = TypeContext {
                        handle: ty,
                        arena: &module.types,
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        first_time: false,
                    };
                    let resolved = options.resolve_local_binding(binding, in_mode)?;
                    write!(self.out, "{}{} {}", back::INDENT, ty_name, name)?;
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
                        write!(self.out, "{}{} {}", back::INDENT, ty_name, name)?;
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
                let name = &self.names[name_key];
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
                    write!(self.out, "{}", back::INDENT)?;
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
                            back::INDENT,
                            NAMESPACE,
                            name
                        )?;
                        self.put_inline_sampler_properties(back::Level(2), sampler)?;
                        writeln!(self.out, "{});", back::INDENT)?;
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
                            back::INDENT,
                            struct_name,
                            arg_name
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
                                back::INDENT,
                                arg_name,
                                varyings_member_name,
                                arg_name
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
                write!(self.out, "{}{} {}", back::INDENT, ty_name, name)?;
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

            let guarded_indices =
                index::find_checked_indexes(module, fun, fun_info, options.bounds_check_policies);

            let context = StatementContext {
                expression: ExpressionContext {
                    function: fun,
                    origin: FunctionOrigin::EntryPoint(ep_index as _),
                    info: fun_info,
                    policies: options.bounds_check_policies,
                    guarded_indices,
                    module,
                    pipeline_options,
                },
                mod_info,
                result_struct: Some(&stage_out_name),
            };
            self.named_expressions.clear();
            self.put_block(back::Level(1), &fun.body, &context)?;
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
    let constant = module.constants.append(
        crate::Constant {
            name: None,
            specialization: None,
            inner: crate::ConstantInner::Scalar {
                value: crate::ScalarValue::Float(1.0),
                width: 4,
            },
        },
        Default::default(),
    );
    let mut fun = crate::Function::default();
    let const_expr = fun
        .expressions
        .append(crate::Expression::Constant(constant), Default::default());
    let nested_expr = fun.expressions.append(
        crate::Expression::Unary {
            op: crate::UnaryOperator::Negate,
            expr: const_expr,
        },
        Default::default(),
    );
    fun.body.push(
        crate::Statement::Emit(fun.expressions.range_from(1)),
        Default::default(),
    );
    fun.body.push(
        crate::Statement::If {
            condition: nested_expr,
            accept: crate::Block::new(),
            reject: crate::Block::new(),
        },
        Default::default(),
    );
    let _ = module.functions.append(fun, Default::default());
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
        // last observed macOS value: 20528 (CI)
        if !(14000..=25000).contains(&stack_size) {
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
        // last observed macOS value: 19152 (CI)
        if !(13000..=20000).contains(&stack_size) {
            panic!("`put_block` stack size {} has changed!", stack_size);
        }
    }
}
