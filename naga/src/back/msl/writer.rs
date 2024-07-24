use super::{sampler as sm, Error, LocationMode, Options, PipelineOptions, TranslationInfo};
use crate::{
    arena::{Handle, HandleSet},
    back::{self, Baked},
    proc::index,
    proc::{self, NameKey, TypeResolution},
    valid, FastHashMap, FastHashSet,
};
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

const RT_NAMESPACE: &str = "metal::raytracing";
const RAY_QUERY_TYPE: &str = "_RayQuery";
const RAY_QUERY_FIELD_INTERSECTOR: &str = "intersector";
const RAY_QUERY_FIELD_INTERSECTION: &str = "intersection";
const RAY_QUERY_FIELD_READY: &str = "ready";
const RAY_QUERY_FUN_MAP_INTERSECTION: &str = "_map_intersection_type";

pub(crate) const MODF_FUNCTION: &str = "naga_modf";
pub(crate) const FREXP_FUNCTION: &str = "naga_frexp";

/// Write the Metal name for a Naga numeric type: scalar, vector, or matrix.
///
/// The `sizes` slice determines whether this function writes a
/// scalar, vector, or matrix type:
///
/// - An empty slice produces a scalar type.
/// - A one-element slice produces a vector type.
/// - A two element slice `[ROWS COLUMNS]` produces a matrix of the given size.
fn put_numeric_type(
    out: &mut impl Write,
    scalar: crate::Scalar,
    sizes: &[crate::VectorSize],
) -> Result<(), FmtError> {
    match (scalar, sizes) {
        (scalar, &[]) => {
            write!(out, "{}", scalar.to_msl_name())
        }
        (scalar, &[rows]) => {
            write!(
                out,
                "{}::{}{}",
                NAMESPACE,
                scalar.to_msl_name(),
                back::vector_size_str(rows)
            )
        }
        (scalar, &[rows, columns]) => {
            write!(
                out,
                "{}::{}{}x{}",
                NAMESPACE,
                scalar.to_msl_name(),
                back::vector_size_str(columns),
                back::vector_size_str(rows)
            )
        }
        (_, _) => Ok(()), // not meaningful
    }
}

const fn scalar_is_int(scalar: crate::Scalar) -> bool {
    use crate::ScalarKind::*;
    match scalar.kind {
        Sint | Uint | AbstractInt | Bool => true,
        Float | AbstractFloat => false,
    }
}

/// Prefix for cached clamped level-of-detail values for `ImageLoad` expressions.
const CLAMPED_LOD_LOAD_PREFIX: &str = "clamped_lod_e";

/// Wrapper for identifier names for clamped level-of-detail values
///
/// Values of this type implement [`std::fmt::Display`], formatting as
/// the name of the variable used to hold the cached clamped
/// level-of-detail value for an `ImageLoad` expression.
struct ClampedLod(Handle<crate::Expression>);

impl Display for ClampedLod {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.write_prefixed(f, CLAMPED_LOD_LOAD_PREFIX)
    }
}

/// Wrapper for generating `struct _mslBufferSizes` member names for
/// runtime-sized array lengths.
///
/// On Metal, `wgpu_hal` passes the element counts for all runtime-sized arrays
/// as an argument to the entry point. This argument's type in the MSL is
/// `struct _mslBufferSizes`, a Naga-synthesized struct with a `uint` member for
/// each global variable containing a runtime-sized array.
///
/// If `global` is a [`Handle`] for a [`GlobalVariable`] that contains a
/// runtime-sized array, then the value `ArraySize(global)` implements
/// [`std::fmt::Display`], formatting as the name of the struct member carrying
/// the number of elements in that runtime-sized array.
///
/// [`GlobalVariable`]: crate::GlobalVariable
struct ArraySizeMember(Handle<crate::GlobalVariable>);

impl Display for ArraySizeMember {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.write_prefixed(f, "size")
    }
}

struct TypeContext<'a> {
    handle: Handle<crate::Type>,
    gctx: proc::GlobalCtx<'a>,
    names: &'a FastHashMap<NameKey, String>,
    access: crate::StorageAccess,
    binding: Option<&'a super::ResolvedBinding>,
    first_time: bool,
}

impl<'a> TypeContext<'a> {
    fn scalar(&self) -> Option<crate::Scalar> {
        let ty = &self.gctx.types[self.handle];
        ty.inner.scalar()
    }

    fn vertex_input_dimension(&self) -> u32 {
        let ty = &self.gctx.types[self.handle];
        match ty.inner {
            crate::TypeInner::Scalar(_) => 1,
            crate::TypeInner::Vector { size, .. } => size as u32,
            _ => unreachable!(),
        }
    }
}

impl<'a> Display for TypeContext<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        let ty = &self.gctx.types[self.handle];
        if ty.needs_alias() && !self.first_time {
            let name = &self.names[&NameKey::Type(self.handle)];
            return write!(out, "{name}");
        }

        match ty.inner {
            crate::TypeInner::Scalar(scalar) => put_numeric_type(out, scalar, &[]),
            crate::TypeInner::Atomic(scalar) => {
                write!(out, "{}::atomic_{}", NAMESPACE, scalar.to_msl_name())
            }
            crate::TypeInner::Vector { size, scalar } => put_numeric_type(out, scalar, &[size]),
            crate::TypeInner::Matrix { columns, rows, .. } => {
                put_numeric_type(out, crate::Scalar::F32, &[rows, columns])
            }
            crate::TypeInner::Pointer { base, space } => {
                let sub = Self {
                    handle: base,
                    first_time: false,
                    ..*self
                };
                let space_name = match space.to_msl_name() {
                    Some(name) => name,
                    None => return Ok(()),
                };
                write!(out, "{space_name} {sub}&")
            }
            crate::TypeInner::ValuePointer {
                size,
                scalar,
                space,
            } => {
                match space.to_msl_name() {
                    Some(name) => write!(out, "{name} ")?,
                    None => return Ok(()),
                };
                match size {
                    Some(rows) => put_numeric_type(out, scalar, &[rows])?,
                    None => put_numeric_type(out, scalar, &[])?,
                };

                write!(out, "&")
            }
            crate::TypeInner::Array { base, .. } => {
                let sub = Self {
                    handle: base,
                    first_time: false,
                    ..*self
                };
                // Array lengths go at the end of the type definition,
                // so just print the element type here.
                write!(out, "{sub}")
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
                let base_name = crate::Scalar { kind, width: 4 }.to_msl_name();
                let array_str = if arrayed { "_array" } else { "" };
                write!(
                    out,
                    "{NAMESPACE}::{texture_str}{dim_str}{msaa_str}{array_str}<{base_name}, {NAMESPACE}::access::{access}>",
                )
            }
            crate::TypeInner::Sampler { comparison: _ } => {
                write!(out, "{NAMESPACE}::sampler")
            }
            crate::TypeInner::AccelerationStructure => {
                write!(out, "{RT_NAMESPACE}::instance_acceleration_structure")
            }
            crate::TypeInner::RayQuery => {
                write!(out, "{RAY_QUERY_TYPE}")
            }
            crate::TypeInner::BindingArray { base, size } => {
                let base_tyname = Self {
                    handle: base,
                    first_time: false,
                    ..*self
                };

                if let Some(&super::ResolvedBinding::Resource(super::BindTarget {
                    binding_array_size: Some(override_size),
                    ..
                })) = self.binding
                {
                    write!(out, "{NAMESPACE}::array<{base_tyname}, {override_size}>")
                } else if let crate::ArraySize::Constant(size) = size {
                    write!(out, "{NAMESPACE}::array<{base_tyname}, {size}>")
                } else {
                    unreachable!("metal requires all arrays be constant sized");
                }
            }
        }
    }
}

struct TypedGlobalVariable<'a> {
    module: &'a crate::Module,
    names: &'a FastHashMap<NameKey, String>,
    handle: Handle<crate::GlobalVariable>,
    usage: valid::GlobalUse,
    binding: Option<&'a super::ResolvedBinding>,
    reference: bool,
}

impl<'a> TypedGlobalVariable<'a> {
    fn try_fmt<W: Write>(&self, out: &mut W) -> BackendResult {
        let var = &self.module.global_variables[self.handle];
        let name = &self.names[&NameKey::GlobalVariable(self.handle)];

        let storage_access = match var.space {
            crate::AddressSpace::Storage { access } => access,
            _ => match self.module.types[var.ty].inner {
                crate::TypeInner::Image {
                    class: crate::ImageClass::Storage { access, .. },
                    ..
                } => access,
                crate::TypeInner::BindingArray { base, .. } => {
                    match self.module.types[base].inner {
                        crate::TypeInner::Image {
                            class: crate::ImageClass::Storage { access, .. },
                            ..
                        } => access,
                        _ => crate::StorageAccess::default(),
                    }
                }
                _ => crate::StorageAccess::default(),
            },
        };
        let ty_name = TypeContext {
            handle: var.ty,
            gctx: self.module.to_ctx(),
            names: self.names,
            access: storage_access,
            binding: self.binding,
            first_time: false,
        };

        let (space, access, reference) = match var.space.to_msl_name() {
            Some(space) if self.reference => {
                let access = if var.space.needs_access_qualifier()
                    && !self.usage.contains(valid::GlobalUse::WRITE)
                {
                    "const"
                } else {
                    ""
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

pub struct Writer<W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    named_expressions: crate::NamedExpressions,
    /// Set of expressions that need to be baked to avoid unnecessary repetition in output
    need_bake_expressions: back::NeedBakeExpressions,
    namer: proc::Namer,
    #[cfg(test)]
    put_expression_stack_pointers: FastHashSet<*const ()>,
    #[cfg(test)]
    put_block_stack_pointers: FastHashSet<*const ()>,
    /// Set of (struct type, struct field index) denoting which fields require
    /// padding inserted **before** them (i.e. between fields at index - 1 and index)
    struct_member_pads: FastHashSet<(Handle<crate::Type>, u32)>,
}

impl crate::Scalar {
    fn to_msl_name(self) -> &'static str {
        use crate::ScalarKind as Sk;
        match self {
            Self {
                kind: Sk::Float,
                width: _,
            } => "float",
            Self {
                kind: Sk::Sint,
                width: 4,
            } => "int",
            Self {
                kind: Sk::Uint,
                width: 4,
            } => "uint",
            Self {
                kind: Sk::Sint,
                width: 8,
            } => "long",
            Self {
                kind: Sk::Uint,
                width: 8,
            } => "ulong",
            Self {
                kind: Sk::Bool,
                width: _,
            } => "bool",
            Self {
                kind: Sk::AbstractInt | Sk::AbstractFloat,
                width: _,
            } => unreachable!("Found Abstract scalar kind"),
            _ => unreachable!("Unsupported scalar kind: {:?}", self),
        }
    }
}

const fn separate(need_separator: bool) -> &'static str {
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
) -> Option<crate::Scalar> {
    let member = &members[index];

    let ty_inner = &module.types[member.ty].inner;
    let last_offset = member.offset + ty_inner.size(module.to_ctx());
    let next_offset = match members.get(index + 1) {
        Some(next) => next.offset,
        None => span,
    };
    let is_tight = next_offset == last_offset;

    match *ty_inner {
        crate::TypeInner::Vector {
            size: crate::VectorSize::Tri,
            scalar: scalar @ crate::Scalar { width: 4, .. },
        } if is_tight => Some(scalar),
        _ => None,
    }
}

fn needs_array_length(ty: Handle<crate::Type>, arena: &crate::UniqueArena<crate::Type>) -> bool {
    match arena[ty].inner {
        crate::TypeInner::Struct { ref members, .. } => {
            if let Some(member) = members.last() {
                if let crate::TypeInner::Array {
                    size: crate::ArraySize::Dynamic,
                    ..
                } = arena[member.ty].inner
                {
                    return true;
                }
            }
            false
        }
        crate::TypeInner::Array {
            size: crate::ArraySize::Dynamic,
            ..
        } => true,
        _ => false,
    }
}

impl crate::AddressSpace {
    /// Returns true if global variables in this address space are
    /// passed in function arguments. These arguments need to be
    /// passed through any functions called from the entry point.
    const fn needs_pass_through(&self) -> bool {
        match *self {
            Self::Uniform
            | Self::Storage { .. }
            | Self::Private
            | Self::WorkGroup
            | Self::PushConstant
            | Self::Handle => true,
            Self::Function => false,
        }
    }

    /// Returns true if the address space may need a "const" qualifier.
    const fn needs_access_qualifier(&self) -> bool {
        match *self {
            //Note: we are ignoring the storage access here, and instead
            // rely on the actual use of a global by functions. This means we
            // may end up with "const" even if the binding is read-write,
            // and that should be OK.
            Self::Storage { .. } => true,
            // These should always be read-write.
            Self::Private | Self::WorkGroup => false,
            // These translate to `constant` address space, no need for qualifiers.
            Self::Uniform | Self::PushConstant => false,
            // Not applicable.
            Self::Handle | Self::Function => false,
        }
    }

    const fn to_msl_name(self) -> Option<&'static str> {
        match self {
            Self::Handle => None,
            Self::Uniform | Self::PushConstant => Some("constant"),
            Self::Storage { .. } => Some("device"),
            Self::Private | Self::Function => Some("thread"),
            Self::WorkGroup => Some("threadgroup"),
        }
    }
}

impl crate::Type {
    // Returns `true` if we need to emit an alias for this type.
    const fn needs_alias(&self) -> bool {
        use crate::TypeInner as Ti;

        match self.inner {
            // value types are concise enough, we only alias them if they are named
            Ti::Scalar(_)
            | Ti::Vector { .. }
            | Ti::Matrix { .. }
            | Ti::Atomic(_)
            | Ti::Pointer { .. }
            | Ti::ValuePointer { .. } => self.name.is_some(),
            // composite types are better to be aliased, regardless of the name
            Ti::Struct { .. } | Ti::Array { .. } => true,
            // handle types may be different, depending on the global var access, so we always inline them
            Ti::Image { .. }
            | Ti::Sampler { .. }
            | Ti::AccelerationStructure
            | Ti::RayQuery
            | Ti::BindingArray { .. } => false,
        }
    }
}

enum FunctionOrigin {
    Handle(Handle<crate::Function>),
    EntryPoint(proc::EntryPointIndex),
}

/// A level of detail argument.
///
/// When [`BoundsCheckPolicy::Restrict`] applies to an [`ImageLoad`] access, we
/// save the clamped level of detail in a temporary variable whose name is based
/// on the handle of the `ImageLoad` expression. But for other policies, we just
/// use the expression directly.
///
/// [`BoundsCheckPolicy::Restrict`]: index::BoundsCheckPolicy::Restrict
/// [`ImageLoad`]: crate::Expression::ImageLoad
#[derive(Clone, Copy)]
enum LevelOfDetail {
    Direct(Handle<crate::Expression>),
    Restricted(Handle<crate::Expression>),
}

/// Values needed to select a particular texel for [`ImageLoad`] and [`ImageStore`].
///
/// When this is used in code paths unconcerned with the `Restrict` bounds check
/// policy, the `LevelOfDetail` enum introduces an unneeded match, since `level`
/// will always be either `None` or `Some(Direct(_))`. But this turns out not to
/// be too awkward. If that changes, we can revisit.
///
/// [`ImageLoad`]: crate::Expression::ImageLoad
/// [`ImageStore`]: crate::Statement::ImageStore
struct TexelAddress {
    coordinate: Handle<crate::Expression>,
    array_index: Option<Handle<crate::Expression>>,
    sample: Option<Handle<crate::Expression>>,
    level: Option<LevelOfDetail>,
}

struct ExpressionContext<'a> {
    function: &'a crate::Function,
    origin: FunctionOrigin,
    info: &'a valid::FunctionInfo,
    module: &'a crate::Module,
    mod_info: &'a valid::ModuleInfo,
    pipeline_options: &'a PipelineOptions,
    lang_version: (u8, u8),
    policies: index::BoundsCheckPolicies,

    /// The set of expressions used as indices in `ReadZeroSkipWrite`-policy
    /// accesses. These may need to be cached in temporary variables. See
    /// `index::find_checked_indexes` for details.
    guarded_indices: HandleSet<crate::Expression>,
}

impl<'a> ExpressionContext<'a> {
    fn resolve_type(&self, handle: Handle<crate::Expression>) -> &'a crate::TypeInner {
        self.info[handle].ty.inner_with(&self.module.types)
    }

    /// Return true if calls to `image`'s `read` and `write` methods should supply a level of detail.
    ///
    /// Only mipmapped images need to specify a level of detail. Since 1D
    /// textures cannot have mipmaps, MSL requires that the level argument to
    /// texture1d queries and accesses must be a constexpr 0. It's easiest
    /// just to omit the level entirely for 1D textures.
    fn image_needs_lod(&self, image: Handle<crate::Expression>) -> bool {
        let image_ty = self.resolve_type(image);
        if let crate::TypeInner::Image { dim, class, .. } = *image_ty {
            class.is_mipmapped() && dim != crate::ImageDimension::D1
        } else {
            false
        }
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

    fn get_packed_vec_kind(&self, expr_handle: Handle<crate::Expression>) -> Option<crate::Scalar> {
        match self.function.expressions[expr_handle] {
            crate::Expression::AccessIndex { base, index } => {
                let ty = match *self.resolve_type(base) {
                    crate::TypeInner::Pointer { base, .. } => &self.module.types[base].inner,
                    ref ty => ty,
                };
                match *ty {
                    crate::TypeInner::Struct {
                        ref members, span, ..
                    } => should_pack_struct_member(members, span, index as usize, self.module),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

struct StatementContext<'a> {
    expression: ExpressionContext<'a>,
    result_struct: Option<&'a str>,
}

impl<W: Write> Writer<W> {
    /// Creates a new `Writer` instance.
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            named_expressions: Default::default(),
            need_bake_expressions: Default::default(),
            namer: proc::Namer::default(),
            #[cfg(test)]
            put_expression_stack_pointers: Default::default(),
            #[cfg(test)]
            put_block_stack_pointers: Default::default(),
            struct_member_pads: FastHashSet::default(),
        }
    }

    /// Finishes writing and returns the output.
    // See https://github.com/rust-lang/rust-clippy/issues/4979.
    #[allow(clippy::missing_const_for_fn)]
    pub fn finish(self) -> W {
        self.out
    }

    fn put_call_parameters(
        &mut self,
        parameters: impl Iterator<Item = Handle<crate::Expression>>,
        context: &ExpressionContext,
    ) -> BackendResult {
        self.put_call_parameters_impl(parameters, context, |writer, context, expr| {
            writer.put_expression(expr, context, true)
        })
    }

    fn put_call_parameters_impl<C, E>(
        &mut self,
        parameters: impl Iterator<Item = Handle<crate::Expression>>,
        ctx: &C,
        put_expression: E,
    ) -> BackendResult
    where
        E: Fn(&mut Self, &C, Handle<crate::Expression>) -> BackendResult,
    {
        write!(self.out, "(")?;
        for (i, handle) in parameters.enumerate() {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            put_expression(self, ctx, handle)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_level_of_detail(
        &mut self,
        level: LevelOfDetail,
        context: &ExpressionContext,
    ) -> BackendResult {
        match level {
            LevelOfDetail::Direct(expr) => self.put_expression(expr, context, true)?,
            LevelOfDetail::Restricted(load) => write!(self.out, "{}", ClampedLod(load))?,
        }
        Ok(())
    }

    fn put_image_query(
        &mut self,
        image: Handle<crate::Expression>,
        query: &str,
        level: Option<LevelOfDetail>,
        context: &ExpressionContext,
    ) -> BackendResult {
        self.put_expression(image, context, false)?;
        write!(self.out, ".get_{query}(")?;
        if let Some(level) = level {
            self.put_level_of_detail(level, context)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_image_size_query(
        &mut self,
        image: Handle<crate::Expression>,
        level: Option<LevelOfDetail>,
        kind: crate::ScalarKind,
        context: &ExpressionContext,
    ) -> BackendResult {
        //Note: MSL only has separate width/height/depth queries,
        // so compose the result of them.
        let dim = match *context.resolve_type(image) {
            crate::TypeInner::Image { dim, .. } => dim,
            ref other => unreachable!("Unexpected type {:?}", other),
        };
        let scalar = crate::Scalar { kind, width: 4 };
        let coordinate_type = scalar.to_msl_name();
        match dim {
            crate::ImageDimension::D1 => {
                // Since 1D textures never have mipmaps, MSL requires that the
                // `level` argument be a constexpr 0. It's simplest for us just
                // to pass `None` and omit the level entirely.
                if kind == crate::ScalarKind::Uint {
                    // No need to construct a vector. No cast needed.
                    self.put_image_query(image, "width", None, context)?;
                } else {
                    // There's no definition for `int` in the `metal` namespace.
                    write!(self.out, "int(")?;
                    self.put_image_query(image, "width", None, context)?;
                    write!(self.out, ")")?;
                }
            }
            crate::ImageDimension::D2 => {
                write!(self.out, "{NAMESPACE}::{coordinate_type}2(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "height", level, context)?;
                write!(self.out, ")")?;
            }
            crate::ImageDimension::D3 => {
                write!(self.out, "{NAMESPACE}::{coordinate_type}3(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "height", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "depth", level, context)?;
                write!(self.out, ")")?;
            }
            crate::ImageDimension::Cube => {
                write!(self.out, "{NAMESPACE}::{coordinate_type}2(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ")")?;
            }
        }
        Ok(())
    }

    fn put_cast_to_uint_scalar_or_vector(
        &mut self,
        expr: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> BackendResult {
        // coordinates in IR are int, but Metal expects uint
        match *context.resolve_type(expr) {
            crate::TypeInner::Scalar(_) => {
                put_numeric_type(&mut self.out, crate::Scalar::U32, &[])?
            }
            crate::TypeInner::Vector { size, .. } => {
                put_numeric_type(&mut self.out, crate::Scalar::U32, &[size])?
            }
            _ => {
                return Err(Error::GenericValidation(
                    "Invalid type for image coordinate".into(),
                ))
            }
        };

        write!(self.out, "(")?;
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
        let has_levels = context.image_needs_lod(image);
        match level {
            crate::SampleLevel::Auto => {}
            crate::SampleLevel::Zero => {
                //TODO: do we support Zero on `Sampled` image classes?
            }
            _ if !has_levels => {
                log::warn!("1D image can't be sampled with level {:?}", level);
            }
            crate::SampleLevel::Exact(h) => {
                write!(self.out, ", {NAMESPACE}::level(")?;
                self.put_expression(h, context, true)?;
                write!(self.out, ")")?;
            }
            crate::SampleLevel::Bias(h) => {
                write!(self.out, ", {NAMESPACE}::bias(")?;
                self.put_expression(h, context, true)?;
                write!(self.out, ")")?;
            }
            crate::SampleLevel::Gradient { x, y } => {
                write!(self.out, ", {NAMESPACE}::gradient2d(")?;
                self.put_expression(x, context, true)?;
                write!(self.out, ", ")?;
                self.put_expression(y, context, true)?;
                write!(self.out, ")")?;
            }
        }
        Ok(())
    }

    fn put_image_coordinate_limits(
        &mut self,
        image: Handle<crate::Expression>,
        level: Option<LevelOfDetail>,
        context: &ExpressionContext,
    ) -> BackendResult {
        self.put_image_size_query(image, level, crate::ScalarKind::Uint, context)?;
        write!(self.out, " - 1")?;
        Ok(())
    }

    /// General function for writing restricted image indexes.
    ///
    /// This is used to produce restricted mip levels, array indices, and sample
    /// indices for [`ImageLoad`] and [`ImageStore`] accesses under the
    /// [`Restrict`] bounds check policy.
    ///
    /// This function writes an expression of the form:
    ///
    /// ```ignore
    ///
    ///     metal::min(uint(INDEX), IMAGE.LIMIT_METHOD() - 1)
    ///
    /// ```
    ///
    /// [`ImageLoad`]: crate::Expression::ImageLoad
    /// [`ImageStore`]: crate::Statement::ImageStore
    /// [`Restrict`]: index::BoundsCheckPolicy::Restrict
    fn put_restricted_scalar_image_index(
        &mut self,
        image: Handle<crate::Expression>,
        index: Handle<crate::Expression>,
        limit_method: &str,
        context: &ExpressionContext,
    ) -> BackendResult {
        write!(self.out, "{NAMESPACE}::min(uint(")?;
        self.put_expression(index, context, true)?;
        write!(self.out, "), ")?;
        self.put_expression(image, context, false)?;
        write!(self.out, ".{limit_method}() - 1)")?;
        Ok(())
    }

    fn put_restricted_texel_address(
        &mut self,
        image: Handle<crate::Expression>,
        address: &TexelAddress,
        context: &ExpressionContext,
    ) -> BackendResult {
        // Write the coordinate.
        write!(self.out, "{NAMESPACE}::min(")?;
        self.put_cast_to_uint_scalar_or_vector(address.coordinate, context)?;
        write!(self.out, ", ")?;
        self.put_image_coordinate_limits(image, address.level, context)?;
        write!(self.out, ")")?;

        // Write the array index, if present.
        if let Some(array_index) = address.array_index {
            write!(self.out, ", ")?;
            self.put_restricted_scalar_image_index(image, array_index, "get_array_size", context)?;
        }

        // Write the sample index, if present.
        if let Some(sample) = address.sample {
            write!(self.out, ", ")?;
            self.put_restricted_scalar_image_index(image, sample, "get_num_samples", context)?;
        }

        // The level of detail should be clamped and cached by
        // `put_cache_restricted_level`, so we don't need to clamp it here.
        if let Some(level) = address.level {
            write!(self.out, ", ")?;
            self.put_level_of_detail(level, context)?;
        }

        Ok(())
    }

    /// Write an expression that is true if the given image access is in bounds.
    fn put_image_access_bounds_check(
        &mut self,
        image: Handle<crate::Expression>,
        address: &TexelAddress,
        context: &ExpressionContext,
    ) -> BackendResult {
        let mut conjunction = "";

        // First, check the level of detail. Only if that is in bounds can we
        // use it to find the appropriate bounds for the coordinates.
        let level = if let Some(level) = address.level {
            write!(self.out, "uint(")?;
            self.put_level_of_detail(level, context)?;
            write!(self.out, ") < ")?;
            self.put_expression(image, context, true)?;
            write!(self.out, ".get_num_mip_levels()")?;
            conjunction = " && ";
            Some(level)
        } else {
            None
        };

        // Check sample index, if present.
        if let Some(sample) = address.sample {
            write!(self.out, "uint(")?;
            self.put_expression(sample, context, true)?;
            write!(self.out, ") < ")?;
            self.put_expression(image, context, true)?;
            write!(self.out, ".get_num_samples()")?;
            conjunction = " && ";
        }

        // Check array index, if present.
        if let Some(array_index) = address.array_index {
            write!(self.out, "{conjunction}uint(")?;
            self.put_expression(array_index, context, true)?;
            write!(self.out, ") < ")?;
            self.put_expression(image, context, true)?;
            write!(self.out, ".get_array_size()")?;
            conjunction = " && ";
        }

        // Finally, check if the coordinates are within bounds.
        let coord_is_vector = match *context.resolve_type(address.coordinate) {
            crate::TypeInner::Vector { .. } => true,
            _ => false,
        };
        write!(self.out, "{conjunction}")?;
        if coord_is_vector {
            write!(self.out, "{NAMESPACE}::all(")?;
        }
        self.put_cast_to_uint_scalar_or_vector(address.coordinate, context)?;
        write!(self.out, " < ")?;
        self.put_image_size_query(image, level, crate::ScalarKind::Uint, context)?;
        if coord_is_vector {
            write!(self.out, ")")?;
        }

        Ok(())
    }

    fn put_image_load(
        &mut self,
        load: Handle<crate::Expression>,
        image: Handle<crate::Expression>,
        mut address: TexelAddress,
        context: &ExpressionContext,
    ) -> BackendResult {
        match context.policies.image_load {
            proc::BoundsCheckPolicy::Restrict => {
                // Use the cached restricted level of detail, if any. Omit the
                // level altogether for 1D textures.
                if address.level.is_some() {
                    address.level = if context.image_needs_lod(image) {
                        Some(LevelOfDetail::Restricted(load))
                    } else {
                        None
                    }
                }

                self.put_expression(image, context, false)?;
                write!(self.out, ".read(")?;
                self.put_restricted_texel_address(image, &address, context)?;
                write!(self.out, ")")?;
            }
            proc::BoundsCheckPolicy::ReadZeroSkipWrite => {
                write!(self.out, "(")?;
                self.put_image_access_bounds_check(image, &address, context)?;
                write!(self.out, " ? ")?;
                self.put_unchecked_image_load(image, &address, context)?;
                write!(self.out, ": DefaultConstructible())")?;
            }
            proc::BoundsCheckPolicy::Unchecked => {
                self.put_unchecked_image_load(image, &address, context)?;
            }
        }

        Ok(())
    }

    fn put_unchecked_image_load(
        &mut self,
        image: Handle<crate::Expression>,
        address: &TexelAddress,
        context: &ExpressionContext,
    ) -> BackendResult {
        self.put_expression(image, context, false)?;
        write!(self.out, ".read(")?;
        // coordinates in IR are int, but Metal expects uint
        self.put_cast_to_uint_scalar_or_vector(address.coordinate, context)?;
        if let Some(expr) = address.array_index {
            write!(self.out, ", ")?;
            self.put_expression(expr, context, true)?;
        }
        if let Some(sample) = address.sample {
            write!(self.out, ", ")?;
            self.put_expression(sample, context, true)?;
        }
        if let Some(level) = address.level {
            if context.image_needs_lod(image) {
                write!(self.out, ", ")?;
                self.put_level_of_detail(level, context)?;
            }
        }
        write!(self.out, ")")?;

        Ok(())
    }

    fn put_image_store(
        &mut self,
        level: back::Level,
        image: Handle<crate::Expression>,
        address: &TexelAddress,
        value: Handle<crate::Expression>,
        context: &StatementContext,
    ) -> BackendResult {
        match context.expression.policies.image_store {
            proc::BoundsCheckPolicy::Restrict => {
                // We don't have a restricted level value, because we don't
                // support writes to mipmapped textures.
                debug_assert!(address.level.is_none());

                write!(self.out, "{level}")?;
                self.put_expression(image, &context.expression, false)?;
                write!(self.out, ".write(")?;
                self.put_expression(value, &context.expression, true)?;
                write!(self.out, ", ")?;
                self.put_restricted_texel_address(image, address, &context.expression)?;
                writeln!(self.out, ");")?;
            }
            proc::BoundsCheckPolicy::ReadZeroSkipWrite => {
                write!(self.out, "{level}if (")?;
                self.put_image_access_bounds_check(image, address, &context.expression)?;
                writeln!(self.out, ") {{")?;
                self.put_unchecked_image_store(level.next(), image, address, value, context)?;
                writeln!(self.out, "{level}}}")?;
            }
            proc::BoundsCheckPolicy::Unchecked => {
                self.put_unchecked_image_store(level, image, address, value, context)?;
            }
        }

        Ok(())
    }

    fn put_unchecked_image_store(
        &mut self,
        level: back::Level,
        image: Handle<crate::Expression>,
        address: &TexelAddress,
        value: Handle<crate::Expression>,
        context: &StatementContext,
    ) -> BackendResult {
        write!(self.out, "{level}")?;
        self.put_expression(image, &context.expression, false)?;
        write!(self.out, ".write(")?;
        self.put_expression(value, &context.expression, true)?;
        write!(self.out, ", ")?;
        // coordinates in IR are int, but Metal expects uint
        self.put_cast_to_uint_scalar_or_vector(address.coordinate, &context.expression)?;
        if let Some(expr) = address.array_index {
            write!(self.out, ", ")?;
            self.put_expression(expr, &context.expression, true)?;
        }
        writeln!(self.out, ");")?;

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
        let (offset, array_ty) = match context.module.types[global.ty].inner {
            crate::TypeInner::Struct { ref members, .. } => match members.last() {
                Some(&crate::StructMember { offset, ty, .. }) => (offset, ty),
                None => return Err(Error::GenericValidation("Struct has no members".into())),
            },
            crate::TypeInner::Array {
                size: crate::ArraySize::Dynamic,
                ..
            } => (0, global.ty),
            ref ty => {
                return Err(Error::GenericValidation(format!(
                    "Expected type with dynamic array, got {ty:?}"
                )))
            }
        };

        let (size, stride) = match context.module.types[array_ty].inner {
            crate::TypeInner::Array { base, stride, .. } => (
                context.module.types[base]
                    .inner
                    .size(context.module.to_ctx()),
                stride,
            ),
            ref ty => {
                return Err(Error::GenericValidation(format!(
                    "Expected array type, got {ty:?}"
                )))
            }
        };

        // When the stride length is larger than the size, the final element's stride of
        // bytes would have padding following the value. But the buffer size in
        // `buffer_sizes.sizeN` may not include this padding - it only needs to be large
        // enough to hold the actual values' bytes.
        //
        // So subtract off the size to get a byte size that falls at the start or within
        // the final element. Then divide by the stride size, to get one less than the
        // length, and then add one. This works even if the buffer size does include the
        // stride padding, since division rounds towards zero (MSL 2.4 §6.1). It will fail
        // if there are zero elements in the array, but the WebGPU `validating shader binding`
        // rules, together with draw-time validation when `minBindingSize` is zero,
        // prevent that.
        write!(
            self.out,
            "(_buffer_sizes.{member} - {offset} - {size}) / {stride}",
            member = ArraySizeMember(handle),
            offset = offset,
            size = size,
            stride = stride,
        )?;
        Ok(())
    }

    fn put_atomic_operation(
        &mut self,
        pointer: Handle<crate::Expression>,
        key: &str,
        value: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> BackendResult {
        // If the pointer we're passing to the atomic operation needs to be conditional
        // for `ReadZeroSkipWrite`, the condition needs to *surround* the atomic op, and
        // the pointer operand should be unchecked.
        let policy = context.choose_bounds_check_policy(pointer);
        let checked = policy == index::BoundsCheckPolicy::ReadZeroSkipWrite
            && self.put_bounds_checks(pointer, context, back::Level(0), "")?;

        // If requested and successfully put bounds checks, continue the ternary expression.
        if checked {
            write!(self.out, " ? ")?;
        }

        write!(
            self.out,
            "{NAMESPACE}::atomic_{key}_explicit({ATOMIC_REFERENCE}"
        )?;
        self.put_access_chain(pointer, policy, context)?;
        write!(self.out, ", ")?;
        self.put_expression(value, context, true)?;
        write!(self.out, ", {NAMESPACE}::memory_order_relaxed)")?;

        // Finish the ternary expression.
        if checked {
            write!(self.out, " : DefaultConstructible()")?;
        }

        Ok(())
    }

    /// Emit code for the arithmetic expression of the dot product.
    ///
    fn put_dot_product(
        &mut self,
        arg: Handle<crate::Expression>,
        arg1: Handle<crate::Expression>,
        size: usize,
        context: &ExpressionContext,
    ) -> BackendResult {
        // Write parentheses around the dot product expression to prevent operators
        // with different precedences from applying earlier.
        write!(self.out, "(")?;

        // Cycle through all the components of the vector
        for index in 0..size {
            let component = back::COMPONENTS[index];
            // Write the addition to the previous product
            // This will print an extra '+' at the beginning but that is fine in msl
            write!(self.out, " + ")?;
            // Write the first vector expression, this expression is marked to be
            // cached so unless it can't be cached (for example, it's a Constant)
            // it shouldn't produce large expressions.
            self.put_expression(arg, context, true)?;
            // Access the current component on the first vector
            write!(self.out, ".{component} * ")?;
            // Write the second vector expression, this expression is marked to be
            // cached so unless it can't be cached (for example, it's a Constant)
            // it shouldn't produce large expressions.
            self.put_expression(arg1, context, true)?;
            // Access the current component on the second vector
            write!(self.out, ".{component}")?;
        }

        write!(self.out, ")")?;
        Ok(())
    }

    /// Emit code for the sign(i32) expression.
    ///
    fn put_isign(
        &mut self,
        arg: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> BackendResult {
        write!(self.out, "{NAMESPACE}::select({NAMESPACE}::select(")?;
        match context.resolve_type(arg) {
            &crate::TypeInner::Vector { size, .. } => {
                let size = back::vector_size_str(size);
                write!(self.out, "int{size}(-1), int{size}(1)")?;
            }
            _ => {
                write!(self.out, "-1, 1")?;
            }
        }
        write!(self.out, ", (")?;
        self.put_expression(arg, context, true)?;
        write!(self.out, " > 0)), 0, (")?;
        self.put_expression(arg, context, true)?;
        write!(self.out, " == 0))")?;
        Ok(())
    }

    fn put_const_expression(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        module: &crate::Module,
        mod_info: &valid::ModuleInfo,
    ) -> BackendResult {
        self.put_possibly_const_expression(
            expr_handle,
            &module.global_expressions,
            module,
            mod_info,
            &(module, mod_info),
            |&(_, mod_info), expr| &mod_info[expr],
            |writer, &(module, _), expr| writer.put_const_expression(expr, module, mod_info),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn put_possibly_const_expression<C, I, E>(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &crate::Arena<crate::Expression>,
        module: &crate::Module,
        mod_info: &valid::ModuleInfo,
        ctx: &C,
        get_expr_ty: I,
        put_expression: E,
    ) -> BackendResult
    where
        I: Fn(&C, Handle<crate::Expression>) -> &TypeResolution,
        E: Fn(&mut Self, &C, Handle<crate::Expression>) -> BackendResult,
    {
        match expressions[expr_handle] {
            crate::Expression::Literal(literal) => match literal {
                crate::Literal::F64(_) => {
                    return Err(Error::CapabilityNotSupported(valid::Capabilities::FLOAT64))
                }
                crate::Literal::F32(value) => {
                    if value.is_infinite() {
                        let sign = if value.is_sign_negative() { "-" } else { "" };
                        write!(self.out, "{sign}INFINITY")?;
                    } else if value.is_nan() {
                        write!(self.out, "NAN")?;
                    } else {
                        let suffix = if value.fract() == 0.0 { ".0" } else { "" };
                        write!(self.out, "{value}{suffix}")?;
                    }
                }
                crate::Literal::U32(value) => {
                    write!(self.out, "{value}u")?;
                }
                crate::Literal::I32(value) => {
                    write!(self.out, "{value}")?;
                }
                crate::Literal::U64(value) => {
                    write!(self.out, "{value}uL")?;
                }
                crate::Literal::I64(value) => {
                    write!(self.out, "{value}L")?;
                }
                crate::Literal::Bool(value) => {
                    write!(self.out, "{value}")?;
                }
                crate::Literal::AbstractInt(_) | crate::Literal::AbstractFloat(_) => {
                    return Err(Error::GenericValidation(
                        "Unsupported abstract literal".into(),
                    ));
                }
            },
            crate::Expression::Constant(handle) => {
                let constant = &module.constants[handle];
                if constant.name.is_some() {
                    write!(self.out, "{}", self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.put_const_expression(constant.init, module, mod_info)?;
                }
            }
            crate::Expression::ZeroValue(ty) => {
                let ty_name = TypeContext {
                    handle: ty,
                    gctx: module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    binding: None,
                    first_time: false,
                };
                write!(self.out, "{ty_name} {{}}")?;
            }
            crate::Expression::Compose { ty, ref components } => {
                let ty_name = TypeContext {
                    handle: ty,
                    gctx: module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    binding: None,
                    first_time: false,
                };
                write!(self.out, "{ty_name}")?;
                match module.types[ty].inner {
                    crate::TypeInner::Scalar(_)
                    | crate::TypeInner::Vector { .. }
                    | crate::TypeInner::Matrix { .. } => {
                        self.put_call_parameters_impl(
                            components.iter().copied(),
                            ctx,
                            put_expression,
                        )?;
                    }
                    crate::TypeInner::Array { .. } | crate::TypeInner::Struct { .. } => {
                        write!(self.out, " {{")?;
                        for (index, &component) in components.iter().enumerate() {
                            if index != 0 {
                                write!(self.out, ", ")?;
                            }
                            // insert padding initialization, if needed
                            if self.struct_member_pads.contains(&(ty, index as u32)) {
                                write!(self.out, "{{}}, ")?;
                            }
                            put_expression(self, ctx, component)?;
                        }
                        write!(self.out, "}}")?;
                    }
                    _ => return Err(Error::UnsupportedCompose(ty)),
                }
            }
            crate::Expression::Splat { size, value } => {
                let scalar = match *get_expr_ty(ctx, value).inner_with(&module.types) {
                    crate::TypeInner::Scalar(scalar) => scalar,
                    ref ty => {
                        return Err(Error::GenericValidation(format!(
                            "Expected splat value type must be a scalar, got {ty:?}",
                        )))
                    }
                };
                put_numeric_type(&mut self.out, scalar, &[size])?;
                write!(self.out, "(")?;
                put_expression(self, ctx, value)?;
                write!(self.out, ")")?;
            }
            _ => unreachable!(),
        }

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
            write!(self.out, "{name}")?;
            return Ok(());
        }

        let expression = &context.function.expressions[expr_handle];
        log::trace!("expression {:?} = {:?}", expr_handle, expression);
        match *expression {
            crate::Expression::Literal(_)
            | crate::Expression::Constant(_)
            | crate::Expression::ZeroValue(_)
            | crate::Expression::Compose { .. }
            | crate::Expression::Splat { .. } => {
                self.put_possibly_const_expression(
                    expr_handle,
                    &context.function.expressions,
                    context.module,
                    context.mod_info,
                    context,
                    |context, expr: Handle<crate::Expression>| &context.info[expr].ty,
                    |writer, context, expr| writer.put_expression(expr, context, true),
                )?;
            }
            crate::Expression::Override(_) => return Err(Error::Override),
            crate::Expression::Access { base, .. }
            | crate::Expression::AccessIndex { base, .. } => {
                // This is an acceptable place to generate a `ReadZeroSkipWrite` check.
                // Since `put_bounds_checks` and `put_access_chain` handle an entire
                // access chain at a time, recursing back through `put_expression` only
                // for index expressions and the base object, we will never see intermediate
                // `Access` or `AccessIndex` expressions here.
                let policy = context.choose_bounds_check_policy(base);
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
            crate::Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                self.put_wrapped_expression_for_packed_vec3_access(vector, context, false)?;
                write!(self.out, ".")?;
                for &sc in pattern[..size as usize].iter() {
                    write!(self.out, "{}", back::COMPONENTS[sc as usize])?;
                }
            }
            crate::Expression::FunctionArgument(index) => {
                let name_key = match context.origin {
                    FunctionOrigin::Handle(handle) => NameKey::FunctionArgument(handle, index),
                    FunctionOrigin::EntryPoint(ep_index) => {
                        NameKey::EntryPointArgument(ep_index, index)
                    }
                };
                let name = &self.names[&name_key];
                write!(self.out, "{name}")?;
            }
            crate::Expression::GlobalVariable(handle) => {
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(self.out, "{name}")?;
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
                write!(self.out, "{name}")?;
            }
            crate::Expression::Load { pointer } => self.put_load(pointer, context, is_scoped)?,
            crate::Expression::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                let main_op = match gather {
                    Some(_) => "gather",
                    None => "sample",
                };
                let comparison_op = match depth_ref {
                    Some(_) => "_compare",
                    None => "",
                };
                self.put_expression(image, context, false)?;
                write!(self.out, ".{main_op}{comparison_op}(")?;
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

                if let Some(offset) = offset {
                    write!(self.out, ", ")?;
                    self.put_const_expression(offset, context.module, context.mod_info)?;
                }

                match gather {
                    None | Some(crate::SwizzleComponent::X) => {}
                    Some(component) => {
                        let is_cube_map = match *context.resolve_type(image) {
                            crate::TypeInner::Image {
                                dim: crate::ImageDimension::Cube,
                                ..
                            } => true,
                            _ => false,
                        };
                        // Offset always comes before the gather, except
                        // in cube maps where it's not applicable
                        if offset.is_none() && !is_cube_map {
                            write!(self.out, ", {NAMESPACE}::int2(0)")?;
                        }
                        let letter = back::COMPONENTS[component as usize];
                        write!(self.out, ", {NAMESPACE}::component::{letter}")?;
                    }
                }
                write!(self.out, ")")?;
            }
            crate::Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let address = TexelAddress {
                    coordinate,
                    array_index,
                    sample,
                    level: level.map(LevelOfDetail::Direct),
                };
                self.put_image_load(expr_handle, image, address, context)?;
            }
            //Note: for all the queries, the signed integers are expected,
            // so a conversion is needed.
            crate::Expression::ImageQuery { image, query } => match query {
                crate::ImageQuery::Size { level } => {
                    self.put_image_size_query(
                        image,
                        level.map(LevelOfDetail::Direct),
                        crate::ScalarKind::Uint,
                        context,
                    )?;
                }
                crate::ImageQuery::NumLevels => {
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_num_mip_levels()")?;
                }
                crate::ImageQuery::NumLayers => {
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_array_size()")?;
                }
                crate::ImageQuery::NumSamples => {
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_num_samples()")?;
                }
            },
            crate::Expression::Unary { op, expr } => {
                let op_str = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::LogicalNot => "!",
                    crate::UnaryOperator::BitwiseNot => "~",
                };
                write!(self.out, "{op_str}(")?;
                self.put_expression(expr, context, false)?;
                write!(self.out, ")")?;
            }
            crate::Expression::Binary { op, left, right } => {
                let op_str = back::binary_operation_str(op);
                let kind = context
                    .resolve_type(left)
                    .scalar_kind()
                    .ok_or(Error::UnsupportedBinaryOp(op))?;

                // TODO: handle undefined behavior of BinaryOperator::Modulo
                //
                // sint:
                // if right == 0 return 0
                // if left == min(type_of(left)) && right == -1 return 0
                // if sign(left) == -1 || sign(right) == -1 return result as defined by WGSL
                //
                // uint:
                // if right == 0 return 0
                //
                // float:
                // if right == 0 return ? see https://github.com/gpuweb/gpuweb/issues/2798

                if op == crate::BinaryOperator::Modulo && kind == crate::ScalarKind::Float {
                    write!(self.out, "{NAMESPACE}::fmod(")?;
                    self.put_expression(left, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(right, context, true)?;
                    write!(self.out, ")")?;
                } else {
                    if !is_scoped {
                        write!(self.out, "(")?;
                    }

                    // Cast packed vector if necessary
                    // Packed vector - matrix multiplications are not supported in MSL
                    if op == crate::BinaryOperator::Multiply
                        && matches!(
                            context.resolve_type(right),
                            &crate::TypeInner::Matrix { .. }
                        )
                    {
                        self.put_wrapped_expression_for_packed_vec3_access(left, context, false)?;
                    } else {
                        self.put_expression(left, context, false)?;
                    }

                    write!(self.out, " {op_str} ")?;

                    // See comment above
                    if op == crate::BinaryOperator::Multiply
                        && matches!(context.resolve_type(left), &crate::TypeInner::Matrix { .. })
                    {
                        self.put_wrapped_expression_for_packed_vec3_access(right, context, false)?;
                    } else {
                        self.put_expression(right, context, false)?;
                    }

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
                crate::TypeInner::Scalar(crate::Scalar {
                    kind: crate::ScalarKind::Bool,
                    ..
                }) => {
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
                    scalar:
                        crate::Scalar {
                            kind: crate::ScalarKind::Bool,
                            ..
                        },
                    ..
                } => {
                    write!(self.out, "{NAMESPACE}::select(")?;
                    self.put_expression(reject, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(accept, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(condition, context, true)?;
                    write!(self.out, ")")?;
                }
                ref ty => {
                    return Err(Error::GenericValidation(format!(
                        "Expected select condition to be a non-bool type, got {ty:?}",
                    )))
                }
            },
            crate::Expression::Derivative { axis, expr, .. } => {
                use crate::DerivativeAxis as Axis;
                let op = match axis {
                    Axis::X => "dfdx",
                    Axis::Y => "dfdy",
                    Axis::Width => "fwidth",
                };
                write!(self.out, "{NAMESPACE}::{op}")?;
                self.put_call_parameters(iter::once(expr), context)?;
            }
            crate::Expression::Relational { fun, argument } => {
                let op = match fun {
                    crate::RelationalFunction::Any => "any",
                    crate::RelationalFunction::All => "all",
                    crate::RelationalFunction::IsNan => "isnan",
                    crate::RelationalFunction::IsInf => "isinf",
                };
                write!(self.out, "{NAMESPACE}::{op}")?;
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

                let arg_type = context.resolve_type(arg);
                let scalar_argument = match arg_type {
                    &crate::TypeInner::Scalar(_) => true,
                    _ => false,
                };

                let fun_name = match fun {
                    // comparison
                    Mf::Abs => "abs",
                    Mf::Min => "min",
                    Mf::Max => "max",
                    Mf::Clamp => "clamp",
                    Mf::Saturate => "saturate",
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
                    Mf::Radians => "",
                    Mf::Degrees => "",
                    // decomposition
                    Mf::Ceil => "ceil",
                    Mf::Floor => "floor",
                    Mf::Round => "rint",
                    Mf::Fract => "fract",
                    Mf::Trunc => "trunc",
                    Mf::Modf => MODF_FUNCTION,
                    Mf::Frexp => FREXP_FUNCTION,
                    Mf::Ldexp => "ldexp",
                    // exponent
                    Mf::Exp => "exp",
                    Mf::Exp2 => "exp2",
                    Mf::Log => "log",
                    Mf::Log2 => "log2",
                    Mf::Pow => "pow",
                    // geometry
                    Mf::Dot => match *context.resolve_type(arg) {
                        crate::TypeInner::Vector {
                            scalar:
                                crate::Scalar {
                                    kind: crate::ScalarKind::Float,
                                    ..
                                },
                            ..
                        } => "dot",
                        crate::TypeInner::Vector { size, .. } => {
                            return self.put_dot_product(arg, arg1.unwrap(), size as usize, context)
                        }
                        _ => unreachable!(
                            "Correct TypeInner for dot product should be already validated"
                        ),
                    },
                    Mf::Outer => return Err(Error::UnsupportedCall(format!("{fun:?}"))),
                    Mf::Cross => "cross",
                    Mf::Distance => "distance",
                    Mf::Length if scalar_argument => "abs",
                    Mf::Length => "length",
                    Mf::Normalize => "normalize",
                    Mf::FaceForward => "faceforward",
                    Mf::Reflect => "reflect",
                    Mf::Refract => "refract",
                    // computational
                    Mf::Sign => match arg_type.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => {
                            return self.put_isign(arg, context);
                        }
                        _ => "sign",
                    },
                    Mf::Fma => "fma",
                    Mf::Mix => "mix",
                    Mf::Step => "step",
                    Mf::SmoothStep => "smoothstep",
                    Mf::Sqrt => "sqrt",
                    Mf::InverseSqrt => "rsqrt",
                    Mf::Inverse => return Err(Error::UnsupportedCall(format!("{fun:?}"))),
                    Mf::Transpose => "transpose",
                    Mf::Determinant => "determinant",
                    // bits
                    Mf::CountTrailingZeros => "ctz",
                    Mf::CountLeadingZeros => "clz",
                    Mf::CountOneBits => "popcount",
                    Mf::ReverseBits => "reverse_bits",
                    Mf::ExtractBits => "",
                    Mf::InsertBits => "",
                    Mf::FindLsb => "",
                    Mf::FindMsb => "",
                    // data packing
                    Mf::Pack4x8snorm => "pack_float_to_snorm4x8",
                    Mf::Pack4x8unorm => "pack_float_to_unorm4x8",
                    Mf::Pack2x16snorm => "pack_float_to_snorm2x16",
                    Mf::Pack2x16unorm => "pack_float_to_unorm2x16",
                    Mf::Pack2x16float => "",
                    Mf::Pack4xI8 => "",
                    Mf::Pack4xU8 => "",
                    // data unpacking
                    Mf::Unpack4x8snorm => "unpack_snorm4x8_to_float",
                    Mf::Unpack4x8unorm => "unpack_unorm4x8_to_float",
                    Mf::Unpack2x16snorm => "unpack_snorm2x16_to_float",
                    Mf::Unpack2x16unorm => "unpack_unorm2x16_to_float",
                    Mf::Unpack2x16float => "",
                    Mf::Unpack4xI8 => "",
                    Mf::Unpack4xU8 => "",
                };

                match fun {
                    Mf::ReverseBits | Mf::ExtractBits | Mf::InsertBits => {
                        // reverse_bits is listed as requiring MSL 2.1 but that
                        // is a copy/paste error. Looking at previous snapshots
                        // on web.archive.org it's present in MSL 1.2.
                        //
                        // https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/WhatsNewiniOS10tvOS10andOSX1012/WhatsNewiniOS10tvOS10andOSX1012.html
                        // also talks about MSL 1.2 adding "New integer
                        // functions to extract, insert, and reverse bits, as
                        // described in Integer Functions."
                        if context.lang_version < (1, 2) {
                            return Err(Error::UnsupportedFunction(fun_name.to_string()));
                        }
                    }
                    _ => {}
                }

                match fun {
                    Mf::Distance if scalar_argument => {
                        write!(self.out, "{NAMESPACE}::abs(")?;
                        self.put_expression(arg, context, false)?;
                        write!(self.out, " - ")?;
                        self.put_expression(arg1.unwrap(), context, false)?;
                        write!(self.out, ")")?;
                    }
                    Mf::FindLsb => {
                        let scalar = context.resolve_type(arg).scalar().unwrap();
                        let constant = scalar.width * 8 + 1;

                        write!(self.out, "((({NAMESPACE}::ctz(")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, ") + 1) % {constant}) - 1)")?;
                    }
                    Mf::FindMsb => {
                        let inner = context.resolve_type(arg);
                        let scalar = inner.scalar().unwrap();
                        let constant = scalar.width * 8 - 1;

                        write!(
                            self.out,
                            "{NAMESPACE}::select({constant} - {NAMESPACE}::clz("
                        )?;

                        if scalar.kind == crate::ScalarKind::Sint {
                            write!(self.out, "{NAMESPACE}::select(")?;
                            self.put_expression(arg, context, true)?;
                            write!(self.out, ", ~")?;
                            self.put_expression(arg, context, true)?;
                            write!(self.out, ", ")?;
                            self.put_expression(arg, context, true)?;
                            write!(self.out, " < 0)")?;
                        } else {
                            self.put_expression(arg, context, true)?;
                        }

                        write!(self.out, "), ")?;

                        // or metal will complain that select is ambiguous
                        match *inner {
                            crate::TypeInner::Vector { size, scalar } => {
                                let size = back::vector_size_str(size);
                                let name = scalar.to_msl_name();
                                write!(self.out, "{name}{size}")?;
                            }
                            crate::TypeInner::Scalar(scalar) => {
                                let name = scalar.to_msl_name();
                                write!(self.out, "{name}")?;
                            }
                            _ => (),
                        }

                        write!(self.out, "(-1), ")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, " == 0 || ")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, " == -1)")?;
                    }
                    Mf::Unpack2x16float => {
                        write!(self.out, "float2(as_type<half2>(")?;
                        self.put_expression(arg, context, false)?;
                        write!(self.out, "))")?;
                    }
                    Mf::Pack2x16float => {
                        write!(self.out, "as_type<uint>(half2(")?;
                        self.put_expression(arg, context, false)?;
                        write!(self.out, "))")?;
                    }
                    Mf::ExtractBits => {
                        // The behavior of ExtractBits is undefined when offset + count > bit_width. We need
                        // to first sanitize the offset and count first. If we don't do this, Apple chips
                        // will return out-of-spec values if the extracted range is not within the bit width.
                        //
                        // This encodes the exact formula specified by the wgsl spec, without temporary values:
                        // https://gpuweb.github.io/gpuweb/wgsl/#extractBits-unsigned-builtin
                        //
                        // w = sizeof(x) * 8
                        // o = min(offset, w)
                        // tmp = w - o
                        // c = min(count, tmp)
                        //
                        // bitfieldExtract(x, o, c)
                        //
                        // extract_bits(e, min(offset, w), min(count, w - min(offset, w))))

                        let scalar_bits = context.resolve_type(arg).scalar_width().unwrap() * 8;

                        write!(self.out, "{NAMESPACE}::extract_bits(")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, ", {NAMESPACE}::min(")?;
                        self.put_expression(arg1.unwrap(), context, true)?;
                        write!(self.out, ", {scalar_bits}u), {NAMESPACE}::min(")?;
                        self.put_expression(arg2.unwrap(), context, true)?;
                        write!(self.out, ", {scalar_bits}u - {NAMESPACE}::min(")?;
                        self.put_expression(arg1.unwrap(), context, true)?;
                        write!(self.out, ", {scalar_bits}u)))")?;
                    }
                    Mf::InsertBits => {
                        // The behavior of InsertBits has the same issue as ExtractBits.
                        //
                        // insertBits(e, newBits, min(offset, w), min(count, w - min(offset, w))))

                        let scalar_bits = context.resolve_type(arg).scalar_width().unwrap() * 8;

                        write!(self.out, "{NAMESPACE}::insert_bits(")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, ", ")?;
                        self.put_expression(arg1.unwrap(), context, true)?;
                        write!(self.out, ", {NAMESPACE}::min(")?;
                        self.put_expression(arg2.unwrap(), context, true)?;
                        write!(self.out, ", {scalar_bits}u), {NAMESPACE}::min(")?;
                        self.put_expression(arg3.unwrap(), context, true)?;
                        write!(self.out, ", {scalar_bits}u - {NAMESPACE}::min(")?;
                        self.put_expression(arg2.unwrap(), context, true)?;
                        write!(self.out, ", {scalar_bits}u)))")?;
                    }
                    Mf::Radians => {
                        write!(self.out, "((")?;
                        self.put_expression(arg, context, false)?;
                        write!(self.out, ") * 0.017453292519943295474)")?;
                    }
                    Mf::Degrees => {
                        write!(self.out, "((")?;
                        self.put_expression(arg, context, false)?;
                        write!(self.out, ") * 57.295779513082322865)")?;
                    }
                    Mf::Modf | Mf::Frexp => {
                        write!(self.out, "{fun_name}")?;
                        self.put_call_parameters(iter::once(arg), context)?;
                    }
                    fun @ (Mf::Pack4xI8 | Mf::Pack4xU8) => {
                        let was_signed = fun == Mf::Pack4xI8;
                        if was_signed {
                            write!(self.out, "uint(")?;
                        }
                        write!(self.out, "(")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, "[0] & 0xFF) | ((")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, "[1] & 0xFF) << 8) | ((")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, "[2] & 0xFF) << 16) | ((")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, "[3] & 0xFF) << 24)")?;
                        if was_signed {
                            write!(self.out, ")")?;
                        }
                    }
                    fun @ (Mf::Unpack4xI8 | Mf::Unpack4xU8) => {
                        if matches!(fun, Mf::Unpack4xU8) {
                            write!(self.out, "u")?;
                        }
                        write!(self.out, "int4(")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, ", ")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, " >> 8, ")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, " >> 16, ")?;
                        self.put_expression(arg, context, true)?;
                        write!(self.out, " >> 24) << 24 >> 24")?;
                    }
                    _ => {
                        write!(self.out, "{NAMESPACE}::{fun_name}")?;
                        self.put_call_parameters(
                            iter::once(arg).chain(arg1).chain(arg2).chain(arg3),
                            context,
                        )?;
                    }
                }
            }
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => match *context.resolve_type(expr) {
                crate::TypeInner::Scalar(src) | crate::TypeInner::Vector { scalar: src, .. } => {
                    let target_scalar = crate::Scalar {
                        kind,
                        width: convert.unwrap_or(src.width),
                    };
                    let op = match convert {
                        Some(_) => "static_cast",
                        None => "as_type",
                    };
                    write!(self.out, "{op}<")?;
                    match *context.resolve_type(expr) {
                        crate::TypeInner::Vector { size, .. } => {
                            put_numeric_type(&mut self.out, target_scalar, &[size])?
                        }
                        _ => put_numeric_type(&mut self.out, target_scalar, &[])?,
                    };
                    write!(self.out, ">(")?;
                    self.put_expression(expr, context, true)?;
                    write!(self.out, ")")?;
                }
                crate::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                } => {
                    let target_scalar = crate::Scalar {
                        kind,
                        width: convert.unwrap_or(scalar.width),
                    };
                    put_numeric_type(&mut self.out, target_scalar, &[rows, columns])?;
                    write!(self.out, "(")?;
                    self.put_expression(expr, context, true)?;
                    write!(self.out, ")")?;
                }
                ref ty => {
                    return Err(Error::GenericValidation(format!(
                        "Unsupported type for As: {ty:?}"
                    )))
                }
            },
            // has to be a named expression
            crate::Expression::CallResult(_)
            | crate::Expression::AtomicResult { .. }
            | crate::Expression::WorkGroupUniformLoadResult { .. }
            | crate::Expression::SubgroupBallotResult
            | crate::Expression::SubgroupOperationResult { .. }
            | crate::Expression::RayQueryProceedResult => {
                unreachable!()
            }
            crate::Expression::ArrayLength(expr) => {
                // Find the global to which the array belongs.
                let global = match context.function.expressions[expr] {
                    crate::Expression::AccessIndex { base, .. } => {
                        match context.function.expressions[base] {
                            crate::Expression::GlobalVariable(handle) => handle,
                            ref ex => {
                                return Err(Error::GenericValidation(format!(
                                    "Expected global variable in AccessIndex, got {ex:?}"
                                )))
                            }
                        }
                    }
                    crate::Expression::GlobalVariable(handle) => handle,
                    ref ex => {
                        return Err(Error::GenericValidation(format!(
                            "Unexpected expression in ArrayLength, got {ex:?}"
                        )))
                    }
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
            crate::Expression::RayQueryGetIntersection { query, committed } => {
                if context.lang_version < (2, 4) {
                    return Err(Error::UnsupportedRayTracing);
                }

                if !committed {
                    unimplemented!()
                }
                let ty = context.module.special_types.ray_intersection.unwrap();
                let type_name = &self.names[&NameKey::Type(ty)];
                write!(self.out, "{type_name} {{{RAY_QUERY_FUN_MAP_INTERSECTION}(")?;
                self.put_expression(query, context, true)?;
                write!(self.out, ".{RAY_QUERY_FIELD_INTERSECTION}.type)")?;
                let fields = [
                    "distance",
                    "user_instance_id", // req Metal 2.4
                    "instance_id",
                    "", // SBT offset
                    "geometry_id",
                    "primitive_id",
                    "triangle_barycentric_coord",
                    "triangle_front_facing",
                    "",                          // padding
                    "object_to_world_transform", // req Metal 2.4
                    "world_to_object_transform", // req Metal 2.4
                ];
                for field in fields {
                    write!(self.out, ", ")?;
                    if field.is_empty() {
                        write!(self.out, "{{}}")?;
                    } else {
                        self.put_expression(query, context, true)?;
                        write!(self.out, ".{RAY_QUERY_FIELD_INTERSECTION}.{field}")?;
                    }
                }
                write!(self.out, "}}")?;
            }
        }
        Ok(())
    }

    /// Used by expressions like Swizzle and Binary since they need packed_vec3's to be casted to a vec3
    fn put_wrapped_expression_for_packed_vec3_access(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        context: &ExpressionContext,
        is_scoped: bool,
    ) -> BackendResult {
        if let Some(scalar) = context.get_packed_vec_kind(expr_handle) {
            write!(self.out, "{}::{}3(", NAMESPACE, scalar.to_msl_name())?;
            self.put_expression(expr_handle, context, is_scoped)?;
            write!(self.out, ")")?;
        } else {
            self.put_expression(expr_handle, context, is_scoped)?;
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
            index::GuardedIndex::Known(value) => write!(self.out, "{value}")?,
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
    /// {level}{prefix}uint(i) < 4 && uint(j) < 10
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
                    let mut base_inner = context.resolve_type(base);
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
                        write!(self.out, "{level}{prefix}")?;
                        check_written = true;
                    }

                    // Check that the index falls within bounds. Do this with a single
                    // comparison, by casting the index to `uint` first, so that negative
                    // indices become large positive values.
                    write!(self.out, "uint(")?;
                    self.put_index(index, context, true)?;
                    self.out.write_str(") < ")?;
                    match length {
                        index::IndexableLength::Known(value) => write!(self.out, "{value}")?,
                        index::IndexableLength::Dynamic => {
                            let global =
                                context.function.originating_global(base).ok_or_else(|| {
                                    Error::GenericValidation(
                                        "Could not find originating global".into(),
                                    )
                                })?;
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
                let mut base_ty = context.resolve_type(base);

                // Look through any pointers to see what we're really indexing.
                if let crate::TypeInner::Pointer { base, space: _ } = *base_ty {
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
                if let crate::TypeInner::Pointer { base, space: _ } = *base_ty {
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
                        write!(self.out, ".{name}")?;
                    }
                    crate::TypeInner::ValuePointer { .. } | crate::TypeInner::Vector { .. } => {
                        self.put_access_chain(base, policy, context)?;
                        // Prior to Metal v2.1 component access for packed vectors wasn't available
                        // however array indexing is
                        if context.get_packed_vec_kind(base).is_some() {
                            write!(self.out, "[{index}]")?;
                        } else {
                            write!(self.out, ".{}", back::COMPONENTS[index as usize])?;
                        }
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
            write!(self.out, ".{WRAPPED_ARRAY_FIELD}")?;
        }
        write!(self.out, "[")?;

        // Decide whether this index needs to be clamped to fall within range.
        let restriction_needed = if policy == index::BoundsCheckPolicy::Restrict {
            context.access_needs_check(base, index)
        } else {
            None
        };
        if let Some(limit) = restriction_needed {
            write!(self.out, "{NAMESPACE}::min(unsigned(")?;
            self.put_index(index, context, true)?;
            write!(self.out, "), ")?;
            match limit {
                index::IndexableLength::Known(limit) => {
                    write!(self.out, "{}u", limit - 1)?;
                }
                index::IndexableLength::Dynamic => {
                    let global = context.function.originating_global(base).ok_or_else(|| {
                        Error::GenericValidation("Could not find originating global".into())
                    })?;
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
        // Since access chains never cross between address spaces, we can just
        // check the index bounds check policy once at the top.
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
        let is_atomic_pointer = context
            .resolve_type(pointer)
            .is_atomic_pointer(&context.module.types);

        if is_atomic_pointer {
            write!(
                self.out,
                "{NAMESPACE}::atomic_load_explicit({ATOMIC_REFERENCE}"
            )?;
            self.put_access_chain(pointer, policy, context)?;
            write!(self.out, ", {NAMESPACE}::memory_order_relaxed)")?;
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
                let mut has_point_size = false;
                let result_ty = context.function.result.as_ref().unwrap().ty;
                match context.module.types[result_ty].inner {
                    crate::TypeInner::Struct { ref members, .. } => {
                        let tmp = "_tmp";
                        write!(self.out, "{level}const auto {tmp} = ")?;
                        self.put_expression(expr_handle, context, true)?;
                        writeln!(self.out, ";")?;
                        write!(self.out, "{level}return {struct_name} {{")?;

                        let mut is_first = true;

                        for (index, member) in members.iter().enumerate() {
                            if let Some(crate::Binding::BuiltIn(crate::BuiltIn::PointSize)) =
                                member.binding
                            {
                                has_point_size = true;
                                if !context.pipeline_options.allow_and_force_point_size {
                                    continue;
                                }
                            }

                            let comma = if is_first { "" } else { "," };
                            is_first = false;
                            let name = &self.names[&NameKey::StructMember(result_ty, index as u32)];
                            // HACK: we are forcefully deduplicating the expression here
                            // to convert from a wrapped struct to a raw array, e.g.
                            // `float gl_ClipDistance1 [[clip_distance]] [1];`.
                            if let crate::TypeInner::Array {
                                size: crate::ArraySize::Constant(size),
                                ..
                            } = context.module.types[member.ty].inner
                            {
                                write!(self.out, "{comma} {{")?;
                                for j in 0..size.get() {
                                    if j != 0 {
                                        write!(self.out, ",")?;
                                    }
                                    write!(self.out, "{tmp}.{name}.{WRAPPED_ARRAY_FIELD}[{j}]")?;
                                }
                                write!(self.out, "}}")?;
                            } else {
                                write!(self.out, "{comma} {tmp}.{name}")?;
                            }
                        }
                    }
                    _ => {
                        write!(self.out, "{level}return {struct_name} {{ ")?;
                        self.put_expression(expr_handle, context, true)?;
                    }
                }

                if let FunctionOrigin::EntryPoint(ep_index) = context.origin {
                    let stage = context.module.entry_points[ep_index as usize].stage;
                    if context.pipeline_options.allow_and_force_point_size
                        && stage == crate::ShaderStage::Vertex
                        && !has_point_size
                    {
                        // point size was injected and comes last
                        write!(self.out, ", 1.0")?;
                    }
                }
                write!(self.out, " }}")?;
            }
            None => {
                write!(self.out, "{level}return ")?;
                self.put_expression(expr_handle, context, true)?;
            }
        }
        writeln!(self.out, ";")?;
        Ok(())
    }

    /// Helper method used to find which expressions of a given function require baking
    ///
    /// # Notes
    /// This function overwrites the contents of `self.need_bake_expressions`
    fn update_expressions_to_bake(
        &mut self,
        func: &crate::Function,
        info: &valid::FunctionInfo,
        context: &ExpressionContext,
    ) {
        use crate::Expression;
        self.need_bake_expressions.clear();

        for (expr_handle, expr) in func.expressions.iter() {
            // Expressions whose reference count is above the
            // threshold should always be stored in temporaries.
            let expr_info = &info[expr_handle];
            let min_ref_count = func.expressions[expr_handle].bake_ref_count();
            if min_ref_count <= expr_info.ref_count {
                self.need_bake_expressions.insert(expr_handle);
            } else {
                match expr_info.ty {
                    // force ray desc to be baked: it's used multiple times internally
                    TypeResolution::Handle(h)
                        if Some(h) == context.module.special_types.ray_desc =>
                    {
                        self.need_bake_expressions.insert(expr_handle);
                    }
                    _ => {}
                }
            }

            if let Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                ..
            } = *expr
            {
                match fun {
                    crate::MathFunction::Dot => {
                        // WGSL's `dot` function works on any `vecN` type, but Metal's only
                        // works on floating-point vectors, so we emit inline code for
                        // integer vector `dot` calls. But that code uses each argument `N`
                        // times, once for each component (see `put_dot_product`), so to
                        // avoid duplicated evaluation, we must bake integer operands.

                        // check what kind of product this is depending
                        // on the resolve type of the Dot function itself
                        let inner = context.resolve_type(expr_handle);
                        if let crate::TypeInner::Scalar(scalar) = *inner {
                            match scalar.kind {
                                crate::ScalarKind::Sint | crate::ScalarKind::Uint => {
                                    self.need_bake_expressions.insert(arg);
                                    self.need_bake_expressions.insert(arg1.unwrap());
                                }
                                _ => {}
                            }
                        }
                    }
                    crate::MathFunction::FindMsb
                    | crate::MathFunction::Pack4xI8
                    | crate::MathFunction::Pack4xU8
                    | crate::MathFunction::Unpack4xI8
                    | crate::MathFunction::Unpack4xU8 => {
                        self.need_bake_expressions.insert(arg);
                    }
                    crate::MathFunction::ExtractBits => {
                        // Only argument 1 is re-used.
                        self.need_bake_expressions.insert(arg1.unwrap());
                    }
                    crate::MathFunction::InsertBits => {
                        // Only argument 2 is re-used.
                        self.need_bake_expressions.insert(arg2.unwrap());
                    }
                    crate::MathFunction::Sign => {
                        // WGSL's `sign` function works also on signed ints, but Metal's only
                        // works on floating points, so we emit inline code for integer `sign`
                        // calls. But that code uses each argument 2 times (see `put_isign`),
                        // so to avoid duplicated evaluation, we must bake the argument.
                        let inner = context.resolve_type(expr_handle);
                        if inner.scalar_kind() == Some(crate::ScalarKind::Sint) {
                            self.need_bake_expressions.insert(arg);
                        }
                    }
                    _ => {}
                }
            }
        }
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
                    gctx: context.module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    binding: None,
                    first_time: false,
                };
                write!(self.out, "{ty_name}")?;
            }
            TypeResolution::Value(crate::TypeInner::Scalar(scalar)) => {
                put_numeric_type(&mut self.out, scalar, &[])?;
            }
            TypeResolution::Value(crate::TypeInner::Vector { size, scalar }) => {
                put_numeric_type(&mut self.out, scalar, &[size])?;
            }
            TypeResolution::Value(crate::TypeInner::Matrix {
                columns,
                rows,
                scalar,
            }) => {
                put_numeric_type(&mut self.out, scalar, &[rows, columns])?;
            }
            TypeResolution::Value(ref other) => {
                log::warn!("Type {:?} isn't a known local", other); //TEMP!
                return Err(Error::FeatureNotImplemented("weird local type".to_string()));
            }
        }

        //TODO: figure out the naming scheme that wouldn't collide with user names.
        write!(self.out, " {name} = ")?;

        Ok(())
    }

    /// Cache a clamped level of detail value, if necessary.
    ///
    /// [`ImageLoad`] accesses covered by [`BoundsCheckPolicy::Restrict`] use a
    /// properly clamped level of detail value both in the access itself, and
    /// for fetching the size of the requested MIP level, needed to clamp the
    /// coordinates. To avoid recomputing this clamped level of detail, we cache
    /// it in a temporary variable, as part of the [`Emit`] statement covering
    /// the [`ImageLoad`] expression.
    ///
    /// [`ImageLoad`]: crate::Expression::ImageLoad
    /// [`BoundsCheckPolicy::Restrict`]: index::BoundsCheckPolicy::Restrict
    /// [`Emit`]: crate::Statement::Emit
    fn put_cache_restricted_level(
        &mut self,
        load: Handle<crate::Expression>,
        image: Handle<crate::Expression>,
        mip_level: Option<Handle<crate::Expression>>,
        indent: back::Level,
        context: &StatementContext,
    ) -> BackendResult {
        // Does this image access actually require (or even permit) a
        // level-of-detail, and does the policy require us to restrict it?
        let level_of_detail = match mip_level {
            Some(level) => level,
            None => return Ok(()),
        };

        if context.expression.policies.image_load != index::BoundsCheckPolicy::Restrict
            || !context.expression.image_needs_lod(image)
        {
            return Ok(());
        }

        write!(self.out, "{}uint {} = ", indent, ClampedLod(load),)?;
        self.put_restricted_scalar_image_index(
            image,
            level_of_detail,
            "get_num_mip_levels",
            &context.expression,
        )?;
        writeln!(self.out, ";")?;

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
                        // `ImageLoad` expressions covered by the `Restrict` bounds check policy
                        // may need to cache a clamped version of their level-of-detail argument.
                        if let crate::Expression::ImageLoad {
                            image,
                            level: mip_level,
                            ..
                        } = context.expression.function.expressions[handle]
                        {
                            self.put_cache_restricted_level(
                                handle, image, mip_level, level, context,
                            )?;
                        }

                        let ptr_class = context.expression.resolve_type(handle).pointer_space();
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
                            let bake = if context.expression.guarded_indices.contains(handle) {
                                true
                            } else {
                                self.need_bake_expressions.contains(&handle)
                            };

                            if bake {
                                Some(Baked(handle).to_string())
                            } else {
                                None
                            }
                        };

                        if let Some(name) = expr_name {
                            write!(self.out, "{level}")?;
                            self.start_baking_expression(handle, &context.expression, &name)?;
                            self.put_expression(handle, &context.expression, true)?;
                            self.named_expressions.insert(handle, name);
                            writeln!(self.out, ";")?;
                        }
                    }
                }
                crate::Statement::Block(ref block) => {
                    if !block.is_empty() {
                        writeln!(self.out, "{level}{{")?;
                        self.put_block(level.next(), block, context)?;
                        writeln!(self.out, "{level}}}")?;
                    }
                }
                crate::Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    write!(self.out, "{level}if (")?;
                    self.put_expression(condition, &context.expression, true)?;
                    writeln!(self.out, ") {{")?;
                    self.put_block(level.next(), accept, context)?;
                    if !reject.is_empty() {
                        writeln!(self.out, "{level}}} else {{")?;
                        self.put_block(level.next(), reject, context)?;
                    }
                    writeln!(self.out, "{level}}}")?;
                }
                crate::Statement::Switch {
                    selector,
                    ref cases,
                } => {
                    write!(self.out, "{level}switch(")?;
                    self.put_expression(selector, &context.expression, true)?;
                    writeln!(self.out, ") {{")?;
                    let lcase = level.next();
                    for case in cases.iter() {
                        match case.value {
                            crate::SwitchValue::I32(value) => {
                                write!(self.out, "{lcase}case {value}:")?;
                            }
                            crate::SwitchValue::U32(value) => {
                                write!(self.out, "{lcase}case {value}u:")?;
                            }
                            crate::SwitchValue::Default => {
                                write!(self.out, "{lcase}default:")?;
                            }
                        }

                        let write_block_braces = !(case.fall_through && case.body.is_empty());
                        if write_block_braces {
                            writeln!(self.out, " {{")?;
                        } else {
                            writeln!(self.out)?;
                        }

                        self.put_block(lcase.next(), &case.body, context)?;
                        if !case.fall_through
                            && case.body.last().map_or(true, |s| !s.is_terminator())
                        {
                            writeln!(self.out, "{}break;", lcase.next())?;
                        }

                        if write_block_braces {
                            writeln!(self.out, "{lcase}}}")?;
                        }
                    }
                    writeln!(self.out, "{level}}}")?;
                }
                crate::Statement::Loop {
                    ref body,
                    ref continuing,
                    break_if,
                } => {
                    if !continuing.is_empty() || break_if.is_some() {
                        let gate_name = self.namer.call("loop_init");
                        writeln!(self.out, "{level}bool {gate_name} = true;")?;
                        writeln!(self.out, "{level}while(true) {{")?;
                        let lif = level.next();
                        let lcontinuing = lif.next();
                        writeln!(self.out, "{lif}if (!{gate_name}) {{")?;
                        self.put_block(lcontinuing, continuing, context)?;
                        if let Some(condition) = break_if {
                            write!(self.out, "{lcontinuing}if (")?;
                            self.put_expression(condition, &context.expression, true)?;
                            writeln!(self.out, ") {{")?;
                            writeln!(self.out, "{}break;", lcontinuing.next())?;
                            writeln!(self.out, "{lcontinuing}}}")?;
                        }
                        writeln!(self.out, "{lif}}}")?;
                        writeln!(self.out, "{lif}{gate_name} = false;")?;
                    } else {
                        writeln!(self.out, "{level}while(true) {{")?;
                    }
                    self.put_block(level.next(), body, context)?;
                    writeln!(self.out, "{level}}}")?;
                }
                crate::Statement::Break => {
                    writeln!(self.out, "{level}break;")?;
                }
                crate::Statement::Continue => {
                    writeln!(self.out, "{level}continue;")?;
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
                    writeln!(self.out, "{level}return;")?;
                }
                crate::Statement::Kill => {
                    writeln!(self.out, "{level}{NAMESPACE}::discard_fragment();")?;
                }
                crate::Statement::Barrier(flags) => {
                    self.write_barrier(flags, level)?;
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
                    let address = TexelAddress {
                        coordinate,
                        array_index,
                        sample: None,
                        level: None,
                    };
                    self.put_image_store(level, image, &address, value, context)?
                }
                crate::Statement::Call {
                    function,
                    ref arguments,
                    result,
                } => {
                    write!(self.out, "{level}")?;
                    if let Some(expr) = result {
                        let name = Baked(expr).to_string();
                        self.start_baking_expression(expr, &context.expression, &name)?;
                        self.named_expressions.insert(expr, name);
                    }
                    let fun_name = &self.names[&NameKey::Function(function)];
                    write!(self.out, "{fun_name}(")?;
                    // first, write down the actual arguments
                    for (i, &handle) in arguments.iter().enumerate() {
                        if i != 0 {
                            write!(self.out, ", ")?;
                        }
                        self.put_expression(handle, &context.expression, true)?;
                    }
                    // follow-up with any global resources used
                    let mut separate = !arguments.is_empty();
                    let fun_info = &context.expression.mod_info[function];
                    let mut needs_buffer_sizes = false;
                    for (handle, var) in context.expression.module.global_variables.iter() {
                        if fun_info[handle].is_empty() {
                            continue;
                        }
                        if var.space.needs_pass_through() {
                            let name = &self.names[&NameKey::GlobalVariable(handle)];
                            if separate {
                                write!(self.out, ", ")?;
                            } else {
                                separate = true;
                            }
                            write!(self.out, "{name}")?;
                        }
                        needs_buffer_sizes |=
                            needs_array_length(var.ty, &context.expression.module.types);
                    }
                    if needs_buffer_sizes {
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
                    // This backend supports `SHADER_INT64_ATOMIC_MIN_MAX` but not
                    // `SHADER_INT64_ATOMIC_ALL_OPS`, so we can assume that if `result` is
                    // `Some`, we are not operating on a 64-bit value, and that if we are
                    // operating on a 64-bit value, `result` is `None`.
                    write!(self.out, "{level}")?;
                    let fun_str = if let Some(result) = result {
                        let res_name = Baked(result).to_string();
                        self.start_baking_expression(result, &context.expression, &res_name)?;
                        self.named_expressions.insert(result, res_name);
                        fun.to_msl()?
                    } else if context.expression.resolve_type(value).scalar_width() == Some(8) {
                        fun.to_msl_64_bit()?
                    } else {
                        fun.to_msl()?
                    };

                    self.put_atomic_operation(pointer, fun_str, value, &context.expression)?;
                    // done
                    writeln!(self.out, ";")?;
                }
                crate::Statement::WorkGroupUniformLoad { pointer, result } => {
                    self.write_barrier(crate::Barrier::WORK_GROUP, level)?;

                    write!(self.out, "{level}")?;
                    let name = self.namer.call("");
                    self.start_baking_expression(result, &context.expression, &name)?;
                    self.put_load(pointer, &context.expression, true)?;
                    self.named_expressions.insert(result, name);

                    writeln!(self.out, ";")?;
                    self.write_barrier(crate::Barrier::WORK_GROUP, level)?;
                }
                crate::Statement::RayQuery { query, ref fun } => {
                    if context.expression.lang_version < (2, 4) {
                        return Err(Error::UnsupportedRayTracing);
                    }

                    match *fun {
                        crate::RayQueryFunction::Initialize {
                            acceleration_structure,
                            descriptor,
                        } => {
                            //TODO: how to deal with winding?
                            write!(self.out, "{level}")?;
                            self.put_expression(query, &context.expression, true)?;
                            writeln!(self.out, ".{RAY_QUERY_FIELD_INTERSECTOR}.assume_geometry_type({RT_NAMESPACE}::geometry_type::triangle);")?;
                            {
                                let f_opaque = back::RayFlag::CULL_OPAQUE.bits();
                                let f_no_opaque = back::RayFlag::CULL_NO_OPAQUE.bits();
                                write!(self.out, "{level}")?;
                                self.put_expression(query, &context.expression, true)?;
                                write!(
                                    self.out,
                                    ".{RAY_QUERY_FIELD_INTERSECTOR}.set_opacity_cull_mode(("
                                )?;
                                self.put_expression(descriptor, &context.expression, true)?;
                                write!(self.out, ".flags & {f_opaque}) != 0 ? {RT_NAMESPACE}::opacity_cull_mode::opaque : (")?;
                                self.put_expression(descriptor, &context.expression, true)?;
                                write!(self.out, ".flags & {f_no_opaque}) != 0 ? {RT_NAMESPACE}::opacity_cull_mode::non_opaque : ")?;
                                writeln!(self.out, "{RT_NAMESPACE}::opacity_cull_mode::none);")?;
                            }
                            {
                                let f_opaque = back::RayFlag::OPAQUE.bits();
                                let f_no_opaque = back::RayFlag::NO_OPAQUE.bits();
                                write!(self.out, "{level}")?;
                                self.put_expression(query, &context.expression, true)?;
                                write!(self.out, ".{RAY_QUERY_FIELD_INTERSECTOR}.force_opacity((")?;
                                self.put_expression(descriptor, &context.expression, true)?;
                                write!(self.out, ".flags & {f_opaque}) != 0 ? {RT_NAMESPACE}::forced_opacity::opaque : (")?;
                                self.put_expression(descriptor, &context.expression, true)?;
                                write!(self.out, ".flags & {f_no_opaque}) != 0 ? {RT_NAMESPACE}::forced_opacity::non_opaque : ")?;
                                writeln!(self.out, "{RT_NAMESPACE}::forced_opacity::none);")?;
                            }
                            {
                                let flag = back::RayFlag::TERMINATE_ON_FIRST_HIT.bits();
                                write!(self.out, "{level}")?;
                                self.put_expression(query, &context.expression, true)?;
                                write!(
                                    self.out,
                                    ".{RAY_QUERY_FIELD_INTERSECTOR}.accept_any_intersection(("
                                )?;
                                self.put_expression(descriptor, &context.expression, true)?;
                                writeln!(self.out, ".flags & {flag}) != 0);")?;
                            }

                            write!(self.out, "{level}")?;
                            self.put_expression(query, &context.expression, true)?;
                            write!(self.out, ".{RAY_QUERY_FIELD_INTERSECTION} = ")?;
                            self.put_expression(query, &context.expression, true)?;
                            write!(
                                self.out,
                                ".{RAY_QUERY_FIELD_INTERSECTOR}.intersect({RT_NAMESPACE}::ray("
                            )?;
                            self.put_expression(descriptor, &context.expression, true)?;
                            write!(self.out, ".origin, ")?;
                            self.put_expression(descriptor, &context.expression, true)?;
                            write!(self.out, ".dir, ")?;
                            self.put_expression(descriptor, &context.expression, true)?;
                            write!(self.out, ".tmin, ")?;
                            self.put_expression(descriptor, &context.expression, true)?;
                            write!(self.out, ".tmax), ")?;
                            self.put_expression(acceleration_structure, &context.expression, true)?;
                            write!(self.out, ", ")?;
                            self.put_expression(descriptor, &context.expression, true)?;
                            write!(self.out, ".cull_mask);")?;

                            write!(self.out, "{level}")?;
                            self.put_expression(query, &context.expression, true)?;
                            writeln!(self.out, ".{RAY_QUERY_FIELD_READY} = true;")?;
                        }
                        crate::RayQueryFunction::Proceed { result } => {
                            write!(self.out, "{level}")?;
                            let name = Baked(result).to_string();
                            self.start_baking_expression(result, &context.expression, &name)?;
                            self.named_expressions.insert(result, name);
                            self.put_expression(query, &context.expression, true)?;
                            writeln!(self.out, ".{RAY_QUERY_FIELD_READY};")?;
                            //TODO: actually proceed?

                            write!(self.out, "{level}")?;
                            self.put_expression(query, &context.expression, true)?;
                            writeln!(self.out, ".{RAY_QUERY_FIELD_READY} = false;")?;
                        }
                        crate::RayQueryFunction::Terminate => {
                            write!(self.out, "{level}")?;
                            self.put_expression(query, &context.expression, true)?;
                            writeln!(self.out, ".{RAY_QUERY_FIELD_INTERSECTION}.abort();")?;
                        }
                    }
                }
                crate::Statement::SubgroupBallot { result, predicate } => {
                    write!(self.out, "{level}")?;
                    let name = self.namer.call("");
                    self.start_baking_expression(result, &context.expression, &name)?;
                    self.named_expressions.insert(result, name);
                    write!(self.out, "uint4((uint64_t){NAMESPACE}::simd_ballot(")?;
                    if let Some(predicate) = predicate {
                        self.put_expression(predicate, &context.expression, true)?;
                    } else {
                        write!(self.out, "true")?;
                    }
                    writeln!(self.out, "), 0, 0, 0);")?;
                }
                crate::Statement::SubgroupCollectiveOperation {
                    op,
                    collective_op,
                    argument,
                    result,
                } => {
                    write!(self.out, "{level}")?;
                    let name = self.namer.call("");
                    self.start_baking_expression(result, &context.expression, &name)?;
                    self.named_expressions.insert(result, name);
                    match (collective_op, op) {
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::All) => {
                            write!(self.out, "{NAMESPACE}::simd_all(")?
                        }
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Any) => {
                            write!(self.out, "{NAMESPACE}::simd_any(")?
                        }
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Add) => {
                            write!(self.out, "{NAMESPACE}::simd_sum(")?
                        }
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Mul) => {
                            write!(self.out, "{NAMESPACE}::simd_product(")?
                        }
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Max) => {
                            write!(self.out, "{NAMESPACE}::simd_max(")?
                        }
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Min) => {
                            write!(self.out, "{NAMESPACE}::simd_min(")?
                        }
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::And) => {
                            write!(self.out, "{NAMESPACE}::simd_and(")?
                        }
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Or) => {
                            write!(self.out, "{NAMESPACE}::simd_or(")?
                        }
                        (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Xor) => {
                            write!(self.out, "{NAMESPACE}::simd_xor(")?
                        }
                        (
                            crate::CollectiveOperation::ExclusiveScan,
                            crate::SubgroupOperation::Add,
                        ) => write!(self.out, "{NAMESPACE}::simd_prefix_exclusive_sum(")?,
                        (
                            crate::CollectiveOperation::ExclusiveScan,
                            crate::SubgroupOperation::Mul,
                        ) => write!(self.out, "{NAMESPACE}::simd_prefix_exclusive_product(")?,
                        (
                            crate::CollectiveOperation::InclusiveScan,
                            crate::SubgroupOperation::Add,
                        ) => write!(self.out, "{NAMESPACE}::simd_prefix_inclusive_sum(")?,
                        (
                            crate::CollectiveOperation::InclusiveScan,
                            crate::SubgroupOperation::Mul,
                        ) => write!(self.out, "{NAMESPACE}::simd_prefix_inclusive_product(")?,
                        _ => unimplemented!(),
                    }
                    self.put_expression(argument, &context.expression, true)?;
                    writeln!(self.out, ");")?;
                }
                crate::Statement::SubgroupGather {
                    mode,
                    argument,
                    result,
                } => {
                    write!(self.out, "{level}")?;
                    let name = self.namer.call("");
                    self.start_baking_expression(result, &context.expression, &name)?;
                    self.named_expressions.insert(result, name);
                    match mode {
                        crate::GatherMode::BroadcastFirst => {
                            write!(self.out, "{NAMESPACE}::simd_broadcast_first(")?;
                        }
                        crate::GatherMode::Broadcast(_) => {
                            write!(self.out, "{NAMESPACE}::simd_broadcast(")?;
                        }
                        crate::GatherMode::Shuffle(_) => {
                            write!(self.out, "{NAMESPACE}::simd_shuffle(")?;
                        }
                        crate::GatherMode::ShuffleDown(_) => {
                            write!(self.out, "{NAMESPACE}::simd_shuffle_down(")?;
                        }
                        crate::GatherMode::ShuffleUp(_) => {
                            write!(self.out, "{NAMESPACE}::simd_shuffle_up(")?;
                        }
                        crate::GatherMode::ShuffleXor(_) => {
                            write!(self.out, "{NAMESPACE}::simd_shuffle_xor(")?;
                        }
                    }
                    self.put_expression(argument, &context.expression, true)?;
                    match mode {
                        crate::GatherMode::BroadcastFirst => {}
                        crate::GatherMode::Broadcast(index)
                        | crate::GatherMode::Shuffle(index)
                        | crate::GatherMode::ShuffleDown(index)
                        | crate::GatherMode::ShuffleUp(index)
                        | crate::GatherMode::ShuffleXor(index) => {
                            write!(self.out, ", ")?;
                            self.put_expression(index, &context.expression, true)?;
                        }
                    }
                    writeln!(self.out, ");")?;
                }
            }
        }

        // un-emit expressions
        //TODO: take care of loop/continuing?
        for statement in statements {
            if let crate::Statement::Emit(ref range) = *statement {
                for handle in range.clone() {
                    self.named_expressions.shift_remove(&handle);
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
            writeln!(self.out, "{level}}}")?;
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
        let is_atomic_pointer = context
            .expression
            .resolve_type(pointer)
            .is_atomic_pointer(&context.expression.module.types);

        if is_atomic_pointer {
            write!(
                self.out,
                "{level}{NAMESPACE}::atomic_store_explicit({ATOMIC_REFERENCE}"
            )?;
            self.put_access_chain(pointer, policy, &context.expression)?;
            write!(self.out, ", ")?;
            self.put_expression(value, &context.expression, true)?;
            writeln!(self.out, ", {NAMESPACE}::memory_order_relaxed);")?;
        } else {
            write!(self.out, "{level}")?;
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
        if !module.overrides.is_empty() {
            return Err(Error::Override);
        }

        self.names.clear();
        self.namer.reset(
            module,
            super::keywords::RESERVED,
            &[],
            &[],
            &[CLAMPED_LOD_LOAD_PREFIX],
            &mut self.names,
        );
        self.struct_member_pads.clear();

        writeln!(
            self.out,
            "// language: metal{}.{}",
            options.lang_version.0, options.lang_version.1
        )?;
        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out)?;
        // Work around Metal bug where `uint` is not available by default
        writeln!(self.out, "using {NAMESPACE}::uint;")?;

        let mut uses_ray_query = false;
        for (_, ty) in module.types.iter() {
            match ty.inner {
                crate::TypeInner::AccelerationStructure => {
                    if options.lang_version < (2, 4) {
                        return Err(Error::UnsupportedRayTracing);
                    }
                }
                crate::TypeInner::RayQuery => {
                    if options.lang_version < (2, 4) {
                        return Err(Error::UnsupportedRayTracing);
                    }
                    uses_ray_query = true;
                }
                _ => (),
            }
        }

        if module.special_types.ray_desc.is_some()
            || module.special_types.ray_intersection.is_some()
        {
            if options.lang_version < (2, 4) {
                return Err(Error::UnsupportedRayTracing);
            }
        }

        if uses_ray_query {
            self.put_ray_query_type()?;
        }

        if options
            .bounds_check_policies
            .contains(index::BoundsCheckPolicy::ReadZeroSkipWrite)
        {
            self.put_default_constructible()?;
        }
        writeln!(self.out)?;

        {
            // Make a `Vec` of all the `GlobalVariable`s that contain
            // runtime-sized arrays.
            let globals: Vec<Handle<crate::GlobalVariable>> = module
                .global_variables
                .iter()
                .filter(|&(_, var)| needs_array_length(var.ty, &module.types))
                .map(|(handle, _)| handle)
                .collect();

            let mut buffer_indices = vec![];
            for vbm in &pipeline_options.vertex_buffer_mappings {
                buffer_indices.push(vbm.id);
            }

            if !globals.is_empty() || !buffer_indices.is_empty() {
                writeln!(self.out, "struct _mslBufferSizes {{")?;

                for global in globals {
                    writeln!(
                        self.out,
                        "{}uint {};",
                        back::INDENT,
                        ArraySizeMember(global)
                    )?;
                }

                for idx in buffer_indices {
                    writeln!(self.out, "{}uint buffer_size{};", back::INDENT, idx)?;
                }

                writeln!(self.out, "}};")?;
                writeln!(self.out)?;
            }
        };

        self.write_type_defs(module)?;
        self.write_global_constants(module, info)?;
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
        let tab = back::INDENT;
        writeln!(self.out, "struct DefaultConstructible {{")?;
        writeln!(self.out, "{tab}template<typename T>")?;
        writeln!(self.out, "{tab}operator T() && {{")?;
        writeln!(self.out, "{tab}{tab}return T {{}};")?;
        writeln!(self.out, "{tab}}}")?;
        writeln!(self.out, "}};")?;
        Ok(())
    }

    fn put_ray_query_type(&mut self) -> BackendResult {
        let tab = back::INDENT;
        writeln!(self.out, "struct {RAY_QUERY_TYPE} {{")?;
        let full_type = format!("{RT_NAMESPACE}::intersector<{RT_NAMESPACE}::instancing, {RT_NAMESPACE}::triangle_data, {RT_NAMESPACE}::world_space_data>");
        writeln!(self.out, "{tab}{full_type} {RAY_QUERY_FIELD_INTERSECTOR};")?;
        writeln!(
            self.out,
            "{tab}{full_type}::result_type {RAY_QUERY_FIELD_INTERSECTION};"
        )?;
        writeln!(self.out, "{tab}bool {RAY_QUERY_FIELD_READY} = false;")?;
        writeln!(self.out, "}};")?;
        writeln!(self.out, "constexpr {NAMESPACE}::uint {RAY_QUERY_FUN_MAP_INTERSECTION}(const {RT_NAMESPACE}::intersection_type ty) {{")?;
        let v_triangle = back::RayIntersectionType::Triangle as u32;
        let v_bbox = back::RayIntersectionType::BoundingBox as u32;
        writeln!(
            self.out,
            "{tab}return ty=={RT_NAMESPACE}::intersection_type::triangle ? {v_triangle} : "
        )?;
        writeln!(
            self.out,
            "{tab}{tab}ty=={RT_NAMESPACE}::intersection_type::bounding_box ? {v_bbox} : 0;"
        )?;
        writeln!(self.out, "}}")?;
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
                        gctx: module.to_ctx(),
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        binding: None,
                        first_time: false,
                    };

                    match size {
                        crate::ArraySize::Constant(size) => {
                            writeln!(self.out, "struct {name} {{")?;
                            writeln!(
                                self.out,
                                "{}{} {}[{}];",
                                back::INDENT,
                                base_name,
                                WRAPPED_ARRAY_FIELD,
                                size
                            )?;
                            writeln!(self.out, "}};")?;
                        }
                        crate::ArraySize::Dynamic => {
                            writeln!(self.out, "typedef {base_name} {name}[1];")?;
                        }
                    }
                }
                crate::TypeInner::Struct {
                    ref members, span, ..
                } => {
                    writeln!(self.out, "struct {name} {{")?;
                    let mut last_offset = 0;
                    for (index, member) in members.iter().enumerate() {
                        if member.offset > last_offset {
                            self.struct_member_pads.insert((handle, index as u32));
                            let pad = member.offset - last_offset;
                            writeln!(self.out, "{}char _pad{}[{}];", back::INDENT, index, pad)?;
                        }
                        let ty_inner = &module.types[member.ty].inner;
                        last_offset = member.offset + ty_inner.size(module.to_ctx());

                        let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];

                        // If the member should be packed (as is the case for a misaligned vec3) issue a packed vector
                        match should_pack_struct_member(members, span, index, module) {
                            Some(scalar) => {
                                writeln!(
                                    self.out,
                                    "{}{}::packed_{}3 {};",
                                    back::INDENT,
                                    NAMESPACE,
                                    scalar.to_msl_name(),
                                    member_name
                                )?;
                            }
                            None => {
                                let base_name = TypeContext {
                                    handle: member.ty,
                                    gctx: module.to_ctx(),
                                    names: &self.names,
                                    access: crate::StorageAccess::empty(),
                                    binding: None,
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
                                    scalar,
                                } = *ty_inner
                                {
                                    last_offset += scalar.width as u32;
                                }
                            }
                        }
                    }
                    writeln!(self.out, "}};")?;
                }
                _ => {
                    let ty_name = TypeContext {
                        handle,
                        gctx: module.to_ctx(),
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        binding: None,
                        first_time: true,
                    };
                    writeln!(self.out, "typedef {ty_name} {name};")?;
                }
            }
        }

        // Write functions to create special types.
        for (type_key, struct_ty) in module.special_types.predeclared_types.iter() {
            match type_key {
                &crate::PredeclaredType::ModfResult { size, width }
                | &crate::PredeclaredType::FrexpResult { size, width } => {
                    let arg_type_name_owner;
                    let arg_type_name = if let Some(size) = size {
                        arg_type_name_owner = format!(
                            "{NAMESPACE}::{}{}",
                            if width == 8 { "double" } else { "float" },
                            size as u8
                        );
                        &arg_type_name_owner
                    } else if width == 8 {
                        "double"
                    } else {
                        "float"
                    };

                    let other_type_name_owner;
                    let (defined_func_name, called_func_name, other_type_name) =
                        if matches!(type_key, &crate::PredeclaredType::ModfResult { .. }) {
                            (MODF_FUNCTION, "modf", arg_type_name)
                        } else {
                            let other_type_name = if let Some(size) = size {
                                other_type_name_owner = format!("int{}", size as u8);
                                &other_type_name_owner
                            } else {
                                "int"
                            };
                            (FREXP_FUNCTION, "frexp", other_type_name)
                        };

                    let struct_name = &self.names[&NameKey::Type(*struct_ty)];

                    writeln!(self.out)?;
                    writeln!(
                        self.out,
                        "{} {defined_func_name}({arg_type_name} arg) {{
    {other_type_name} other;
    {arg_type_name} fract = {NAMESPACE}::{called_func_name}(arg, other);
    return {}{{ fract, other }};
}}",
                        struct_name, struct_name
                    )?;
                }
                &crate::PredeclaredType::AtomicCompareExchangeWeakResult { .. } => {}
            }
        }

        Ok(())
    }

    /// Writes all named constants
    fn write_global_constants(
        &mut self,
        module: &crate::Module,
        mod_info: &valid::ModuleInfo,
    ) -> BackendResult {
        let constants = module.constants.iter().filter(|&(_, c)| c.name.is_some());

        for (handle, constant) in constants {
            let ty_name = TypeContext {
                handle: constant.ty,
                gctx: module.to_ctx(),
                names: &self.names,
                access: crate::StorageAccess::empty(),
                binding: None,
                first_time: false,
            };
            let name = &self.names[&NameKey::Constant(handle)];
            write!(self.out, "constant {ty_name} {name} = ")?;
            self.put_const_expression(constant.init, module, mod_info)?;
            writeln!(self.out, ";")?;
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

    fn write_unpacking_function(
        &mut self,
        format: back::msl::VertexFormat,
    ) -> Result<(String, u32, u32), Error> {
        use back::msl::VertexFormat::*;
        match format {
            Uint8x2 => {
                let name = self.namer.call("unpackUint8x2");
                writeln!(
                    self.out,
                    "metal::uint2 {name}(metal::uchar b0, \
                                         metal::uchar b1) {{"
                )?;
                writeln!(self.out, "{}return metal::uint2(b0, b1);", back::INDENT)?;
                writeln!(self.out, "}}")?;
                Ok((name, 2, 2))
            }
            Uint8x4 => {
                let name = self.namer.call("unpackUint8x4");
                writeln!(
                    self.out,
                    "metal::uint4 {name}(metal::uchar b0, \
                                         metal::uchar b1, \
                                         metal::uchar b2, \
                                         metal::uchar b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::uint4(b0, b1, b2, b3);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 4))
            }
            Sint8x2 => {
                let name = self.namer.call("unpackSint8x2");
                writeln!(
                    self.out,
                    "metal::int2 {name}(metal::uchar b0, \
                                        metal::uchar b1) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::int2(as_type<char>(b0), \
                                          as_type<char>(b1));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 2, 2))
            }
            Sint8x4 => {
                let name = self.namer.call("unpackSint8x4");
                writeln!(
                    self.out,
                    "metal::int4 {name}(metal::uchar b0, \
                                        metal::uchar b1, \
                                        metal::uchar b2, \
                                        metal::uchar b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::int4(as_type<char>(b0), \
                                          as_type<char>(b1), \
                                          as_type<char>(b2), \
                                          as_type<char>(b3));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 4))
            }
            Unorm8x2 => {
                let name = self.namer.call("unpackUnorm8x2");
                writeln!(
                    self.out,
                    "metal::float2 {name}(metal::uchar b0, \
                                          metal::uchar b1) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float2(float(b0) / 255.0f, \
                                            float(b1) / 255.0f);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 2, 2))
            }
            Unorm8x4 => {
                let name = self.namer.call("unpackUnorm8x4");
                writeln!(
                    self.out,
                    "metal::float4 {name}(metal::uchar b0, \
                                          metal::uchar b1, \
                                          metal::uchar b2, \
                                          metal::uchar b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float4(float(b0) / 255.0f, \
                                            float(b1) / 255.0f, \
                                            float(b2) / 255.0f, \
                                            float(b3) / 255.0f);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 4))
            }
            Snorm8x2 => {
                let name = self.namer.call("unpackSnorm8x2");
                writeln!(
                    self.out,
                    "metal::float2 {name}(metal::uchar b0, \
                                          metal::uchar b1) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float2(metal::max(-1.0f, as_type<char>(b0) / 127.0f), \
                                            metal::max(-1.0f, as_type<char>(b1) / 127.0f));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 2, 2))
            }
            Snorm8x4 => {
                let name = self.namer.call("unpackSnorm8x4");
                writeln!(
                    self.out,
                    "metal::float4 {name}(metal::uchar b0, \
                                          metal::uchar b1, \
                                          metal::uchar b2, \
                                          metal::uchar b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float4(metal::max(-1.0f, as_type<char>(b0) / 127.0f), \
                                            metal::max(-1.0f, as_type<char>(b1) / 127.0f), \
                                            metal::max(-1.0f, as_type<char>(b2) / 127.0f), \
                                            metal::max(-1.0f, as_type<char>(b3) / 127.0f));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 4))
            }
            Uint16x2 => {
                let name = self.namer.call("unpackUint16x2");
                writeln!(
                    self.out,
                    "metal::uint2 {name}(metal::uint b0, \
                                         metal::uint b1, \
                                         metal::uint b2, \
                                         metal::uint b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::uint2(b1 << 8 | b0, \
                                           b3 << 8 | b2);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 2))
            }
            Uint16x4 => {
                let name = self.namer.call("unpackUint16x4");
                writeln!(
                    self.out,
                    "metal::uint4 {name}(metal::uint b0, \
                                         metal::uint b1, \
                                         metal::uint b2, \
                                         metal::uint b3, \
                                         metal::uint b4, \
                                         metal::uint b5, \
                                         metal::uint b6, \
                                         metal::uint b7) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::uint4(b1 << 8 | b0, \
                                           b3 << 8 | b2, \
                                           b5 << 8 | b4, \
                                           b7 << 8 | b6);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 8, 4))
            }
            Sint16x2 => {
                let name = self.namer.call("unpackSint16x2");
                writeln!(
                    self.out,
                    "metal::int2 {name}(metal::ushort b0, \
                                        metal::ushort b1, \
                                        metal::ushort b2, \
                                        metal::ushort b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::int2(as_type<short>(metal::ushort(b1 << 8 | b0)), \
                                          as_type<short>(metal::ushort(b3 << 8 | b2)));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 2))
            }
            Sint16x4 => {
                let name = self.namer.call("unpackSint16x4");
                writeln!(
                    self.out,
                    "metal::int4 {name}(metal::ushort b0, \
                                        metal::ushort b1, \
                                        metal::ushort b2, \
                                        metal::ushort b3, \
                                        metal::ushort b4, \
                                        metal::ushort b5, \
                                        metal::ushort b6, \
                                        metal::ushort b7) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::int4(as_type<short>(metal::ushort(b1 << 8 | b0)), \
                                          as_type<short>(metal::ushort(b3 << 8 | b2)), \
                                          as_type<short>(metal::ushort(b5 << 8 | b4)), \
                                          as_type<short>(metal::ushort(b7 << 8 | b6)));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 8, 4))
            }
            Unorm16x2 => {
                let name = self.namer.call("unpackUnorm16x2");
                writeln!(
                    self.out,
                    "metal::float2 {name}(metal::ushort b0, \
                                          metal::ushort b1, \
                                          metal::ushort b2, \
                                          metal::ushort b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float2(float(b1 << 8 | b0) / 65535.0f, \
                                            float(b3 << 8 | b2) / 65535.0f);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 2))
            }
            Unorm16x4 => {
                let name = self.namer.call("unpackUnorm16x4");
                writeln!(
                    self.out,
                    "metal::float4 {name}(metal::ushort b0, \
                                          metal::ushort b1, \
                                          metal::ushort b2, \
                                          metal::ushort b3, \
                                          metal::ushort b4, \
                                          metal::ushort b5, \
                                          metal::ushort b6, \
                                          metal::ushort b7) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float4(float(b1 << 8 | b0) / 65535.0f, \
                                            float(b3 << 8 | b2) / 65535.0f, \
                                            float(b5 << 8 | b4) / 65535.0f, \
                                            float(b7 << 8 | b6) / 65535.0f);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 8, 4))
            }
            Snorm16x2 => {
                let name = self.namer.call("unpackSnorm16x2");
                writeln!(
                    self.out,
                    "metal::float2 {name}(metal::ushort b0, \
                                          metal::ushort b1, \
                                          metal::ushort b2, \
                                          metal::ushort b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::unpack_snorm2x16_to_float(b1 << 24 | b0 << 16 | b3 << 8 | b2);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 2))
            }
            Snorm16x4 => {
                let name = self.namer.call("unpackSnorm16x4");
                writeln!(
                    self.out,
                    "metal::float4 {name}(metal::ushort b0, \
                                          metal::ushort b1, \
                                          metal::ushort b2, \
                                          metal::ushort b3, \
                                          metal::ushort b4, \
                                          metal::ushort b5, \
                                          metal::ushort b6, \
                                          metal::ushort b7) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float4(metal::unpack_snorm2x16_to_float(b1 << 24 | b0 << 16 | b3 << 8 | b2), \
                                            metal::unpack_snorm2x16_to_float(b5 << 24 | b4 << 16 | b7 << 8 | b6));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 8, 4))
            }
            Float16x2 => {
                let name = self.namer.call("unpackFloat16x2");
                writeln!(
                    self.out,
                    "metal::float2 {name}(metal::ushort b0, \
                                          metal::ushort b1, \
                                          metal::ushort b2, \
                                          metal::ushort b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float2(as_type<half>(metal::ushort(b1 << 8 | b0)), \
                                            as_type<half>(metal::ushort(b3 << 8 | b2)));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 2))
            }
            Float16x4 => {
                let name = self.namer.call("unpackFloat16x4");
                writeln!(
                    self.out,
                    "metal::float4 {name}(metal::ushort b0, \
                                        metal::ushort b1, \
                                        metal::ushort b2, \
                                        metal::ushort b3, \
                                        metal::ushort b4, \
                                        metal::ushort b5, \
                                        metal::ushort b6, \
                                        metal::ushort b7) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float4(as_type<half>(metal::ushort(b1 << 8 | b0)), \
                                          as_type<half>(metal::ushort(b3 << 8 | b2)), \
                                          as_type<half>(metal::ushort(b5 << 8 | b4)), \
                                          as_type<half>(metal::ushort(b7 << 8 | b6)));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 8, 4))
            }
            Float32 => {
                let name = self.namer.call("unpackFloat32");
                writeln!(
                    self.out,
                    "float {name}(uint b0, \
                                  uint b1, \
                                  uint b2, \
                                  uint b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return as_type<float>(b3 << 24 | b2 << 16 | b1 << 8 | b0);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 1))
            }
            Float32x2 => {
                let name = self.namer.call("unpackFloat32x2");
                writeln!(
                    self.out,
                    "metal::float2 {name}(uint b0, \
                                          uint b1, \
                                          uint b2, \
                                          uint b3, \
                                          uint b4, \
                                          uint b5, \
                                          uint b6, \
                                          uint b7) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float2(as_type<float>(b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                            as_type<float>(b7 << 24 | b6 << 16 | b5 << 8 | b4));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 8, 2))
            }
            Float32x3 => {
                let name = self.namer.call("unpackFloat32x3");
                writeln!(
                    self.out,
                    "metal::float3 {name}(uint b0, \
                                          uint b1, \
                                          uint b2, \
                                          uint b3, \
                                          uint b4, \
                                          uint b5, \
                                          uint b6, \
                                          uint b7, \
                                          uint b8, \
                                          uint b9, \
                                          uint b10, \
                                          uint b11) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float3(as_type<float>(b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                            as_type<float>(b7 << 24 | b6 << 16 | b5 << 8 | b4), \
                                            as_type<float>(b11 << 24 | b10 << 16 | b9 << 8 | b8));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 12, 3))
            }
            Float32x4 => {
                let name = self.namer.call("unpackFloat32x4");
                writeln!(
                    self.out,
                    "metal::float4 {name}(uint b0, \
                                          uint b1, \
                                          uint b2, \
                                          uint b3, \
                                          uint b4, \
                                          uint b5, \
                                          uint b6, \
                                          uint b7, \
                                          uint b8, \
                                          uint b9, \
                                          uint b10, \
                                          uint b11, \
                                          uint b12, \
                                          uint b13, \
                                          uint b14, \
                                          uint b15) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::float4(as_type<float>(b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                            as_type<float>(b7 << 24 | b6 << 16 | b5 << 8 | b4), \
                                            as_type<float>(b11 << 24 | b10 << 16 | b9 << 8 | b8), \
                                            as_type<float>(b15 << 24 | b14 << 16 | b13 << 8 | b12));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 16, 4))
            }
            Uint32 => {
                let name = self.namer.call("unpackUint32");
                writeln!(
                    self.out,
                    "uint {name}(uint b0, \
                                 uint b1, \
                                 uint b2, \
                                 uint b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return (b3 << 24 | b2 << 16 | b1 << 8 | b0);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 1))
            }
            Uint32x2 => {
                let name = self.namer.call("unpackUint32x2");
                writeln!(
                    self.out,
                    "uint2 {name}(uint b0, \
                                  uint b1, \
                                  uint b2, \
                                  uint b3, \
                                  uint b4, \
                                  uint b5, \
                                  uint b6, \
                                  uint b7) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return uint2((b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                    (b7 << 24 | b6 << 16 | b5 << 8 | b4));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 8, 2))
            }
            Uint32x3 => {
                let name = self.namer.call("unpackUint32x3");
                writeln!(
                    self.out,
                    "uint3 {name}(uint b0, \
                                  uint b1, \
                                  uint b2, \
                                  uint b3, \
                                  uint b4, \
                                  uint b5, \
                                  uint b6, \
                                  uint b7, \
                                  uint b8, \
                                  uint b9, \
                                  uint b10, \
                                  uint b11) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return uint3((b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                    (b7 << 24 | b6 << 16 | b5 << 8 | b4), \
                                    (b11 << 24 | b10 << 16 | b9 << 8 | b8));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 12, 3))
            }
            Uint32x4 => {
                let name = self.namer.call("unpackUint32x4");
                writeln!(
                    self.out,
                    "uint4 {name}(uint b0, \
                                  uint b1, \
                                  uint b2, \
                                  uint b3, \
                                  uint b4, \
                                  uint b5, \
                                  uint b6, \
                                  uint b7, \
                                  uint b8, \
                                  uint b9, \
                                  uint b10, \
                                  uint b11, \
                                  uint b12, \
                                  uint b13, \
                                  uint b14, \
                                  uint b15) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return uint4((b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                    (b7 << 24 | b6 << 16 | b5 << 8 | b4), \
                                    (b11 << 24 | b10 << 16 | b9 << 8 | b8), \
                                    (b15 << 24 | b14 << 16 | b13 << 8 | b12));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 16, 4))
            }
            Sint32 => {
                let name = self.namer.call("unpackSint32");
                writeln!(
                    self.out,
                    "int {name}(uint b0, \
                                uint b1, \
                                uint b2, \
                                uint b3) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return as_type<int>(b3 << 24 | b2 << 16 | b1 << 8 | b0);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 1))
            }
            Sint32x2 => {
                let name = self.namer.call("unpackSint32x2");
                writeln!(
                    self.out,
                    "metal::int2 {name}(uint b0, \
                                        uint b1, \
                                        uint b2, \
                                        uint b3, \
                                        uint b4, \
                                        uint b5, \
                                        uint b6, \
                                        uint b7) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::int2(as_type<int>(b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                          as_type<int>(b7 << 24 | b6 << 16 | b5 << 8 | b4));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 8, 2))
            }
            Sint32x3 => {
                let name = self.namer.call("unpackSint32x3");
                writeln!(
                    self.out,
                    "metal::int3 {name}(uint b0, \
                                        uint b1, \
                                        uint b2, \
                                        uint b3, \
                                        uint b4, \
                                        uint b5, \
                                        uint b6, \
                                        uint b7, \
                                        uint b8, \
                                        uint b9, \
                                        uint b10, \
                                        uint b11) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::int3(as_type<int>(b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                          as_type<int>(b7 << 24 | b6 << 16 | b5 << 8 | b4), \
                                          as_type<int>(b11 << 24 | b10 << 16 | b9 << 8 | b8));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 12, 3))
            }
            Sint32x4 => {
                let name = self.namer.call("unpackSint32x4");
                writeln!(
                    self.out,
                    "metal::int4 {name}(uint b0, \
                                        uint b1, \
                                        uint b2, \
                                        uint b3, \
                                        uint b4, \
                                        uint b5, \
                                        uint b6, \
                                        uint b7, \
                                        uint b8, \
                                        uint b9, \
                                        uint b10, \
                                        uint b11, \
                                        uint b12, \
                                        uint b13, \
                                        uint b14, \
                                        uint b15) {{"
                )?;
                writeln!(
                    self.out,
                    "{}return metal::int4(as_type<int>(b3 << 24 | b2 << 16 | b1 << 8 | b0), \
                                          as_type<int>(b7 << 24 | b6 << 16 | b5 << 8 | b4), \
                                          as_type<int>(b11 << 24 | b10 << 16 | b9 << 8 | b8), \
                                          as_type<int>(b15 << 24 | b14 << 16 | b13 << 8 | b12));",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 16, 4))
            }
            Unorm10_10_10_2 => {
                let name = self.namer.call("unpackUnorm10_10_10_2");
                writeln!(
                    self.out,
                    "metal::float4 {name}(uint b0, \
                                          uint b1, \
                                          uint b2, \
                                          uint b3) {{"
                )?;
                writeln!(
                    self.out,
                    // The following is correct for RGBA packing, but our format seems to
                    // match ABGR, which can be fed into the Metal builtin function
                    // unpack_unorm10a2_to_float.
                    /*
                    "{}uint v = (b3 << 24 | b2 << 16 | b1 << 8 | b0); \
                       uint r = (v & 0xFFC00000) >> 22; \
                       uint g = (v & 0x003FF000) >> 12; \
                       uint b = (v & 0x00000FFC) >> 2; \
                       uint a = (v & 0x00000003); \
                       return metal::float4(float(r) / 1023.0f, float(g) / 1023.0f, float(b) / 1023.0f, float(a) / 3.0f);",
                    */
                    "{}return metal::unpack_unorm10a2_to_float(b3 << 24 | b2 << 16 | b1 << 8 | b0);",
                    back::INDENT
                )?;
                writeln!(self.out, "}}")?;
                Ok((name, 4, 4))
            }
        }
    }

    // Returns the array of mapped entry point names.
    fn write_functions(
        &mut self,
        module: &crate::Module,
        mod_info: &valid::ModuleInfo,
        options: &Options,
        pipeline_options: &PipelineOptions,
    ) -> Result<TranslationInfo, Error> {
        use back::msl::VertexFormat;

        // Define structs to hold resolved/generated data for vertex buffers and
        // their attributes.
        struct AttributeMappingResolved {
            ty_name: String,
            dimension: u32,
            ty_is_int: bool,
            name: String,
        }
        let mut am_resolved = FastHashMap::<u32, AttributeMappingResolved>::default();

        struct VertexBufferMappingResolved<'a> {
            id: u32,
            stride: u32,
            indexed_by_vertex: bool,
            ty_name: String,
            param_name: String,
            elem_name: String,
            attributes: &'a Vec<back::msl::AttributeMapping>,
        }
        let mut vbm_resolved = Vec::<VertexBufferMappingResolved>::new();

        // Define a struct to hold a named reference to a byte-unpacking function.
        struct UnpackingFunction {
            name: String,
            byte_count: u32,
            dimension: u32,
        }
        let mut unpacking_functions = FastHashMap::<VertexFormat, UnpackingFunction>::default();

        // Check if we are attempting vertex pulling. If we are, generate some
        // names we'll need, and iterate the vertex buffer mappings to output
        // all the conversion functions we'll need to unpack the attribute data.
        // We can re-use these names for all entry points that need them, since
        // those entry points also use self.namer.
        let mut needs_vertex_id = false;
        let v_id = self.namer.call("v_id");

        let mut needs_instance_id = false;
        let i_id = self.namer.call("i_id");
        if pipeline_options.vertex_pulling_transform {
            for vbm in &pipeline_options.vertex_buffer_mappings {
                let buffer_id = vbm.id;
                let buffer_stride = vbm.stride;

                assert!(
                    buffer_stride > 0,
                    "Vertex pulling requires a non-zero buffer stride."
                );

                if vbm.indexed_by_vertex {
                    needs_vertex_id = true;
                } else {
                    needs_instance_id = true;
                }

                let buffer_ty = self.namer.call(format!("vb_{buffer_id}_type").as_str());
                let buffer_param = self.namer.call(format!("vb_{buffer_id}_in").as_str());
                let buffer_elem = self.namer.call(format!("vb_{buffer_id}_elem").as_str());

                vbm_resolved.push(VertexBufferMappingResolved {
                    id: buffer_id,
                    stride: buffer_stride,
                    indexed_by_vertex: vbm.indexed_by_vertex,
                    ty_name: buffer_ty,
                    param_name: buffer_param,
                    elem_name: buffer_elem,
                    attributes: &vbm.attributes,
                });

                // Iterate the attributes and generate needed unpacking functions.
                for attribute in &vbm.attributes {
                    if unpacking_functions.contains_key(&attribute.format) {
                        continue;
                    }
                    let (name, byte_count, dimension) =
                        match self.write_unpacking_function(attribute.format) {
                            Ok((name, byte_count, dimension)) => (name, byte_count, dimension),
                            _ => {
                                continue;
                            }
                        };
                    unpacking_functions.insert(
                        attribute.format,
                        UnpackingFunction {
                            name,
                            byte_count,
                            dimension,
                        },
                    );
                }
            }
        }

        let mut pass_through_globals = Vec::new();
        for (fun_handle, fun) in module.functions.iter() {
            log::trace!(
                "function {:?}, handle {:?}",
                fun.name.as_deref().unwrap_or("(anonymous)"),
                fun_handle
            );

            let fun_info = &mod_info[fun_handle];
            pass_through_globals.clear();
            let mut needs_buffer_sizes = false;
            for (handle, var) in module.global_variables.iter() {
                if !fun_info[handle].is_empty() {
                    if var.space.needs_pass_through() {
                        pass_through_globals.push(handle);
                    }
                    needs_buffer_sizes |= needs_array_length(var.ty, &module.types);
                }
            }

            writeln!(self.out)?;
            let fun_name = &self.names[&NameKey::Function(fun_handle)];
            match fun.result {
                Some(ref result) => {
                    let ty_name = TypeContext {
                        handle: result.ty,
                        gctx: module.to_ctx(),
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        binding: None,
                        first_time: false,
                    };
                    write!(self.out, "{ty_name}")?;
                }
                None => {
                    write!(self.out, "void")?;
                }
            }
            writeln!(self.out, " {fun_name}(")?;

            for (index, arg) in fun.arguments.iter().enumerate() {
                let name = &self.names[&NameKey::FunctionArgument(fun_handle, index as u32)];
                let param_type_name = TypeContext {
                    handle: arg.ty,
                    gctx: module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    binding: None,
                    first_time: false,
                };
                let separator = separate(
                    !pass_through_globals.is_empty()
                        || index + 1 != fun.arguments.len()
                        || needs_buffer_sizes,
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
                    binding: None,
                    reference: true,
                };
                let separator =
                    separate(index + 1 != pass_through_globals.len() || needs_buffer_sizes);
                write!(self.out, "{}", back::INDENT)?;
                tyvar.try_fmt(&mut self.out)?;
                writeln!(self.out, "{separator}")?;
            }

            if needs_buffer_sizes {
                writeln!(
                    self.out,
                    "{}constant _mslBufferSizes& _buffer_sizes",
                    back::INDENT
                )?;
            }

            writeln!(self.out, ") {{")?;

            let guarded_indices =
                index::find_checked_indexes(module, fun, fun_info, options.bounds_check_policies);

            let context = StatementContext {
                expression: ExpressionContext {
                    function: fun,
                    origin: FunctionOrigin::Handle(fun_handle),
                    info: fun_info,
                    lang_version: options.lang_version,
                    policies: options.bounds_check_policies,
                    guarded_indices,
                    module,
                    mod_info,
                    pipeline_options,
                },
                result_struct: None,
            };

            for (local_handle, local) in fun.local_variables.iter() {
                let ty_name = TypeContext {
                    handle: local.ty,
                    gctx: module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    binding: None,
                    first_time: false,
                };
                let local_name = &self.names[&NameKey::FunctionLocal(fun_handle, local_handle)];
                write!(self.out, "{}{} {}", back::INDENT, ty_name, local_name)?;
                match local.init {
                    Some(value) => {
                        write!(self.out, " = ")?;
                        self.put_expression(value, &context.expression, true)?;
                    }
                    None => {
                        write!(self.out, " = {{}}")?;
                    }
                };
                writeln!(self.out, ";")?;
            }

            self.update_expressions_to_bake(fun, fun_info, &context.expression);
            self.put_block(back::Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
            self.named_expressions.clear();
        }

        let mut info = TranslationInfo {
            entry_point_names: Vec::with_capacity(module.entry_points.len()),
        };
        for (ep_index, ep) in module.entry_points.iter().enumerate() {
            let fun = &ep.function;
            let fun_info = mod_info.get_entry_point(ep_index);
            let mut ep_error = None;

            // For vertex_id and instance_id arguments, presume that we'll
            // use our generated names, but switch to the name of an
            // existing @builtin param, if we find one.
            let mut v_existing_id = None;
            let mut i_existing_id = None;

            log::trace!(
                "entry point {:?}, index {:?}",
                fun.name.as_deref().unwrap_or("(anonymous)"),
                ep_index
            );

            let (em_str, in_mode, out_mode, can_vertex_pull) = match ep.stage {
                crate::ShaderStage::Vertex => (
                    "vertex",
                    LocationMode::VertexInput,
                    LocationMode::VertexOutput,
                    true,
                ),
                crate::ShaderStage::Fragment { .. } => (
                    "fragment",
                    LocationMode::FragmentInput,
                    LocationMode::FragmentOutput,
                    false,
                ),
                crate::ShaderStage::Compute { .. } => (
                    "kernel",
                    LocationMode::Uniform,
                    LocationMode::Uniform,
                    false,
                ),
            };

            // Should this entry point be modified to do vertex pulling?
            let do_vertex_pulling = can_vertex_pull
                && pipeline_options.vertex_pulling_transform
                && !pipeline_options.vertex_buffer_mappings.is_empty();

            // Is any global variable used by this entry point dynamically sized?
            let needs_buffer_sizes = do_vertex_pulling
                || module
                    .global_variables
                    .iter()
                    .filter(|&(handle, _)| !fun_info[handle].is_empty())
                    .any(|(_, var)| needs_array_length(var.ty, &module.types));

            // skip this entry point if any global bindings are missing,
            // or their types are incompatible.
            if !options.fake_missing_bindings {
                for (var_handle, var) in module.global_variables.iter() {
                    if fun_info[var_handle].is_empty() {
                        continue;
                    }
                    match var.space {
                        crate::AddressSpace::Uniform
                        | crate::AddressSpace::Storage { .. }
                        | crate::AddressSpace::Handle => {
                            let br = match var.binding {
                                Some(ref br) => br,
                                None => {
                                    let var_name = var.name.clone().unwrap_or_default();
                                    ep_error =
                                        Some(super::EntryPointError::MissingBinding(var_name));
                                    break;
                                }
                            };
                            let target = options.get_resource_binding_target(ep, br);
                            let good = match target {
                                Some(target) => {
                                    let binding_ty = match module.types[var.ty].inner {
                                        crate::TypeInner::BindingArray { base, .. } => {
                                            &module.types[base].inner
                                        }
                                        ref ty => ty,
                                    };
                                    match *binding_ty {
                                        crate::TypeInner::Image { .. } => target.texture.is_some(),
                                        crate::TypeInner::Sampler { .. } => {
                                            target.sampler.is_some()
                                        }
                                        _ => target.buffer.is_some(),
                                    }
                                }
                                None => false,
                            };
                            if !good {
                                ep_error =
                                    Some(super::EntryPointError::MissingBindTarget(br.clone()));
                                break;
                            }
                        }
                        crate::AddressSpace::PushConstant => {
                            if let Err(e) = options.resolve_push_constants(ep) {
                                ep_error = Some(e);
                                break;
                            }
                        }
                        crate::AddressSpace::Function
                        | crate::AddressSpace::Private
                        | crate::AddressSpace::WorkGroup => {}
                    }
                }
                if needs_buffer_sizes {
                    if let Err(err) = options.resolve_sizes_buffer(ep) {
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

            // Since `Namer.reset` wasn't expecting struct members to be
            // suddenly injected into another namespace like this,
            // `self.names` doesn't keep them distinct from other variables.
            // Generate fresh names for these arguments, and remember the
            // mapping.
            let mut flattened_member_names = FastHashMap::default();
            // Varyings' members get their own namespace
            let mut varyings_namer = proc::Namer::default();

            // List all the Naga `EntryPoint`'s `Function`'s arguments,
            // flattening structs into their members. In Metal, we will pass
            // each of these values to the entry point as a separate argument—
            // except for the varyings, handled next.
            let mut flattened_arguments = Vec::new();
            for (arg_index, arg) in fun.arguments.iter().enumerate() {
                match module.types[arg.ty].inner {
                    crate::TypeInner::Struct { ref members, .. } => {
                        for (member_index, member) in members.iter().enumerate() {
                            let member_index = member_index as u32;
                            flattened_arguments.push((
                                NameKey::StructMember(arg.ty, member_index),
                                member.ty,
                                member.binding.as_ref(),
                            ));
                            let name_key = NameKey::StructMember(arg.ty, member_index);
                            let name = match member.binding {
                                Some(crate::Binding::Location { .. }) => {
                                    if do_vertex_pulling {
                                        self.namer.call(&self.names[&name_key])
                                    } else {
                                        varyings_namer.call(&self.names[&name_key])
                                    }
                                }
                                _ => self.namer.call(&self.names[&name_key]),
                            };
                            flattened_member_names.insert(name_key, name);
                        }
                    }
                    _ => flattened_arguments.push((
                        NameKey::EntryPointArgument(ep_index as _, arg_index as u32),
                        arg.ty,
                        arg.binding.as_ref(),
                    )),
                }
            }

            // Identify the varyings among the argument values, and maybe emit
            // a struct type named `<fun>Input` to hold them. If we are doing
            // vertex pulling, we instead update our attribute mapping to
            // note the types, names, and zero values of the attributes.
            let stage_in_name = format!("{fun_name}Input");
            let varyings_member_name = self.namer.call("varyings");
            let mut has_varyings = false;
            if !flattened_arguments.is_empty() {
                if !do_vertex_pulling {
                    writeln!(self.out, "struct {stage_in_name} {{")?;
                }
                for &(ref name_key, ty, binding) in flattened_arguments.iter() {
                    let (binding, location) = match binding {
                        Some(ref binding @ &crate::Binding::Location { location, .. }) => {
                            (binding, location)
                        }
                        _ => continue,
                    };
                    let name = match *name_key {
                        NameKey::StructMember(..) => &flattened_member_names[name_key],
                        _ => &self.names[name_key],
                    };
                    let ty_name = TypeContext {
                        handle: ty,
                        gctx: module.to_ctx(),
                        names: &self.names,
                        access: crate::StorageAccess::empty(),
                        binding: None,
                        first_time: false,
                    };
                    let resolved = options.resolve_local_binding(binding, in_mode)?;
                    if do_vertex_pulling {
                        // Update our attribute mapping.
                        am_resolved.insert(
                            location,
                            AttributeMappingResolved {
                                ty_name: ty_name.to_string(),
                                dimension: ty_name.vertex_input_dimension(),
                                ty_is_int: ty_name.scalar().map_or(false, scalar_is_int),
                                name: name.to_string(),
                            },
                        );
                    } else {
                        has_varyings = true;
                        write!(self.out, "{}{} {}", back::INDENT, ty_name, name)?;
                        resolved.try_fmt(&mut self.out)?;
                        writeln!(self.out, ";")?;
                    }
                }
                if !do_vertex_pulling {
                    writeln!(self.out, "}};")?;
                }
            }

            // Define a struct type named for the return value, if any, named
            // `<fun>Output`.
            let stage_out_name = format!("{fun_name}Output");
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

                    writeln!(self.out, "struct {stage_out_name} {{")?;
                    let mut has_point_size = false;
                    for (name, ty, binding) in result_members {
                        let ty_name = TypeContext {
                            handle: ty,
                            gctx: module.to_ctx(),
                            names: &self.names,
                            access: crate::StorageAccess::empty(),
                            binding: None,
                            first_time: true,
                        };
                        let binding = binding.ok_or_else(|| {
                            Error::GenericValidation("Expected binding, got None".into())
                        })?;

                        if let crate::Binding::BuiltIn(crate::BuiltIn::PointSize) = *binding {
                            has_point_size = true;
                            if !pipeline_options.allow_and_force_point_size {
                                continue;
                            }
                        }

                        let array_len = match module.types[ty].inner {
                            crate::TypeInner::Array {
                                size: crate::ArraySize::Constant(size),
                                ..
                            } => Some(size),
                            _ => None,
                        };
                        let resolved = options.resolve_local_binding(binding, out_mode)?;
                        write!(self.out, "{}{} {}", back::INDENT, ty_name, name)?;
                        if let Some(array_len) = array_len {
                            write!(self.out, " [{array_len}]")?;
                        }
                        resolved.try_fmt(&mut self.out)?;
                        writeln!(self.out, ";")?;
                    }

                    if pipeline_options.allow_and_force_point_size
                        && ep.stage == crate::ShaderStage::Vertex
                        && !has_point_size
                    {
                        // inject the point size output last
                        writeln!(
                            self.out,
                            "{}float _point_size [[point_size]];",
                            back::INDENT
                        )?;
                    }
                    writeln!(self.out, "}};")?;
                    &stage_out_name
                }
                None => "void",
            };

            // If we're doing a vertex pulling transform, define the buffer
            // structure types.
            if do_vertex_pulling {
                for vbm in &vbm_resolved {
                    let buffer_stride = vbm.stride;
                    let buffer_ty = &vbm.ty_name;

                    // Define a structure of bytes of the appropriate size.
                    // When we access the attributes, we'll be unpacking these
                    // bytes at some offset.
                    writeln!(
                        self.out,
                        "struct {buffer_ty} {{ metal::uchar data[{buffer_stride}]; }};"
                    )?;
                }
            }

            // Write the entry point function's name, and begin its argument list.
            writeln!(self.out, "{em_str} {result_type_name} {fun_name}(")?;
            let mut is_first_argument = true;

            // If we have produced a struct holding the `EntryPoint`'s
            // `Function`'s arguments' varyings, pass that struct first.
            if has_varyings {
                writeln!(
                    self.out,
                    "  {stage_in_name} {varyings_member_name} [[stage_in]]"
                )?;
                is_first_argument = false;
            }

            let mut local_invocation_id = None;

            // Then pass the remaining arguments not included in the varyings
            // struct.
            for &(ref name_key, ty, binding) in flattened_arguments.iter() {
                let binding = match binding {
                    Some(binding @ &crate::Binding::BuiltIn { .. }) => binding,
                    _ => continue,
                };
                let name = match *name_key {
                    NameKey::StructMember(..) => &flattened_member_names[name_key],
                    _ => &self.names[name_key],
                };

                if binding == &crate::Binding::BuiltIn(crate::BuiltIn::LocalInvocationId) {
                    local_invocation_id = Some(name_key);
                }

                let ty_name = TypeContext {
                    handle: ty,
                    gctx: module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    binding: None,
                    first_time: false,
                };

                match *binding {
                    crate::Binding::BuiltIn(crate::BuiltIn::VertexIndex) => {
                        v_existing_id = Some(name.clone());
                    }
                    crate::Binding::BuiltIn(crate::BuiltIn::InstanceIndex) => {
                        i_existing_id = Some(name.clone());
                    }
                    _ => {}
                };

                let resolved = options.resolve_local_binding(binding, in_mode)?;
                let separator = if is_first_argument {
                    is_first_argument = false;
                    ' '
                } else {
                    ','
                };
                write!(self.out, "{separator} {ty_name} {name}")?;
                resolved.try_fmt(&mut self.out)?;
                writeln!(self.out)?;
            }

            let need_workgroup_variables_initialization =
                self.need_workgroup_variables_initialization(options, ep, module, fun_info);

            if need_workgroup_variables_initialization && local_invocation_id.is_none() {
                let separator = if is_first_argument {
                    is_first_argument = false;
                    ' '
                } else {
                    ','
                };
                writeln!(
                    self.out,
                    "{separator} {NAMESPACE}::uint3 __local_invocation_id [[thread_position_in_threadgroup]]"
                )?;
            }

            // Those global variables used by this entry point and its callees
            // get passed as arguments. `Private` globals are an exception, they
            // don't outlive this invocation, so we declare them below as locals
            // within the entry point.
            for (handle, var) in module.global_variables.iter() {
                let usage = fun_info[handle];
                if usage.is_empty() || var.space == crate::AddressSpace::Private {
                    continue;
                }

                if options.lang_version < (1, 2) {
                    match var.space {
                        // This restriction is not documented in the MSL spec
                        // but validation will fail if it is not upheld.
                        //
                        // We infer the required version from the "Function
                        // Buffer Read-Writes" section of [what's new], where
                        // the feature sets listed correspond with the ones
                        // supporting MSL 1.2.
                        //
                        // [what's new]: https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/WhatsNewiniOS10tvOS10andOSX1012/WhatsNewiniOS10tvOS10andOSX1012.html
                        crate::AddressSpace::Storage { access }
                            if access.contains(crate::StorageAccess::STORE)
                                && ep.stage == crate::ShaderStage::Fragment =>
                        {
                            return Err(Error::UnsupportedWriteableStorageBuffer)
                        }
                        crate::AddressSpace::Handle => {
                            match module.types[var.ty].inner {
                                crate::TypeInner::Image {
                                    class: crate::ImageClass::Storage { access, .. },
                                    ..
                                } => {
                                    // This restriction is not documented in the MSL spec
                                    // but validation will fail if it is not upheld.
                                    //
                                    // We infer the required version from the "Function
                                    // Texture Read-Writes" section of [what's new], where
                                    // the feature sets listed correspond with the ones
                                    // supporting MSL 1.2.
                                    //
                                    // [what's new]: https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/WhatsNewiniOS10tvOS10andOSX1012/WhatsNewiniOS10tvOS10andOSX1012.html
                                    if access.contains(crate::StorageAccess::STORE)
                                        && (ep.stage == crate::ShaderStage::Vertex
                                            || ep.stage == crate::ShaderStage::Fragment)
                                    {
                                        return Err(Error::UnsupportedWriteableStorageTexture(
                                            ep.stage,
                                        ));
                                    }

                                    if access.contains(
                                        crate::StorageAccess::LOAD | crate::StorageAccess::STORE,
                                    ) {
                                        return Err(Error::UnsupportedRWStorageTexture);
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }

                // Check min MSL version for binding arrays
                match var.space {
                    crate::AddressSpace::Handle => match module.types[var.ty].inner {
                        crate::TypeInner::BindingArray { base, .. } => {
                            match module.types[base].inner {
                                crate::TypeInner::Sampler { .. } => {
                                    if options.lang_version < (2, 0) {
                                        return Err(Error::UnsupportedArrayOf(
                                            "samplers".to_string(),
                                        ));
                                    }
                                }
                                crate::TypeInner::Image { class, .. } => match class {
                                    crate::ImageClass::Sampled { .. }
                                    | crate::ImageClass::Depth { .. }
                                    | crate::ImageClass::Storage {
                                        access: crate::StorageAccess::LOAD,
                                        ..
                                    } => {
                                        // Array of textures since:
                                        // - iOS: Metal 1.2 (check depends on https://github.com/gfx-rs/naga/issues/2164)
                                        // - macOS: Metal 2

                                        if options.lang_version < (2, 0) {
                                            return Err(Error::UnsupportedArrayOf(
                                                "textures".to_string(),
                                            ));
                                        }
                                    }
                                    crate::ImageClass::Storage {
                                        access: crate::StorageAccess::STORE,
                                        ..
                                    } => {
                                        // Array of write-only textures since:
                                        // - iOS: Metal 2.2 (check depends on https://github.com/gfx-rs/naga/issues/2164)
                                        // - macOS: Metal 2

                                        if options.lang_version < (2, 0) {
                                            return Err(Error::UnsupportedArrayOf(
                                                "write-only textures".to_string(),
                                            ));
                                        }
                                    }
                                    crate::ImageClass::Storage { .. } => {
                                        return Err(Error::UnsupportedArrayOf(
                                            "read-write textures".to_string(),
                                        ));
                                    }
                                },
                                _ => {
                                    return Err(Error::UnsupportedArrayOfType(base));
                                }
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }

                // the resolves have already been checked for `!fake_missing_bindings` case
                let resolved = match var.space {
                    crate::AddressSpace::PushConstant => options.resolve_push_constants(ep).ok(),
                    crate::AddressSpace::WorkGroup => None,
                    _ => options
                        .resolve_resource_binding(ep, var.binding.as_ref().unwrap())
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
                    binding: resolved.as_ref(),
                    reference: true,
                };
                let separator = if is_first_argument {
                    is_first_argument = false;
                    ' '
                } else {
                    ','
                };
                write!(self.out, "{separator} ")?;
                tyvar.try_fmt(&mut self.out)?;
                if let Some(resolved) = resolved {
                    resolved.try_fmt(&mut self.out)?;
                }
                if let Some(value) = var.init {
                    write!(self.out, " = ")?;
                    self.put_const_expression(value, module, mod_info)?;
                }
                writeln!(self.out)?;
            }

            if do_vertex_pulling {
                assert!(needs_vertex_id || needs_instance_id);

                let mut separator = if is_first_argument {
                    is_first_argument = false;
                    ' '
                } else {
                    ','
                };

                if needs_vertex_id && v_existing_id.is_none() {
                    // Write the [[vertex_id]] argument.
                    writeln!(self.out, "{separator} uint {v_id} [[vertex_id]]")?;
                    separator = ',';
                }

                if needs_instance_id && i_existing_id.is_none() {
                    writeln!(self.out, "{separator} uint {i_id} [[instance_id]]")?;
                }

                // Iterate vbm_resolved, output one argument for every vertex buffer,
                // using the names we generated earlier.
                for vbm in &vbm_resolved {
                    let id = &vbm.id;
                    let ty_name = &vbm.ty_name;
                    let param_name = &vbm.param_name;
                    writeln!(
                        self.out,
                        ", const device {ty_name}* {param_name} [[buffer({id})]]"
                    )?;
                }
            }

            // If this entry uses any variable-length arrays, their sizes are
            // passed as a final struct-typed argument.
            if needs_buffer_sizes {
                // this is checked earlier
                let resolved = options.resolve_sizes_buffer(ep).unwrap();
                let separator = if is_first_argument { ' ' } else { ',' };
                write!(
                    self.out,
                    "{separator} constant _mslBufferSizes& _buffer_sizes",
                )?;
                resolved.try_fmt(&mut self.out)?;
                writeln!(self.out)?;
            }

            // end of the entry point argument list
            writeln!(self.out, ") {{")?;

            // Starting the function body.
            if do_vertex_pulling {
                // Provide zero values for all the attributes, which we will overwrite with
                // real data from the vertex attribute buffers, if the indices are in-bounds.
                for vbm in &vbm_resolved {
                    for attribute in vbm.attributes {
                        let location = attribute.shader_location;
                        let am_option = am_resolved.get(&location);
                        if am_option.is_none() {
                            // This bound attribute isn't used in this entry point, so
                            // don't bother zero-initializing it.
                            continue;
                        }
                        let am = am_option.unwrap();
                        let attribute_ty_name = &am.ty_name;
                        let attribute_name = &am.name;

                        writeln!(
                            self.out,
                            "{}{attribute_ty_name} {attribute_name} = {{}};",
                            back::Level(1)
                        )?;
                    }

                    // Output a bounds check block that will set real values for the
                    // attributes, if the bounds are satisfied.
                    write!(self.out, "{}if (", back::Level(1))?;

                    let idx = &vbm.id;
                    let stride = &vbm.stride;
                    let index_name = if vbm.indexed_by_vertex {
                        if let Some(ref name) = v_existing_id {
                            name
                        } else {
                            &v_id
                        }
                    } else if let Some(ref name) = i_existing_id {
                        name
                    } else {
                        &i_id
                    };
                    write!(
                        self.out,
                        "{index_name} < (_buffer_sizes.buffer_size{idx} / {stride})"
                    )?;

                    writeln!(self.out, ") {{")?;

                    // Pull the bytes out of the vertex buffer.
                    let ty_name = &vbm.ty_name;
                    let elem_name = &vbm.elem_name;
                    let param_name = &vbm.param_name;

                    writeln!(
                        self.out,
                        "{}const {ty_name} {elem_name} = {param_name}[{index_name}];",
                        back::Level(2),
                    )?;

                    // Now set real values for each of the attributes, by unpacking the data
                    // from the buffer elements.
                    for attribute in vbm.attributes {
                        let location = attribute.shader_location;
                        let am_option = am_resolved.get(&location);
                        if am_option.is_none() {
                            // This bound attribute isn't used in this entry point, so
                            // don't bother extracting the data. Too bad we emitted the
                            // unpacking function earlier -- it might not get used.
                            continue;
                        }
                        let am = am_option.unwrap();
                        let attribute_name = &am.name;
                        let attribute_ty_name = &am.ty_name;

                        let offset = attribute.offset;
                        let func = unpacking_functions
                            .get(&attribute.format)
                            .expect("Should have generated this unpacking function earlier.");
                        let func_name = &func.name;

                        write!(self.out, "{}{attribute_name} = ", back::Level(2),)?;

                        // Check dimensionality of the attribute compared to the unpacking
                        // function. If attribute dimension is < unpack dimension, then
                        // we need to explicitly cast down the result. Otherwise, if attribute
                        // dimension > unpack dimension, we have to pad out the unpack value
                        // from a vec4(0, 0, 0, 1) of matching scalar type.

                        let needs_truncate_or_padding = am.dimension != func.dimension;
                        if needs_truncate_or_padding {
                            write!(self.out, "{attribute_ty_name}(")?;
                        }

                        write!(self.out, "{func_name}({elem_name}.data[{offset}]",)?;
                        for i in (offset + 1)..(offset + func.byte_count) {
                            write!(self.out, ", {elem_name}.data[{i}]")?;
                        }
                        write!(self.out, ")")?;

                        if needs_truncate_or_padding {
                            let zero_value = if am.ty_is_int { "0" } else { "0.0" };
                            let one_value = if am.ty_is_int { "1" } else { "1.0" };
                            for i in func.dimension..am.dimension {
                                write!(
                                    self.out,
                                    ", {}",
                                    if i == 3 { one_value } else { zero_value }
                                )?;
                            }
                            write!(self.out, ")")?;
                        }

                        writeln!(self.out, ";")?;
                    }

                    // End the bounds check / attribute setting block.
                    writeln!(self.out, "{}}}", back::Level(1))?;
                }
            }

            if need_workgroup_variables_initialization {
                self.write_workgroup_variables_initialization(
                    module,
                    mod_info,
                    fun_info,
                    local_invocation_id,
                )?;
            }

            // Metal doesn't support private mutable variables outside of functions,
            // so we put them here, just like the locals.
            for (handle, var) in module.global_variables.iter() {
                let usage = fun_info[handle];
                if usage.is_empty() {
                    continue;
                }
                if var.space == crate::AddressSpace::Private {
                    let tyvar = TypedGlobalVariable {
                        module,
                        names: &self.names,
                        handle,
                        usage,
                        binding: None,
                        reference: false,
                    };
                    write!(self.out, "{}", back::INDENT)?;
                    tyvar.try_fmt(&mut self.out)?;
                    match var.init {
                        Some(value) => {
                            write!(self.out, " = ")?;
                            self.put_const_expression(value, module, mod_info)?;
                            writeln!(self.out, ";")?;
                        }
                        None => {
                            writeln!(self.out, " = {{}};")?;
                        }
                    };
                } else if let Some(ref binding) = var.binding {
                    // write an inline sampler
                    let resolved = options.resolve_resource_binding(ep, binding).unwrap();
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

            // Now take the arguments that we gathered into structs, and the
            // structs that we flattened into arguments, and emit local
            // variables with initializers that put everything back the way the
            // body code expects.
            //
            // If we had to generate fresh names for struct members passed as
            // arguments, be sure to use those names when rebuilding the struct.
            //
            // "Each day, I change some zeros to ones, and some ones to zeros.
            // The rest, I leave alone."
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
                            let key = NameKey::StructMember(arg.ty, member_index as u32);
                            let name = &flattened_member_names[&key];
                            if member_index != 0 {
                                write!(self.out, ", ")?;
                            }
                            // insert padding initialization, if needed
                            if self
                                .struct_member_pads
                                .contains(&(arg.ty, member_index as u32))
                            {
                                write!(self.out, "{{}}, ")?;
                            }
                            if let Some(crate::Binding::Location { .. }) = member.binding {
                                if has_varyings {
                                    write!(self.out, "{varyings_member_name}.")?;
                                }
                            }
                            write!(self.out, "{name}")?;
                        }
                        writeln!(self.out, " }};")?;
                    }
                    _ => {
                        if let Some(crate::Binding::Location { .. }) = arg.binding {
                            if has_varyings {
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
            }

            let guarded_indices =
                index::find_checked_indexes(module, fun, fun_info, options.bounds_check_policies);

            let context = StatementContext {
                expression: ExpressionContext {
                    function: fun,
                    origin: FunctionOrigin::EntryPoint(ep_index as _),
                    info: fun_info,
                    lang_version: options.lang_version,
                    policies: options.bounds_check_policies,
                    guarded_indices,
                    module,
                    mod_info,
                    pipeline_options,
                },
                result_struct: Some(&stage_out_name),
            };

            // Finally, declare all the local variables that we need
            //TODO: we can postpone this till the relevant expressions are emitted
            for (local_handle, local) in fun.local_variables.iter() {
                let name = &self.names[&NameKey::EntryPointLocal(ep_index as _, local_handle)];
                let ty_name = TypeContext {
                    handle: local.ty,
                    gctx: module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    binding: None,
                    first_time: false,
                };
                write!(self.out, "{}{} {}", back::INDENT, ty_name, name)?;
                match local.init {
                    Some(value) => {
                        write!(self.out, " = ")?;
                        self.put_expression(value, &context.expression, true)?;
                    }
                    None => {
                        write!(self.out, " = {{}}")?;
                    }
                };
                writeln!(self.out, ";")?;
            }

            self.update_expressions_to_bake(fun, fun_info, &context.expression);
            self.put_block(back::Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
            if ep_index + 1 != module.entry_points.len() {
                writeln!(self.out)?;
            }
            self.named_expressions.clear();
        }

        Ok(info)
    }

    fn write_barrier(&mut self, flags: crate::Barrier, level: back::Level) -> BackendResult {
        // Note: OR-ring bitflags requires `__HAVE_MEMFLAG_OPERATORS__`,
        // so we try to avoid it here.
        if flags.is_empty() {
            writeln!(
                self.out,
                "{level}{NAMESPACE}::threadgroup_barrier({NAMESPACE}::mem_flags::mem_none);",
            )?;
        }
        if flags.contains(crate::Barrier::STORAGE) {
            writeln!(
                self.out,
                "{level}{NAMESPACE}::threadgroup_barrier({NAMESPACE}::mem_flags::mem_device);",
            )?;
        }
        if flags.contains(crate::Barrier::WORK_GROUP) {
            writeln!(
                self.out,
                "{level}{NAMESPACE}::threadgroup_barrier({NAMESPACE}::mem_flags::mem_threadgroup);",
            )?;
        }
        if flags.contains(crate::Barrier::SUB_GROUP) {
            writeln!(
                self.out,
                "{level}{NAMESPACE}::simdgroup_barrier({NAMESPACE}::mem_flags::mem_threadgroup);",
            )?;
        }
        Ok(())
    }
}

/// Initializing workgroup variables is more tricky for Metal because we have to deal
/// with atomics at the type-level (which don't have a copy constructor).
mod workgroup_mem_init {
    use crate::EntryPoint;

    use super::*;

    enum Access {
        GlobalVariable(Handle<crate::GlobalVariable>),
        StructMember(Handle<crate::Type>, u32),
        Array(usize),
    }

    impl Access {
        fn write<W: Write>(
            &self,
            writer: &mut W,
            names: &FastHashMap<NameKey, String>,
        ) -> Result<(), core::fmt::Error> {
            match *self {
                Access::GlobalVariable(handle) => {
                    write!(writer, "{}", &names[&NameKey::GlobalVariable(handle)])
                }
                Access::StructMember(handle, index) => {
                    write!(writer, ".{}", &names[&NameKey::StructMember(handle, index)])
                }
                Access::Array(depth) => write!(writer, ".{WRAPPED_ARRAY_FIELD}[__i{depth}]"),
            }
        }
    }

    struct AccessStack {
        stack: Vec<Access>,
        array_depth: usize,
    }

    impl AccessStack {
        const fn new() -> Self {
            Self {
                stack: Vec::new(),
                array_depth: 0,
            }
        }

        fn enter_array<R>(&mut self, cb: impl FnOnce(&mut Self, usize) -> R) -> R {
            let array_depth = self.array_depth;
            self.stack.push(Access::Array(array_depth));
            self.array_depth += 1;
            let res = cb(self, array_depth);
            self.stack.pop();
            self.array_depth -= 1;
            res
        }

        fn enter<R>(&mut self, new: Access, cb: impl FnOnce(&mut Self) -> R) -> R {
            self.stack.push(new);
            let res = cb(self);
            self.stack.pop();
            res
        }

        fn write<W: Write>(
            &self,
            writer: &mut W,
            names: &FastHashMap<NameKey, String>,
        ) -> Result<(), core::fmt::Error> {
            for next in self.stack.iter() {
                next.write(writer, names)?;
            }
            Ok(())
        }
    }

    impl<W: Write> Writer<W> {
        pub(super) fn need_workgroup_variables_initialization(
            &mut self,
            options: &Options,
            ep: &EntryPoint,
            module: &crate::Module,
            fun_info: &valid::FunctionInfo,
        ) -> bool {
            options.zero_initialize_workgroup_memory
                && ep.stage == crate::ShaderStage::Compute
                && module.global_variables.iter().any(|(handle, var)| {
                    !fun_info[handle].is_empty() && var.space == crate::AddressSpace::WorkGroup
                })
        }

        pub(super) fn write_workgroup_variables_initialization(
            &mut self,
            module: &crate::Module,
            module_info: &valid::ModuleInfo,
            fun_info: &valid::FunctionInfo,
            local_invocation_id: Option<&NameKey>,
        ) -> BackendResult {
            let level = back::Level(1);

            writeln!(
                self.out,
                "{}if ({}::all({} == {}::uint3(0u))) {{",
                level,
                NAMESPACE,
                local_invocation_id
                    .map(|name_key| self.names[name_key].as_str())
                    .unwrap_or("__local_invocation_id"),
                NAMESPACE,
            )?;

            let mut access_stack = AccessStack::new();

            let vars = module.global_variables.iter().filter(|&(handle, var)| {
                !fun_info[handle].is_empty() && var.space == crate::AddressSpace::WorkGroup
            });

            for (handle, var) in vars {
                access_stack.enter(Access::GlobalVariable(handle), |access_stack| {
                    self.write_workgroup_variable_initialization(
                        module,
                        module_info,
                        var.ty,
                        access_stack,
                        level.next(),
                    )
                })?;
            }

            writeln!(self.out, "{level}}}")?;
            self.write_barrier(crate::Barrier::WORK_GROUP, level)
        }

        fn write_workgroup_variable_initialization(
            &mut self,
            module: &crate::Module,
            module_info: &valid::ModuleInfo,
            ty: Handle<crate::Type>,
            access_stack: &mut AccessStack,
            level: back::Level,
        ) -> BackendResult {
            if module_info[ty].contains(valid::TypeFlags::CONSTRUCTIBLE) {
                write!(self.out, "{level}")?;
                access_stack.write(&mut self.out, &self.names)?;
                writeln!(self.out, " = {{}};")?;
            } else {
                match module.types[ty].inner {
                    crate::TypeInner::Atomic { .. } => {
                        write!(
                            self.out,
                            "{level}{NAMESPACE}::atomic_store_explicit({ATOMIC_REFERENCE}"
                        )?;
                        access_stack.write(&mut self.out, &self.names)?;
                        writeln!(self.out, ", 0, {NAMESPACE}::memory_order_relaxed);")?;
                    }
                    crate::TypeInner::Array { base, size, .. } => {
                        let count = match size.to_indexable_length(module).expect("Bad array size")
                        {
                            proc::IndexableLength::Known(count) => count,
                            proc::IndexableLength::Dynamic => unreachable!(),
                        };

                        access_stack.enter_array(|access_stack, array_depth| {
                            writeln!(
                                self.out,
                                "{level}for (int __i{array_depth} = 0; __i{array_depth} < {count}; __i{array_depth}++) {{"
                            )?;
                            self.write_workgroup_variable_initialization(
                                module,
                                module_info,
                                base,
                                access_stack,
                                level.next(),
                            )?;
                            writeln!(self.out, "{level}}}")?;
                            BackendResult::Ok(())
                        })?;
                    }
                    crate::TypeInner::Struct { ref members, .. } => {
                        for (index, member) in members.iter().enumerate() {
                            access_stack.enter(
                                Access::StructMember(ty, index as u32),
                                |access_stack| {
                                    self.write_workgroup_variable_initialization(
                                        module,
                                        module_info,
                                        member.ty,
                                        access_stack,
                                        level,
                                    )
                                },
                            )?;
                        }
                    }
                    _ => unreachable!(),
                }
            }

            Ok(())
        }
    }
}

#[test]
fn test_stack_size() {
    use crate::valid::{Capabilities, ValidationFlags};
    // create a module with at least one expression nested
    let mut module = crate::Module::default();
    let mut fun = crate::Function::default();
    let const_expr = fun.expressions.append(
        crate::Expression::Literal(crate::Literal::F32(1.0)),
        Default::default(),
    );
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
    let info = valid::Validator::new(ValidationFlags::empty(), Capabilities::empty())
        .validate(&module)
        .unwrap();
    // process the module
    let mut writer = Writer::new(String::new());
    writer
        .write(&module, &info, &Default::default(), &Default::default())
        .unwrap();

    {
        // check expression stack
        let mut addresses_start = usize::MAX;
        let mut addresses_end = 0usize;
        for pointer in writer.put_expression_stack_pointers {
            addresses_start = addresses_start.min(pointer as usize);
            addresses_end = addresses_end.max(pointer as usize);
        }
        let stack_size = addresses_end - addresses_start;
        // check the size (in debug only)
        // last observed macOS value: 20528 (CI)
        if !(11000..=25000).contains(&stack_size) {
            panic!("`put_expression` stack size {stack_size} has changed!");
        }
    }

    {
        // check block stack
        let mut addresses_start = usize::MAX;
        let mut addresses_end = 0usize;
        for pointer in writer.put_block_stack_pointers {
            addresses_start = addresses_start.min(pointer as usize);
            addresses_end = addresses_end.max(pointer as usize);
        }
        let stack_size = addresses_end - addresses_start;
        // check the size (in debug only)
        // last observed macOS value: 22256 (CI)
        if !(15000..=25000).contains(&stack_size) {
            panic!("`put_block` stack size {stack_size} has changed!");
        }
    }
}

impl crate::AtomicFunction {
    fn to_msl(self) -> Result<&'static str, Error> {
        Ok(match self {
            Self::Add => "fetch_add",
            Self::Subtract => "fetch_sub",
            Self::And => "fetch_and",
            Self::InclusiveOr => "fetch_or",
            Self::ExclusiveOr => "fetch_xor",
            Self::Min => "fetch_min",
            Self::Max => "fetch_max",
            Self::Exchange { compare: None } => "exchange",
            Self::Exchange { compare: Some(_) } => Err(Error::FeatureNotImplemented(
                "atomic CompareExchange".to_string(),
            ))?,
        })
    }

    fn to_msl_64_bit(self) -> Result<&'static str, Error> {
        Ok(match self {
            Self::Min => "min",
            Self::Max => "max",
            _ => Err(Error::FeatureNotImplemented(
                "64-bit atomic operation other than min/max".to_string(),
            ))?,
        })
    }
}
