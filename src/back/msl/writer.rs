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
    kind: crate::ScalarKind,
    sizes: &[crate::VectorSize],
) -> Result<(), FmtError> {
    match (kind, sizes) {
        (kind, &[]) => {
            write!(out, "{}", kind.to_msl_name())
        }
        (kind, &[rows]) => {
            write!(
                out,
                "{}::{}{}",
                NAMESPACE,
                kind.to_msl_name(),
                back::vector_size_str(rows)
            )
        }
        (kind, &[rows, columns]) => {
            write!(
                out,
                "{}::{}{}x{}",
                NAMESPACE,
                kind.to_msl_name(),
                back::vector_size_str(columns),
                back::vector_size_str(rows)
            )
        }
        (_, _) => Ok(()), // not meaningful
    }
}

/// Prefix for cached clamped level-of-detail values for `ImageLoad` expressions.
const CLAMPED_LOD_LOAD_PREFIX: &str = "clamped_lod_e";

struct TypeContext<'a> {
    handle: Handle<crate::Type>,
    gctx: proc::GlobalCtx<'a>,
    names: &'a FastHashMap<NameKey, String>,
    access: crate::StorageAccess,
    binding: Option<&'a super::ResolvedBinding>,
    first_time: bool,
}

impl<'a> Display for TypeContext<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        let ty = &self.gctx.types[self.handle];
        if ty.needs_alias() && !self.first_time {
            let name = &self.names[&NameKey::Type(self.handle)];
            return write!(out, "{name}");
        }

        match ty.inner {
            crate::TypeInner::Scalar { kind, .. } => put_numeric_type(out, kind, &[]),
            crate::TypeInner::Atomic { kind, .. } => {
                write!(out, "{}::atomic_{}", NAMESPACE, kind.to_msl_name())
            }
            crate::TypeInner::Vector { size, kind, .. } => put_numeric_type(out, kind, &[size]),
            crate::TypeInner::Matrix { columns, rows, .. } => {
                put_numeric_type(out, crate::ScalarKind::Float, &[rows, columns])
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
                kind,
                width: _,
                space,
            } => {
                match space.to_msl_name() {
                    Some(name) => write!(out, "{name} ")?,
                    None => return Ok(()),
                };
                match size {
                    Some(rows) => put_numeric_type(out, kind, &[rows])?,
                    None => put_numeric_type(out, kind, &[])?,
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
                let base_name = kind.to_msl_name();
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

impl crate::ScalarKind {
    const fn to_msl_name(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Sint => "int",
            Self::Uint => "uint",
            Self::Bool => "bool",
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
) -> Option<crate::ScalarKind> {
    let member = &members[index];
    //Note: this is imperfect - the same structure can be used for host-shared
    // things, where packed float would matter.
    if member.binding.is_some() {
        return None;
    }

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
            width: 4,
            kind,
        } if member.offset & 0xF != 0 || is_tight => Some(kind),
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
            Ti::Scalar { .. }
            | Ti::Vector { .. }
            | Ti::Matrix { .. }
            | Ti::Atomic { .. }
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

    fn get_packed_vec_kind(
        &self,
        expr_handle: Handle<crate::Expression>,
    ) -> Option<crate::ScalarKind> {
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
    mod_info: &'a valid::ModuleInfo,
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
        self.put_call_parameters_impl(parameters, |writer, expr| {
            writer.put_expression(expr, context, true)
        })
    }

    fn put_call_parameters_impl<E>(
        &mut self,
        parameters: impl Iterator<Item = Handle<crate::Expression>>,
        put_expression: E,
    ) -> BackendResult
    where
        E: Fn(&mut Self, Handle<crate::Expression>) -> BackendResult,
    {
        write!(self.out, "(")?;
        for (i, handle) in parameters.enumerate() {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            put_expression(self, handle)?;
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
            LevelOfDetail::Restricted(load) => {
                write!(self.out, "{}{}", CLAMPED_LOD_LOAD_PREFIX, load.index())?
            }
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
        let coordinate_type = kind.to_msl_name();
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
            crate::TypeInner::Scalar { .. } => {
                put_numeric_type(&mut self.out, crate::ScalarKind::Uint, &[])?
            }
            crate::TypeInner::Vector { size, .. } => {
                put_numeric_type(&mut self.out, crate::ScalarKind::Uint, &[size])?
            }
            _ => return Err(Error::Validation),
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
                None => return Err(Error::Validation),
            },
            crate::TypeInner::Array {
                size: crate::ArraySize::Dynamic,
                ..
            } => (0, global.ty),
            _ => return Err(Error::Validation),
        };

        let (size, stride) = match context.module.types[array_ty].inner {
            crate::TypeInner::Array { base, stride, .. } => (
                context.module.types[base]
                    .inner
                    .size(context.module.to_ctx()),
                stride,
            ),
            _ => return Err(Error::Validation),
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
            "(_buffer_sizes.size{idx} - {offset} - {size}) / {stride}",
            idx = handle.index(),
            offset = offset,
            size = size,
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
        self.put_atomic_operation(pointer, "fetch_", key, value, context)
    }

    fn put_atomic_operation(
        &mut self,
        pointer: Handle<crate::Expression>,
        key1: &str,
        key2: &str,
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
            "{NAMESPACE}::atomic_{key1}{key2}_explicit({ATOMIC_REFERENCE}"
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
        // Write parantheses around the dot product expression to prevent operators
        // with different precedences from applying earlier.
        write!(self.out, "(")?;

        // Cycle trough all the components of the vector
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
    ) -> BackendResult {
        self.put_possibly_const_expression(
            expr_handle,
            &module.const_expressions,
            module,
            |writer, expr| writer.put_const_expression(expr, module),
        )
    }

    fn put_possibly_const_expression<E>(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &crate::Arena<crate::Expression>,
        module: &crate::Module,
        put_expression: E,
    ) -> BackendResult
    where
        E: Fn(&mut Self, Handle<crate::Expression>) -> BackendResult,
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
                crate::Literal::Bool(value) => {
                    write!(self.out, "{value}")?;
                }
            },
            crate::Expression::Constant(handle) => {
                let constant = &module.constants[handle];
                if constant.name.is_some() {
                    write!(self.out, "{}", self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.put_const_expression(constant.init, module)?;
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
                    crate::TypeInner::Scalar { .. }
                    | crate::TypeInner::Vector { .. }
                    | crate::TypeInner::Matrix { .. } => {
                        self.put_call_parameters_impl(components.iter().copied(), put_expression)?;
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
                            put_expression(self, component)?;
                        }
                        write!(self.out, "}}")?;
                    }
                    _ => return Err(Error::UnsupportedCompose(ty)),
                }
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
            | crate::Expression::Compose { .. } => {
                self.put_possibly_const_expression(
                    expr_handle,
                    &context.function.expressions,
                    context.module,
                    |writer, expr| writer.put_expression(expr, context, true),
                )?;
            }
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
            crate::Expression::Splat { size, value } => {
                let scalar_kind = match *context.resolve_type(value) {
                    crate::TypeInner::Scalar { kind, .. } => kind,
                    _ => return Err(Error::Validation),
                };
                put_numeric_type(&mut self.out, scalar_kind, &[size])?;
                write!(self.out, "(")?;
                self.put_expression(value, context, true)?;
                write!(self.out, ")")?;
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
                    self.put_const_expression(offset, context.module)?;
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
                use crate::{ScalarKind as Sk, UnaryOperator as Uo};
                let op_str = match op {
                    Uo::Negate => "-",
                    Uo::Not => match context.resolve_type(expr).scalar_kind() {
                        Some(Sk::Sint) | Some(Sk::Uint) => "~",
                        Some(Sk::Bool) => "!",
                        _ => return Err(Error::Validation),
                    },
                };
                write!(self.out, "{op_str}(")?;
                self.put_expression(expr, context, false)?;
                write!(self.out, ")")?;
            }
            crate::Expression::Binary { op, left, right } => {
                let op_str = crate::back::binary_operation_str(op);
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
                    write!(self.out, "{NAMESPACE}::select(")?;
                    self.put_expression(reject, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(accept, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(condition, context, true)?;
                    write!(self.out, ")")?;
                }
                _ => return Err(Error::Validation),
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
                    crate::RelationalFunction::IsFinite => "isfinite",
                    crate::RelationalFunction::IsNormal => "isnormal",
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
                    &crate::TypeInner::Scalar { .. } => true,
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
                            kind: crate::ScalarKind::Float,
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
                    Mf::ExtractBits => "extract_bits",
                    Mf::InsertBits => "insert_bits",
                    Mf::FindLsb => "",
                    Mf::FindMsb => "",
                    // data packing
                    Mf::Pack4x8snorm => "pack_float_to_snorm4x8",
                    Mf::Pack4x8unorm => "pack_float_to_unorm4x8",
                    Mf::Pack2x16snorm => "pack_float_to_snorm2x16",
                    Mf::Pack2x16unorm => "pack_float_to_unorm2x16",
                    Mf::Pack2x16float => "",
                    // data unpacking
                    Mf::Unpack4x8snorm => "unpack_snorm4x8_to_float",
                    Mf::Unpack4x8unorm => "unpack_unorm4x8_to_float",
                    Mf::Unpack2x16snorm => "unpack_snorm2x16_to_float",
                    Mf::Unpack2x16unorm => "unpack_unorm2x16_to_float",
                    Mf::Unpack2x16float => "",
                };

                if fun == Mf::Distance && scalar_argument {
                    write!(self.out, "{NAMESPACE}::abs(")?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, " - ")?;
                    self.put_expression(arg1.unwrap(), context, false)?;
                    write!(self.out, ")")?;
                } else if fun == Mf::FindLsb {
                    write!(self.out, "((({NAMESPACE}::ctz(")?;
                    self.put_expression(arg, context, true)?;
                    write!(self.out, ") + 1) % 33) - 1)")?;
                } else if fun == Mf::FindMsb {
                    let inner = context.resolve_type(arg);

                    write!(self.out, "{NAMESPACE}::select(31 - {NAMESPACE}::clz(")?;

                    if let Some(crate::ScalarKind::Sint) = inner.scalar_kind() {
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
                        crate::TypeInner::Vector { size, kind, .. } => {
                            let size = back::vector_size_str(size);
                            if let crate::ScalarKind::Sint = kind {
                                write!(self.out, "int{size}")?;
                            } else {
                                write!(self.out, "uint{size}")?;
                            }
                        }
                        crate::TypeInner::Scalar { kind, .. } => {
                            if let crate::ScalarKind::Sint = kind {
                                write!(self.out, "int")?;
                            } else {
                                write!(self.out, "uint")?;
                            }
                        }
                        _ => (),
                    }

                    write!(self.out, "(-1), ")?;
                    self.put_expression(arg, context, true)?;
                    write!(self.out, " == 0 || ")?;
                    self.put_expression(arg, context, true)?;
                    write!(self.out, " == -1)")?;
                } else if fun == Mf::Unpack2x16float {
                    write!(self.out, "float2(as_type<half2>(")?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, "))")?;
                } else if fun == Mf::Pack2x16float {
                    write!(self.out, "as_type<uint>(half2(")?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, "))")?;
                } else if fun == Mf::Radians {
                    write!(self.out, "((")?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, ") * 0.017453292519943295474)")?;
                } else if fun == Mf::Degrees {
                    write!(self.out, "((")?;
                    self.put_expression(arg, context, false)?;
                    write!(self.out, ") * 57.295779513082322865)")?;
                } else if fun == Mf::Modf || fun == Mf::Frexp {
                    write!(self.out, "{fun_name}")?;
                    self.put_call_parameters(iter::once(arg), context)?;
                } else {
                    write!(self.out, "{NAMESPACE}::{fun_name}")?;
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
            } => match *context.resolve_type(expr) {
                crate::TypeInner::Scalar {
                    kind: src_kind,
                    width: src_width,
                }
                | crate::TypeInner::Vector {
                    kind: src_kind,
                    width: src_width,
                    ..
                } => {
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
                    write!(self.out, "{op}<")?;
                    match *context.resolve_type(expr) {
                        crate::TypeInner::Vector { size, .. } => {
                            put_numeric_type(&mut self.out, kind, &[size])?
                        }
                        _ => put_numeric_type(&mut self.out, kind, &[])?,
                    };
                    write!(self.out, ">(")?;
                    self.put_expression(expr, context, true)?;
                    write!(self.out, ")")?;
                }
                crate::TypeInner::Matrix { columns, rows, .. } => {
                    put_numeric_type(&mut self.out, kind, &[rows, columns])?;
                    write!(self.out, "(")?;
                    self.put_expression(expr, context, true)?;
                    write!(self.out, ")")?;
                }
                _ => return Err(Error::Validation),
            },
            // has to be a named expression
            crate::Expression::CallResult(_)
            | crate::Expression::AtomicResult { .. }
            | crate::Expression::WorkGroupUniformLoadResult { .. }
            | crate::Expression::RayQueryProceedResult => {
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
                    crate::Expression::GlobalVariable(handle) => handle,
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
            crate::Expression::RayQueryGetIntersection { query, committed } => {
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
                    "user_instance_id",
                    "instance_id",
                    "", // SBT offset
                    "geometry_id",
                    "primitive_id",
                    "triangle_barycentric_coord",
                    "triangle_front_facing",
                    "", // padding
                    "object_to_world_transform",
                    "world_to_object_transform",
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
        if let Some(scalar_kind) = context.get_packed_vec_kind(expr_handle) {
            write!(self.out, "{}::{}3(", NAMESPACE, scalar_kind.to_msl_name())?;
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

            if let Expression::Math { fun, arg, arg1, .. } = *expr {
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
                        if let crate::TypeInner::Scalar { kind, .. } = *inner {
                            match kind {
                                crate::ScalarKind::Sint | crate::ScalarKind::Uint => {
                                    self.need_bake_expressions.insert(arg);
                                    self.need_bake_expressions.insert(arg1.unwrap());
                                }
                                _ => {}
                            }
                        }
                    }
                    crate::MathFunction::FindMsb => {
                        self.need_bake_expressions.insert(arg);
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
            TypeResolution::Value(crate::TypeInner::Scalar { kind, .. }) => {
                put_numeric_type(&mut self.out, kind, &[])?;
            }
            TypeResolution::Value(crate::TypeInner::Vector { size, kind, .. }) => {
                put_numeric_type(&mut self.out, kind, &[size])?;
            }
            TypeResolution::Value(crate::TypeInner::Matrix { columns, rows, .. }) => {
                put_numeric_type(&mut self.out, crate::ScalarKind::Float, &[rows, columns])?;
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

        write!(
            self.out,
            "{}uint {}{} = ",
            indent,
            CLAMPED_LOD_LOAD_PREFIX,
            load.index(),
        )?;
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

                        let info = &context.expression.info[handle];
                        let ptr_class = info
                            .ty
                            .inner_with(&context.expression.module.types)
                            .pointer_space();
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
                                    self.need_bake_expressions.contains(&handle)
                                };

                            if bake {
                                Some(format!("{}{}", back::BAKE_PREFIX, handle.index()))
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
                        let name = format!("{}{}", back::BAKE_PREFIX, expr.index());
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
                    let fun_info = &context.mod_info[function];
                    let mut supports_array_length = false;
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
                    write!(self.out, "{level}")?;
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
                            self.put_atomic_operation(
                                pointer,
                                "exchange",
                                "",
                                value,
                                &context.expression,
                            )?;
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
                            let name = format!("{}{}", back::BAKE_PREFIX, result.index());
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
        self.names.clear();
        self.namer.reset(
            module,
            super::keywords::RESERVED,
            &[],
            &[],
            &[],
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

        if module.types.iter().any(|(_, t)| match t.inner {
            crate::TypeInner::RayQuery => true,
            _ => false,
        }) {
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
        }
        if options
            .bounds_check_policies
            .contains(index::BoundsCheckPolicy::ReadZeroSkipWrite)
        {
            self.put_default_constructible()?;
        }
        writeln!(self.out)?;

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
                    writeln!(self.out, "{}uint size{};", back::INDENT, idx)?;
                }

                writeln!(self.out, "}};")?;
                writeln!(self.out)?;
            }
        };

        self.write_type_defs(module)?;
        self.write_global_constants(module)?;
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
                        // quick and dirty way to figure out if we need this...
                        if member.binding.is_none() && member.offset > last_offset {
                            self.struct_member_pads.insert((handle, index as u32));
                            let pad = member.offset - last_offset;
                            writeln!(self.out, "{}char _pad{}[{}];", back::INDENT, index, pad)?;
                        }
                        let ty_inner = &module.types[member.ty].inner;
                        last_offset = member.offset + ty_inner.size(module.to_ctx());

                        let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];

                        // If the member should be packed (as is the case for a misaligned vec3) issue a packed vector
                        match should_pack_struct_member(members, span, index, module) {
                            Some(kind) => {
                                writeln!(
                                    self.out,
                                    "{}{}::packed_{}3 {};",
                                    back::INDENT,
                                    NAMESPACE,
                                    kind.to_msl_name(),
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
    fn write_global_constants(&mut self, module: &crate::Module) -> BackendResult {
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
            self.put_const_expression(constant.init, module)?;
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
                    if var.space.needs_pass_through() {
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
                    binding: None,
                    reference: true,
                };
                let separator =
                    separate(index + 1 != pass_through_globals.len() || supports_array_length);
                write!(self.out, "{}", back::INDENT)?;
                tyvar.try_fmt(&mut self.out)?;
                writeln!(self.out, "{separator}")?;
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
                        self.put_const_expression(value, module)?;
                    }
                    None => {
                        write!(self.out, " = {{}}")?;
                    }
                };
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
            self.update_expressions_to_bake(fun, fun_info, &context.expression);
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

            log::trace!(
                "entry point {:?}, index {:?}",
                fun.name.as_deref().unwrap_or("(anonymous)"),
                ep_index
            );

            // Is any global variable used by this entry point dynamically sized?
            let supports_array_length = module
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
                if supports_array_length {
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

            let (em_str, in_mode, out_mode) = match ep.stage {
                crate::ShaderStage::Vertex => (
                    "vertex",
                    LocationMode::VertexInput,
                    LocationMode::VertexOutput,
                ),
                crate::ShaderStage::Fragment { .. } => (
                    "fragment",
                    LocationMode::FragmentInput,
                    LocationMode::FragmentOutput,
                ),
                crate::ShaderStage::Compute { .. } => {
                    ("kernel", LocationMode::Uniform, LocationMode::Uniform)
                }
            };

            // Since `Namer.reset` wasn't expecting struct members to be
            // suddenly injected into another namespace like this,
            // `self.names` doesn't keep them distinct from other variables.
            // Generate fresh names for these arguments, and remember the
            // mapping.
            let mut flattened_member_names = FastHashMap::default();
            // Varyings' members get their own namespace
            let mut varyings_namer = crate::proc::Namer::default();

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
                                    varyings_namer.call(&self.names[&name_key])
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

            // Identify the varyings among the argument values, and emit a
            // struct type named `<fun>Input` to hold them.
            let stage_in_name = format!("{fun_name}Input");
            let varyings_member_name = self.namer.call("varyings");
            let mut has_varyings = false;
            if !flattened_arguments.is_empty() {
                writeln!(self.out, "struct {stage_in_name} {{")?;
                for &(ref name_key, ty, binding) in flattened_arguments.iter() {
                    let binding = match binding {
                        Some(ref binding @ &crate::Binding::Location { .. }) => binding,
                        _ => continue,
                    };
                    has_varyings = true;
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
                    write!(self.out, "{}{} {}", back::INDENT, ty_name, name)?;
                    resolved.try_fmt(&mut self.out)?;
                    writeln!(self.out, ";")?;
                }
                writeln!(self.out, "}};")?;
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
                        let binding = binding.ok_or(Error::Validation)?;

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
                // the resolves have already been checked for `!fake_missing_bindings` case
                let resolved = match var.space {
                    crate::AddressSpace::PushConstant => options.resolve_push_constants(ep).ok(),
                    crate::AddressSpace::WorkGroup => None,
                    crate::AddressSpace::Storage { .. } if options.lang_version < (2, 0) => {
                        return Err(Error::UnsupportedAddressSpace(var.space))
                    }
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
                    self.put_const_expression(value, module)?;
                }
                writeln!(self.out)?;
            }

            // If this entry uses any variable-length arrays, their sizes are
            // passed as a final struct-typed argument.
            if supports_array_length {
                // this is checked earlier
                let resolved = options.resolve_sizes_buffer(ep).unwrap();
                let separator = if module.global_variables.is_empty() {
                    ' '
                } else {
                    ','
                };
                write!(
                    self.out,
                    "{separator} constant _mslBufferSizes& _buffer_sizes",
                )?;
                resolved.try_fmt(&mut self.out)?;
                writeln!(self.out)?;
            }

            // end of the entry point argument list
            writeln!(self.out, ") {{")?;

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
                            self.put_const_expression(value, module)?;
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
                            if let Some(crate::Binding::Location { .. }) = member.binding {
                                write!(self.out, "{varyings_member_name}.")?;
                            }
                            write!(self.out, "{name}")?;
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
                        self.put_const_expression(value, module)?;
                    }
                    None => {
                        write!(self.out, " = {{}}")?;
                    }
                };
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
            self.update_expressions_to_bake(fun, fun_info, &context.expression);
            self.put_block(back::Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
            if ep_index + 1 != module.entry_points.len() {
                writeln!(self.out)?;
            }
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
        // last observed macOS value: 19152 (CI)
        if !(9000..=20000).contains(&stack_size) {
            panic!("`put_block` stack size {stack_size} has changed!");
        }
    }
}
