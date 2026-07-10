use alloc::{
    format,
    string::{String, ToString},
};
use core::fmt::Write;

use crate::{
    back::{
        self,
        msl::{
            writer::{NameKeyExt, StatementContext, TypeContext, WrappedFunction},
            BackendResult, Error, Writer, NAMESPACE,
        },
        Baked, INDENT,
    },
    Handle,
};

pub(super) const RT_NAMESPACE: &str = "metal::raytracing";

/// The ray query type, needs to be a function so it can format the constants.
pub(super) fn metal_intersector_ty() -> String {
    format!("{RT_NAMESPACE}::intersection_query<{RT_NAMESPACE}::instancing, {RT_NAMESPACE}::triangle_data>")
}

pub(super) const INTERSECTION_FUNCTION_NAME: &str = "ray_query_get_intersection";
pub(crate) const RAY_QUERY_TRACKER_VARIABLE_PREFIX: &str = "naga_query_init_tracker_for_";
pub(crate) const RAY_QUERY_T_MAX_TRACKER_VARIABLE_PREFIX: &str = "naga_query_tmax_tracker_for_";

impl<W: Write> Writer<W> {
    fn write_not_finite(&mut self, expr: &str) -> BackendResult {
        self.write_contains_flags(&format!("as_type<uint>({expr})"), 0x7f800000)
    }

    /// Checks whether `expr` does not have the bitpattern of IEEE f32 `NaN`.
    ///
    /// Note that this evaluates `expr` in the written code multiple times.
    fn write_is_nan(&mut self, expr: &str) -> BackendResult {
        write!(self.out, "(")?;
        self.write_not_finite(expr)?;
        write!(self.out, " && ((as_type<uint>({expr}) & 0x7fffff) != 0))")?;
        Ok(())
    }

    fn write_contains_flags(&mut self, expr: &str, flags: u32) -> BackendResult {
        write!(self.out, "(({expr} & {flags}) == {flags})")?;
        Ok(())
    }

    /// Writes a function to get the current intersection from the ray query
    ///
    /// Like other backends, this is needed to have a single branch for constructing
    /// the parts of the intersection that need to be checked whether they do or don't
    /// hit.
    pub(super) fn write_rq_get_intersection_function(
        &mut self,
        module: &crate::Module,
        committed: bool,
        options: &super::Options,
    ) -> BackendResult {
        let wrapped = WrappedFunction::RayQueryGetIntersection { committed };
        if !self.wrapped_functions.insert(wrapped) {
            return Ok(());
        }

        let ty = if committed { "committed" } else { "candidate" };
        let intersection = TypeContext {
            handle: module
                .special_types
                .ray_intersection
                .expect("intersection ty should be there for intersection function"),
            gctx: module.to_ctx(),
            names: &self.names,
            access: crate::StorageAccess::empty(),
            first_time: false,
        };
        let mut base_level = back::Level(1);
        writeln!(
            self.out,
            "{intersection} {INTERSECTION_FUNCTION_NAME}_{committed}({} intersector",
            metal_intersector_ty()
        )?;
        if options.ray_query_initialization_tracking {
            writeln!(self.out, ", uint intersector_tracker")?;
        }
        writeln!(self.out, ") {{")?;
        // Initialize the intersection to its default values (which should be zero).
        writeln!(
            self.out,
            "{base_level}{intersection} intersection = {intersection} {{}};"
        )?;

        if options.ray_query_initialization_tracking {
            write!(self.out, "{base_level}if (")?;
            if committed {
                self.write_contains_flags(
                    "intersector_tracker",
                    back::RayQueryPoint::FINISHED_TRAVERSAL.bits(),
                )?;
            } else {
                self.write_contains_flags(
                    "intersector_tracker",
                    back::RayQueryPoint::PROCEED.bits(),
                )?;
                write!(self.out, " && !")?;
                self.write_contains_flags(
                    "intersector_tracker",
                    back::RayQueryPoint::FINISHED_TRAVERSAL.bits(),
                )?;
            }
            writeln!(self.out, ") {{")?;
            base_level = base_level.next();
        }

        writeln!(self.out, "{base_level}{RT_NAMESPACE}::intersection_type ty = intersector.get_{ty}_intersection_type();")?;
        // If the ray hit a triangle, call all methods that require that and set the intersection type.
        writeln!(
            self.out,
            "{base_level}if (ty == {RT_NAMESPACE}::intersection_type::triangle) {{"
        )?;
        writeln!(
            self.out,
            "{base_level}{INDENT}intersection.kind = {};",
            crate::RayQueryIntersection::Triangle as u32
        )?;
        if !committed {
            writeln!(
                self.out,
                "{base_level}{INDENT}intersection.t = intersector.get_candidate_triangle_distance();"
            )?;
        }
        writeln!(self.out, "{base_level}{INDENT}intersection.barycentrics = intersector.get_{ty}_triangle_barycentric_coord();")?;
        writeln!(
            self.out,
            "{base_level}{INDENT}intersection.front_face = intersector.is_{ty}_triangle_front_facing();"
        )?;
        // Otherwise, if the ray hit an AABB (called a bounding box in metal) set the intersection type
        // (which depends on whether this is a committed or candidate intersection).
        writeln!(
            self.out,
            "{base_level}}} else if (ty == {RT_NAMESPACE}::intersection_type::bounding_box) {{"
        )?;
        if committed {
            writeln!(
                self.out,
                "{base_level}{INDENT}intersection.kind = {};",
                crate::RayQueryIntersection::Generated as u32
            )?;
        } else {
            writeln!(
                self.out,
                "{base_level}{INDENT}intersection.kind = {};",
                crate::RayQueryIntersection::Aabb as u32
            )?;
        }
        writeln!(self.out, "{base_level}}}")?;

        // If the ray hit anything at all, call all methods that require that.
        writeln!(
            self.out,
            "{base_level}if (ty != {RT_NAMESPACE}::intersection_type::none) {{"
        )?;
        if committed {
            writeln!(
                self.out,
                "{base_level}{INDENT}intersection.t = intersector.get_committed_distance();"
            )?;
        }
        writeln!(self.out, "{base_level}{INDENT}intersection.instance_custom_data = intersector.get_{ty}_user_instance_id();")?;
        writeln!(
            self.out,
            "{base_level}{INDENT}intersection.instance_index = intersector.get_{ty}_instance_id();"
        )?;
        // Metal does not appear to support obtaining the intersection offset from a ray query.
        //writeln!(self.out, "{level}{level}intersection.sbt_record_offset = intersector.get_{ty}_user_instance_id();")?;
        writeln!(
            self.out,
            "{base_level}{INDENT}intersection.geometry_index = intersector.get_{ty}_geometry_id();"
        )?;
        writeln!(
            self.out,
            "{base_level}{INDENT}intersection.primitive_index = intersector.get_{ty}_primitive_id();"
        )?;
        writeln!(self.out, "{base_level}{INDENT}intersection.object_to_world = intersector.get_{ty}_object_to_world_transform();")?;
        writeln!(self.out, "{base_level}{INDENT}intersection.world_to_object = intersector.get_{ty}_world_to_object_transform();")?;
        writeln!(self.out, "{base_level}}}")?;

        if options.ray_query_initialization_tracking {
            writeln!(self.out, "{INDENT}}}")?;
        }

        writeln!(self.out, "{INDENT}return intersection;")?;
        writeln!(self.out, "}}")?;

        Ok(())
    }

    pub(super) fn write_ray_query_stmt(
        &mut self,
        level: back::Level,
        context: &StatementContext,
        query: Handle<crate::Expression>,
        fun: &crate::RayQueryFunction,
    ) -> BackendResult {
        if context.expression.lang_version < (2, 4) {
            return Err(Error::UnsupportedRayTracing);
        }

        // There are three possibilities for a ptr to be:
        // 1. A variable
        // 2. A function argument
        // 3. part of a struct
        //
        // 2 and 3 are not possible, a ray query (in naga IR)
        // is not allowed to be passed into a function, and
        // all languages disallow it in a struct (you get fun results if
        // you try it :) ).
        //
        // Therefore, the ray query expression must be a variable.
        let crate::Expression::LocalVariable(query_var) =
            context.expression.function.expressions[query]
        else {
            unreachable!()
        };

        let tracker_expr_name = format!(
            "{RAY_QUERY_TRACKER_VARIABLE_PREFIX}{}",
            self.names[&crate::proc::NameKey::local(context.expression.origin, query_var)]
        );

        let tmax_tracker_expr_name = format!(
            "{RAY_QUERY_T_MAX_TRACKER_VARIABLE_PREFIX}{}",
            self.names[&crate::proc::NameKey::local(context.expression.origin, query_var)]
        );

        // TODO: check for misuse.
        match *fun {
            crate::RayQueryFunction::Initialize {
                acceleration_structure,
                descriptor,
            } => {
                //TODO: how to deal with winding? Is it by default the same as the other APIs?

                // Put everything in a block so that the variable names
                // do not conflict with user variable names
                writeln!(self.out, "{level}{{")?;

                let inner_level = level.next();

                let naga_ray_desc_ty = TypeContext {
                    handle: context
                        .expression
                        .module
                        .special_types
                        .ray_desc
                        .expect("ray desc is required as an argument so should be there"),
                    gctx: context.expression.module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    first_time: false,
                };

                write!(self.out, "{inner_level}{naga_ray_desc_ty} desc = ")?;
                self.put_expression(descriptor, &context.expression, false)?;
                writeln!(self.out, ";")?;

                // Set up intersection parameters
                writeln!(
                    self.out,
                    "{inner_level}{RT_NAMESPACE}::intersection_params params;"
                )?;

                {
                    // Determine whether or not to cull opaque/non-opaques
                    let f_opaque = back::RayFlag::CULL_OPAQUE.bits();
                    let f_no_opaque = back::RayFlag::CULL_NO_OPAQUE.bits();
                    writeln!(self.out, "{inner_level}{RT_NAMESPACE}::opacity_cull_mode cull_mode = 
{inner_level}{INDENT}(desc.flags & {f_opaque}) != 0 ? {RT_NAMESPACE}::opacity_cull_mode::opaque : (
{inner_level}{INDENT}{INDENT}(desc.flags & {f_no_opaque}) != 0 ? {RT_NAMESPACE}::opacity_cull_mode::non_opaque : {RT_NAMESPACE}::opacity_cull_mode::none
{inner_level}{INDENT});")?;
                    writeln!(
                        self.out,
                        "{inner_level}params.set_opacity_cull_mode(cull_mode);"
                    )?;

                    if context.expression.ray_query_initialization_tracking {
                        writeln!(self.out, "{inner_level}bool force_opacity = cull_mode == {RT_NAMESPACE}::opacity_cull_mode::none;")?;
                    }
                }
                {
                    let mut current_level = inner_level;
                    if context.expression.ray_query_initialization_tracking {
                        writeln!(self.out, "{inner_level}if (force_opacity) {{")?;
                        current_level = current_level.next();
                    }
                    // Determine whether to force a particular opacity
                    let f_opaque = back::RayFlag::OPAQUE.bits();
                    let f_no_opaque = back::RayFlag::NO_OPAQUE.bits();
                    writeln!(self.out, "{current_level}params.force_opacity(
{current_level}    (desc.flags & {f_opaque}) != 0 ? {RT_NAMESPACE}::forced_opacity::opaque : (
{current_level}        (desc.flags & {f_no_opaque}) != 0 ? {RT_NAMESPACE}::forced_opacity::non_opaque : {RT_NAMESPACE}::forced_opacity::none
{current_level}    )
{current_level});")?;

                    if context.expression.ray_query_initialization_tracking {
                        writeln!(self.out, "{inner_level}}}")?;
                    }
                }
                {
                    let flag = back::RayFlag::TERMINATE_ON_FIRST_HIT.bits();
                    writeln!(
                        self.out,
                        "{inner_level}params.accept_any_intersection((desc.flags & {flag}) != 0);"
                    )?;
                }

                writeln!(
                    self.out,
                    "{inner_level}{RT_NAMESPACE}::ray ray = {RT_NAMESPACE}::ray(desc.origin, desc.dir, desc.tmin, desc.tmax);"
                )?;

                let mut init_level = inner_level;

                // The `reset` function is virtually undocumented (many of the Metal ray tracing functions lack it), so to be safe,
                // this assumes an invalid ray is UB (NOTE: invalid ray behaviour is defined for intersectors).
                if context.expression.ray_query_initialization_tracking {
                    write!(self.out, "{inner_level}bool invalid_nan_infs = ")?;
                    // tmax needs special handling because it can be INF
                    for (idx, &field_access) in [
                        "origin.x", "origin.y", "origin.z", "dir.x", "dir.y", "dir.z", "tmin",
                    ]
                    .iter()
                    .enumerate()
                    {
                        if idx != 0 {
                            write!(self.out, " || ")?;
                        }

                        self.write_not_finite(&format!("desc.{field_access}"))?;
                    }

                    write!(self.out, " || ")?;
                    self.write_is_nan("desc.tmax")?;
                    writeln!(self.out, ";")?;

                    // Metal also requires that tmax >= 0.0, but if tmax >= tmin and tmin >= 0.0, tmax must be >= 0.0
                    writeln!(self.out, "{inner_level}bool invalid_t = (desc.tmin > desc.tmax) || (desc.tmin < 0.0);")?;
                    // Metal requires that the length of the direction is not 0.0. This is the case only when all the
                    // components are zero.
                    //
                    // Use absolute to cover signed zero.
                    writeln!(self.out, "{inner_level}bool invalid_dir = {NAMESPACE}::all({NAMESPACE}::abs(desc.dir) == 0.0);")?;

                    writeln!(
                        self.out,
                        "{inner_level}if (!(invalid_dir || invalid_t || invalid_nan_infs)) {{"
                    )?;
                    init_level = init_level.next();
                }

                write!(self.out, "{init_level}")?;
                // A ray query can by initialized in metal by either using a "non-default constructor"
                // or by calling reset. Ray queries cannot be assigned to in metal, so reset needs to
                // be called.
                self.put_expression(query, &context.expression, true)?;
                write!(self.out, ".reset(ray,")?;
                self.put_expression(acceleration_structure, &context.expression, true)?;
                writeln!(self.out, ", desc.cull_mask, params);")?;
                if context.expression.ray_query_initialization_tracking {
                    // We don't set the initialization tracker to zero (uninitialized)
                    // if the call fails. Resetting to uninitialized might be useful
                    // for debugging, but for everything else it is just extra code.
                    writeln!(
                        self.out,
                        "{init_level}{tracker_expr_name} = {};",
                        back::RayQueryPoint::INITIALIZED.bits()
                    )?;
                    writeln!(
                        self.out,
                        "{init_level}{tmax_tracker_expr_name} = desc.tmax;"
                    )?;
                    writeln!(self.out, "{inner_level}}}")?;
                }
                writeln!(self.out, "{level}}}")?;
            }
            crate::RayQueryFunction::Proceed { result } => {
                let mut current_level = level;
                write!(self.out, "{current_level}")?;
                let name = Baked(result).to_string();
                self.start_baking_expression(result, &context.expression, &name)?;
                self.named_expressions.insert(result, name.clone());

                writeln!(self.out, "false;")?;

                if context.expression.ray_query_initialization_tracking {
                    write!(self.out, "{level}if (")?;
                    self.write_contains_flags(
                        &tracker_expr_name,
                        back::RayQueryPoint::INITIALIZED.bits(),
                    )?;
                    write!(self.out, " && !")?;
                    self.write_contains_flags(
                        &tracker_expr_name,
                        back::RayQueryPoint::FINISHED_TRAVERSAL.bits(),
                    )?;
                    write!(self.out, ")")?;
                    writeln!(self.out, " {{")?;
                    current_level = current_level.next();
                }
                write!(self.out, "{current_level}{name} = ")?;
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ".next();")?;
                if context.expression.ray_query_initialization_tracking {
                    writeln!(self.out, "{current_level}{tracker_expr_name} = {tracker_expr_name} | ({name} ? {}: {});", back::RayQueryPoint::PROCEED.bits(), (back::RayQueryPoint::PROCEED | back::RayQueryPoint::FINISHED_TRAVERSAL).bits())?;
                    writeln!(self.out, "{level}}}")?;
                }
            }
            crate::RayQueryFunction::GenerateIntersection { hit_t } => {
                let mut current_level = level;
                if context.expression.ray_query_initialization_tracking {
                    write!(self.out, "{level}if (")?;
                    self.write_contains_flags(
                        &tracker_expr_name,
                        back::RayQueryPoint::PROCEED.bits(),
                    )?;
                    write!(self.out, " && !")?;
                    self.write_contains_flags(
                        &tracker_expr_name,
                        back::RayQueryPoint::FINISHED_TRAVERSAL.bits(),
                    )?;
                    write!(self.out, ")")?;
                } else {
                    // For readability
                    write!(self.out, "{level}")?;
                }
                writeln!(self.out, "{{")?;
                current_level = current_level.next();
                write!(self.out, "{current_level}float t = ")?;
                self.put_expression(hit_t, &context.expression, true)?;
                writeln!(self.out, ";")?;
                if context.expression.ray_query_initialization_tracking {
                    write!(
                        self.out,
                        "{current_level}float current_max_t = {tmax_tracker_expr_name};
{current_level}if ("
                    )?;
                    self.put_expression(query, &context.expression, true)?;
                    write!(self.out, ".get_committed_intersection_type() != {RT_NAMESPACE}::intersection_type::none) {{
{current_level}{INDENT}current_max_t = ")?;
                    self.put_expression(query, &context.expression, true)?;
                    write!(
                        self.out,
                        ".get_committed_distance();
{current_level}}}
{current_level}if ("
                    )?;
                    self.put_expression(query, &context.expression, true)?;
                    write!(self.out, ".get_candidate_intersection_type() == {RT_NAMESPACE}::intersection_type::bounding_box && (")?;
                    self.put_expression(query, &context.expression, true)?;
                    write!(self.out, ".get_ray_min_distance()")?;
                    writeln!(self.out, " <= t) && (t <= current_max_t)) {{")?;
                    current_level = current_level.next();
                }
                write!(self.out, "{current_level}")?;
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ".commit_bounding_box_intersection(t);")?;
                if context.expression.ray_query_initialization_tracking {
                    writeln!(self.out, "{level}{INDENT}}}")?;
                }
                writeln!(self.out, "{level}}}")?;
            }
            crate::RayQueryFunction::ConfirmIntersection => {
                let mut current_level = level;
                if context.expression.ray_query_initialization_tracking {
                    write!(self.out, "{level}if (")?;
                    self.write_contains_flags(
                        &tracker_expr_name,
                        back::RayQueryPoint::PROCEED.bits(),
                    )?;
                    write!(self.out, " && !")?;
                    self.write_contains_flags(
                        &tracker_expr_name,
                        back::RayQueryPoint::FINISHED_TRAVERSAL.bits(),
                    )?;
                    writeln!(self.out, ") {{")?;
                    current_level = current_level.next();
                    write!(self.out, "{current_level}if (")?;
                    self.put_expression(query, &context.expression, true)?;
                    writeln!(self.out, ".get_candidate_intersection_type() == {RT_NAMESPACE}::intersection_type::triangle) {{")?;
                }
                write!(self.out, "{level}")?;
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ".commit_triangle_intersection();")?;
                if context.expression.ray_query_initialization_tracking {
                    writeln!(
                        self.out,
                        "{level}{INDENT}}}
{level}}}"
                    )?;
                }
            }
            crate::RayQueryFunction::Terminate => {
                let mut current_level = level;
                if context.expression.ray_query_initialization_tracking {
                    write!(self.out, "{level}if (")?;
                    self.write_contains_flags(
                        &tracker_expr_name,
                        back::RayQueryPoint::PROCEED.bits(),
                    )?;
                    write!(self.out, " && !")?;
                    self.write_contains_flags(
                        &tracker_expr_name,
                        back::RayQueryPoint::FINISHED_TRAVERSAL.bits(),
                    )?;
                    writeln!(self.out, ") {{")?;
                    current_level = current_level.next();
                }
                write!(self.out, "{current_level}")?;
                self.put_expression(query, &context.expression, true)?;
                // Terminate appears to map to abort in spirv-cross, but metal only documents
                // the existence of this method, not what it does.
                writeln!(self.out, ".abort();")?;
                // To get the committed intersection, an extra proceed must occur as specified in
                // the API docs.
                if context.expression.ray_query_initialization_tracking {
                    writeln!(self.out, "{level}}}")?;
                }
            }
        }

        Ok(())
    }
}
