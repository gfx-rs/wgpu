use alloc::{
    format,
    string::{String, ToString},
};
use core::fmt::Write;

use crate::{
    back::{
        self,
        msl::{
            writer::{StatementContext, TypeContext, WrappedFunction},
            BackendResult, Error, Writer,
        },
        Baked,
    },
    Handle,
};

pub(super) const RT_NAMESPACE: &str = "metal::raytracing";

/// The ray query type, needs to be a function so it can format the constants.
pub(super) fn metal_intersector_ty() -> String {
    format!("{RT_NAMESPACE}::intersection_query<{RT_NAMESPACE}::instancing, {RT_NAMESPACE}::triangle_data>")
}

pub(super) const INTERSECTION_FUNCTION_NAME: &str = "ray_query_get_intersection";

impl<W: Write> Writer<W> {
    /// Writes a function to get the current intersection from the ray query
    ///
    /// Like other backends, this is needed to have a single branch for constructing
    /// the parts of the intersection that need to be checked whether they do or don't
    /// hit.
    pub(super) fn write_rq_get_intersection_function(
        &mut self,
        module: &crate::Module,
        committed: bool,
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
        let level = back::Level(1);
        writeln!(
            self.out,
            "{intersection} {INTERSECTION_FUNCTION_NAME}_{committed}({} intersector) {{",
            metal_intersector_ty()
        )?;
        // Initialize the intersection to its default values (which should be zero).
        writeln!(
            self.out,
            "{level}{intersection} intersection = {intersection} {{}};"
        )?;
        writeln!(self.out, "{level}{RT_NAMESPACE}::intersection_type ty = intersector.get_{ty}_intersection_type();")?;
        // If the ray hit a triangle, call all methods that require that and set the intersection type.
        writeln!(
            self.out,
            "{level}if (ty == {RT_NAMESPACE}::intersection_type::triangle) {{"
        )?;
        writeln!(
            self.out,
            "{level}{level}intersection.kind = {};",
            crate::RayQueryIntersection::Triangle as u32
        )?;
        if !committed {
            writeln!(
                self.out,
                "{level}{level}intersection.t = intersector.get_candidate_triangle_distance();"
            )?;
        }
        writeln!(self.out, "{level}{level}intersection.barycentrics = intersector.get_{ty}_triangle_barycentric_coord();")?;
        writeln!(
            self.out,
            "{level}{level}intersection.front_face = intersector.is_{ty}_triangle_front_facing();"
        )?;
        // Otherwise, if the ray hit an AABB (called a bounding box in metal) set the intersection type
        // (which depends on whether this is a committed or candidate intersection).
        writeln!(
            self.out,
            "{level}}} else if (ty == {RT_NAMESPACE}::intersection_type::bounding_box) {{"
        )?;
        if committed {
            writeln!(
                self.out,
                "{level}{level}intersection.kind = {};",
                crate::RayQueryIntersection::Generated as u32
            )?;
        } else {
            writeln!(
                self.out,
                "{level}{level}intersection.kind = {};",
                crate::RayQueryIntersection::Aabb as u32
            )?;
        }
        writeln!(self.out, "{level}}}")?;

        // If the ray hit anything at all, call all methods that require that.
        writeln!(
            self.out,
            "{level}if (ty != {RT_NAMESPACE}::intersection_type::none) {{"
        )?;
        if committed {
            writeln!(
                self.out,
                "{level}{level}intersection.t = intersector.get_committed_distance();"
            )?;
        }
        writeln!(self.out, "{level}{level}intersection.instance_custom_data = intersector.get_{ty}_user_instance_id();")?;
        writeln!(
            self.out,
            "{level}{level}intersection.instance_index = intersector.get_{ty}_instance_id();"
        )?;
        // Metal does not appear to support obtaining the intersection offset from a ray query.
        //writeln!(self.out, "{level}{level}intersection.sbt_record_offset = intersector.get_{ty}_user_instance_id();")?;
        writeln!(
            self.out,
            "{level}{level}intersection.geometry_index = intersector.get_{ty}_geometry_id();"
        )?;
        writeln!(
            self.out,
            "{level}{level}intersection.primitive_index = intersector.get_{ty}_primitive_id();"
        )?;
        writeln!(self.out, "{level}{level}intersection.object_to_world = intersector.get_{ty}_object_to_world_transform();")?;
        writeln!(self.out, "{level}{level}intersection.world_to_object = intersector.get_{ty}_world_to_object_transform();")?;
        writeln!(self.out, "{level}}}")?;
        writeln!(self.out, "{level}return intersection;")?;
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
                    writeln!(
                        self.out,
                        "{inner_level}params.set_opacity_cull_mode(
{inner_level}    (desc.flags & {f_opaque}) != 0 ? {RT_NAMESPACE}::opacity_cull_mode::opaque : (
{inner_level}        (desc.flags & {f_no_opaque}) != 0 ? {RT_NAMESPACE}::opacity_cull_mode::non_opaque : {RT_NAMESPACE}::opacity_cull_mode::none
{inner_level}    )
{inner_level});"
                    )?;
                }
                {
                    // Determine whether to force a particular opacity
                    let f_opaque = back::RayFlag::OPAQUE.bits();
                    let f_no_opaque = back::RayFlag::NO_OPAQUE.bits();
                    writeln!(self.out, "{inner_level}params.force_opacity(
{inner_level}    (desc.flags & {f_opaque}) != 0 ? {RT_NAMESPACE}::forced_opacity::opaque : (
{inner_level}        (desc.flags & {f_no_opaque}) != 0 ? {RT_NAMESPACE}::forced_opacity::non_opaque : {RT_NAMESPACE}::forced_opacity::none
{inner_level}    )
{inner_level});")?;
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

                write!(self.out, "{inner_level}")?;
                // A ray query can by initialized in metal by either using a "non-default constructor"
                // or by calling reset. Ray queries cannot be assigned to in metal, so reset needs to
                // be called.
                self.put_expression(query, &context.expression, true)?;
                write!(self.out, ".reset(ray,")?;
                self.put_expression(acceleration_structure, &context.expression, true)?;
                writeln!(self.out, ", desc.cull_mask, params);")?;
                writeln!(self.out, "{level}}}")?;
            }
            crate::RayQueryFunction::Proceed { result } => {
                write!(self.out, "{level}")?;
                let name = Baked(result).to_string();
                self.start_baking_expression(result, &context.expression, &name)?;
                self.named_expressions.insert(result, name);
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ".next();")?;
            }
            crate::RayQueryFunction::GenerateIntersection { hit_t } => {
                write!(self.out, "{level}")?;
                self.put_expression(query, &context.expression, true)?;
                write!(self.out, ".commit_bounding_box_intersection(")?;
                self.put_expression(hit_t, &context.expression, true)?;
                writeln!(self.out, ");")?;
            }
            crate::RayQueryFunction::ConfirmIntersection => {
                write!(self.out, "{level}")?;
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ".commit_triangle_intersection();")?;
            }
            crate::RayQueryFunction::Terminate => {
                write!(self.out, "{level}")?;
                self.put_expression(query, &context.expression, true)?;
                // Terminate appears to map to abort in spirv-cross, but metal only documents
                // the existence of this method, not what it does.
                writeln!(self.out, ".abort();")?;
            }
        }

        Ok(())
    }
}
