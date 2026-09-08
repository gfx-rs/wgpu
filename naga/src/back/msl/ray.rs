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

pub(super) const QUERY_TYPE: &str = "NagaRayQuery";

pub(super) const INTERSECTION_FUNCTION_NAME: &str = "ray_query_get_intersection";

impl<W: Write> Writer<W> {
    pub(super) fn write_ray_query_type(&mut self) -> BackendResult {
        // MSL 6.19.5 requires reset before traversal and a live candidate for
        // candidate operations. Keep state with the query so pointer aliases agree.
        writeln!(self.out, "struct {QUERY_TYPE} {{")?;
        writeln!(self.out, "    thread {}& query;", metal_intersector_ty())?;
        self.out.write_str(
            r#"    bool initialized = false;
    bool candidate = false;
    bool finished = false;
    float tmin = 0.0;
    float tmax = 0.0;

    bool next() thread {
        if (!initialized || finished) return false;
        candidate = query.next();
        finished = !candidate;
        return candidate;
    }

    void abort() thread {
        if (candidate) {
            query.abort();
            candidate = false;
            finished = true;
        }
    }

    void commit_triangle_intersection() thread {
        if (candidate && query.get_candidate_intersection_type() == metal::raytracing::intersection_type::triangle) {
            query.commit_triangle_intersection();
        }
    }

    void commit_bounding_box_intersection(float distance) thread {
        if (candidate && query.get_candidate_intersection_type() == metal::raytracing::intersection_type::bounding_box) {
            float closest = tmax;
            if (query.get_committed_intersection_type() != metal::raytracing::intersection_type::none) {
                closest = query.get_committed_distance();
            }
            if ((as_type<uint>(distance) & 0x7fffffffu) <= 0x7f800000u && distance >= tmin && distance <= closest) {
                query.commit_bounding_box_intersection(distance);
            }
        }
    }
};
"#
        )?;
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
            "{intersection} {INTERSECTION_FUNCTION_NAME}_{committed}(thread {QUERY_TYPE}& rq) {{"
        )?;
        // Initialize the intersection to its default values (which should be zero).
        writeln!(
            self.out,
            "{level}{intersection} intersection = {intersection} {{}};"
        )?;
        let available = if committed {
            "rq.finished"
        } else {
            "rq.candidate"
        };
        writeln!(self.out, "{level}if (!{available}) return intersection;")?;
        writeln!(self.out, "{level}thread auto& intersector = rq.query;")?;
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

        match *fun {
            crate::RayQueryFunction::Initialize {
                acceleration_structure,
                descriptor,
            } => {
                // Put everything in a block so that the variable names
                // do not conflict with user variable names
                writeln!(self.out, "{level}{{")?;

                let inner_level = level.next();

                let naga_ray_desc_ty = TypeContext {
                    handle: context.expression.module.special_types.ray_desc.expect(
                        "ray naga_ray_query_desc is required as an argument so should be there",
                    ),
                    gctx: context.expression.module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    first_time: false,
                };

                write!(
                    self.out,
                    "{inner_level}{naga_ray_desc_ty} naga_ray_query_desc = "
                )?;
                self.put_expression(descriptor, &context.expression, false)?;
                writeln!(self.out, ";")?;

                write!(
                    self.out,
                    "{inner_level}thread {QUERY_TYPE}& naga_ray_query_ref = "
                )?;
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ";")?;
                writeln!(self.out, "{inner_level}naga_ray_query_ref.initialized = false; naga_ray_query_ref.candidate = false; naga_ray_query_ref.finished = false;")?;
                // Bit tests remain valid with Metal fast-math enabled, unlike isnan/isfinite.
                writeln!(self.out, "{inner_level}bool naga_ray_query_valid_origin = metal::all((as_type<metal::uint3>(naga_ray_query_desc.origin) & 0x7f800000u) != 0x7f800000u);")?;
                writeln!(self.out, "{inner_level}bool naga_ray_query_valid_dir = metal::all((as_type<metal::uint3>(naga_ray_query_desc.dir) & 0x7f800000u) != 0x7f800000u) && metal::any(naga_ray_query_desc.dir != metal::float3(0.0));")?;
                writeln!(self.out, "{inner_level}bool naga_ray_query_valid_range = (as_type<uint>(naga_ray_query_desc.tmin) & 0x7f800000u) != 0x7f800000u && (as_type<uint>(naga_ray_query_desc.tmax) & 0x7fffffffu) <= 0x7f800000u && naga_ray_query_desc.tmin >= 0.0 && naga_ray_query_desc.tmax >= naga_ray_query_desc.tmin;")?;
                let opacity = (back::RayFlag::OPAQUE
                    | back::RayFlag::NO_OPAQUE
                    | back::RayFlag::CULL_OPAQUE
                    | back::RayFlag::CULL_NO_OPAQUE)
                    .bits();
                let faces = (back::RayFlag::CULL_BACK_FACING
                    | back::RayFlag::CULL_FRONT_FACING
                    | back::RayFlag::SKIP_TRIANGLES)
                    .bits();
                let geometry = (back::RayFlag::SKIP_TRIANGLES | back::RayFlag::SKIP_AABBS).bits();
                writeln!(self.out, "{inner_level}bool naga_ray_query_valid_flags = metal::popcount(naga_ray_query_desc.flags & {opacity}u) <= 1 && metal::popcount(naga_ray_query_desc.flags & {faces}u) <= 1 && metal::popcount(naga_ray_query_desc.flags & {geometry}u) <= 1;")?;
                writeln!(self.out, "{inner_level}if (naga_ray_query_valid_origin && naga_ray_query_valid_dir && naga_ray_query_valid_range && naga_ray_query_valid_flags) {{")?;
                let inner_level = inner_level.next();

                // Set up intersection parameters
                writeln!(
                    self.out,
                    "{inner_level}{RT_NAMESPACE}::intersection_params naga_ray_query_params;"
                )?;

                {
                    // Determine whether or not to cull opaque/non-opaques
                    let f_opaque = back::RayFlag::CULL_OPAQUE.bits();
                    let f_no_opaque = back::RayFlag::CULL_NO_OPAQUE.bits();
                    writeln!(
                        self.out,
                        "{inner_level}naga_ray_query_params.set_opacity_cull_mode(
{inner_level}    (naga_ray_query_desc.flags & {f_opaque}) != 0 ? {RT_NAMESPACE}::opacity_cull_mode::opaque : (
{inner_level}        (naga_ray_query_desc.flags & {f_no_opaque}) != 0 ? {RT_NAMESPACE}::opacity_cull_mode::non_opaque : {RT_NAMESPACE}::opacity_cull_mode::none
{inner_level}    )
{inner_level});"
                    )?;
                }
                {
                    // Determine whether to force a particular opacity
                    let f_opaque = back::RayFlag::OPAQUE.bits();
                    let f_no_opaque = back::RayFlag::NO_OPAQUE.bits();
                    writeln!(self.out, "{inner_level}naga_ray_query_params.force_opacity(
{inner_level}    (naga_ray_query_desc.flags & {f_opaque}) != 0 ? {RT_NAMESPACE}::forced_opacity::opaque : (
{inner_level}        (naga_ray_query_desc.flags & {f_no_opaque}) != 0 ? {RT_NAMESPACE}::forced_opacity::non_opaque : {RT_NAMESPACE}::forced_opacity::none
{inner_level}    )
{inner_level});")?;
                }
                {
                    let flag = back::RayFlag::TERMINATE_ON_FIRST_HIT.bits();
                    writeln!(
                        self.out,
                        "{inner_level}naga_ray_query_params.accept_any_intersection((naga_ray_query_desc.flags & {flag}) != 0);"
                    )?;
                }

                let back = back::RayFlag::CULL_BACK_FACING.bits();
                let front = back::RayFlag::CULL_FRONT_FACING.bits();
                let triangles = back::RayFlag::SKIP_TRIANGLES.bits();
                let aabbs = back::RayFlag::SKIP_AABBS.bits();
                writeln!(self.out, "{inner_level}naga_ray_query_params.set_triangle_front_facing_winding(metal::winding::clockwise);")?;
                writeln!(self.out, "{inner_level}naga_ray_query_params.set_triangle_cull_mode((naga_ray_query_desc.flags & {back}u) != 0 ? {RT_NAMESPACE}::triangle_cull_mode::back : ((naga_ray_query_desc.flags & {front}u) != 0 ? {RT_NAMESPACE}::triangle_cull_mode::front : {RT_NAMESPACE}::triangle_cull_mode::none));")?;
                writeln!(self.out, "{inner_level}naga_ray_query_params.set_geometry_cull_mode((naga_ray_query_desc.flags & {triangles}u) != 0 ? {RT_NAMESPACE}::geometry_cull_mode::triangle : ((naga_ray_query_desc.flags & {aabbs}u) != 0 ? {RT_NAMESPACE}::geometry_cull_mode::bounding_box : {RT_NAMESPACE}::geometry_cull_mode::none));")?;

                writeln!(
                    self.out,
                    "{inner_level}{RT_NAMESPACE}::ray naga_ray_query_ray = {RT_NAMESPACE}::ray(naga_ray_query_desc.origin, naga_ray_query_desc.dir, naga_ray_query_desc.tmin, naga_ray_query_desc.tmax);"
                )?;

                write!(
                    self.out,
                    "{inner_level}naga_ray_query_ref.query.reset(naga_ray_query_ray,"
                )?;
                self.put_expression(acceleration_structure, &context.expression, true)?;
                writeln!(
                    self.out,
                    ", naga_ray_query_desc.cull_mask, naga_ray_query_params);"
                )?;
                writeln!(self.out, "{inner_level}naga_ray_query_ref.initialized = true; naga_ray_query_ref.tmin = naga_ray_query_desc.tmin; naga_ray_query_ref.tmax = naga_ray_query_desc.tmax;")?;
                writeln!(self.out, "{}}}", level.next())?;
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
                writeln!(self.out, ".abort();")?;
            }
        }

        Ok(())
    }
}
