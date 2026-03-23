use core::fmt::Write;
use alloc::string::ToString;

use crate::{Handle, back::{self, Baked, msl::{BackendResult, Error, Writer, writer::{RAY_QUERY_FIELD_INTERSECTION, RAY_QUERY_FIELD_INTERSECTOR, RAY_QUERY_FIELD_READY, RAY_QUERY_MODERN_SUPPORT, RT_NAMESPACE, StatementContext}}}};

impl<W: Write> Writer<W> {
    pub(super) fn write_ray_query_stmt(&mut self, level: back::Level, context: &StatementContext, query: Handle<crate::Expression>, fun: &crate::RayQueryFunction) -> BackendResult {
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
                // next returns bool
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ".?.next();")?;
            }
            crate::RayQueryFunction::GenerateIntersection { hit_t } => {
                write!(self.out, "{level}")?;
                self.put_expression(query, &context.expression, true)?;
                write!(self.out, ".?.commit_bounding_box_intersection(")?;
                self.put_expression(hit_t, &context.expression, true)?;
                writeln!(self.out, ");")?;
            }
            crate::RayQueryFunction::ConfirmIntersection => {
                write!(self.out, "{level}")?;
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ".?.commit_triangle_intersection();")?;
            }
            crate::RayQueryFunction::Terminate => {
                if RAY_QUERY_MODERN_SUPPORT {
                    write!(self.out, "{level}")?;
                    self.put_expression(query, &context.expression, true)?;
                    writeln!(self.out, ".?.abort();")?;
                }
                write!(self.out, "{level}")?;
                self.put_expression(query, &context.expression, true)?;
                writeln!(self.out, ".{RAY_QUERY_FIELD_READY} = false;")?;
            }
        }

        Ok(())
    }
}