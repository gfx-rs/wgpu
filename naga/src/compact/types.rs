use super::{HandleSet, ModuleMap};
use crate::Handle;

pub struct TypeTracer<'a> {
    pub types_used: &'a mut HandleSet<crate::Type>,
    pub expressions_used: &'a mut HandleSet<crate::Expression>,
}

impl TypeTracer<'_> {
    pub fn trace_type(&mut self, ty: &crate::Type) {
        use crate::TypeInner as Ti;
        match ty.inner {
            // Types that do not contain handles.
            Ti::Scalar { .. }
            | Ti::Vector { .. }
            | Ti::Matrix { .. }
            | Ti::Atomic { .. }
            | Ti::ValuePointer { .. }
            | Ti::Image { .. }
            | Ti::Sampler { .. }
            | Ti::AccelerationStructure
            | Ti::RayQuery => {}

            // Types that do contain handles.
            Ti::Array {
                base,
                size: crate::ArraySize::Pending(crate::PendingArraySize::Expression(expr)),
                stride: _,
            }
            | Ti::BindingArray {
                base,
                size: crate::ArraySize::Pending(crate::PendingArraySize::Expression(expr)),
            } => {
                self.expressions_used.insert(expr);
                self.types_used.insert(base);
            }
            Ti::Pointer { base, space: _ }
            | Ti::Array {
                base,
                size: _,
                stride: _,
            }
            | Ti::BindingArray { base, size: _ } => {
                self.types_used.insert(base);
            }
            Ti::Struct {
                ref members,
                span: _,
            } => {
                self.types_used.insert_iter(members.iter().map(|m| m.ty));
            }
        }
    }
}

impl ModuleMap {
    pub fn adjust_type(&self, ty: &mut crate::Type) {
        let adjust = |ty: &mut Handle<crate::Type>| self.types.adjust(ty);

        use crate::TypeInner as Ti;
        match ty.inner {
            // Types that do not contain handles.
            Ti::Scalar(_)
            | Ti::Vector { .. }
            | Ti::Matrix { .. }
            | Ti::Atomic(_)
            | Ti::ValuePointer { .. }
            | Ti::Image { .. }
            | Ti::Sampler { .. }
            | Ti::AccelerationStructure
            | Ti::RayQuery => {}

            // Types that do contain handles.
            Ti::Pointer {
                ref mut base,
                space: _,
            } => adjust(base),
            Ti::Array {
                ref mut base,
                ref mut size,
                stride: _,
            } => {
                adjust(base);
                if let crate::ArraySize::Pending(crate::PendingArraySize::Expression(
                    ref mut size_expr,
                )) = *size
                {
                    self.global_expressions.adjust(size_expr);
                }
            }
            Ti::Struct {
                ref mut members,
                span: _,
            } => {
                for member in members {
                    self.types.adjust(&mut member.ty);
                }
            }
            Ti::BindingArray {
                ref mut base,
                size: _,
            } => {
                adjust(base);
            }
        };
    }
}
