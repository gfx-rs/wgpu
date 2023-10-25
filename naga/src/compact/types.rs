use super::{HandleSet, ModuleMap};
use crate::{Handle, UniqueArena};

pub struct TypeTracer<'a> {
    pub types: &'a UniqueArena<crate::Type>,
    pub types_used: &'a mut HandleSet<crate::Type>,
}

impl<'a> TypeTracer<'a> {
    pub fn trace_type(&mut self, ty: Handle<crate::Type>) {
        let mut work_list = vec![ty];
        while let Some(ty) = work_list.pop() {
            // If we've already seen this type, no need to traverse further.
            if !self.types_used.insert(ty) {
                continue;
            }

            use crate::TypeInner as Ti;
            match self.types[ty].inner {
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
                Ti::Pointer { base, space: _ } => work_list.push(base),
                Ti::Array {
                    base,
                    size: _,
                    stride: _,
                } => work_list.push(base),
                Ti::Struct {
                    ref members,
                    span: _,
                } => {
                    work_list.extend(members.iter().map(|m| m.ty));
                }
                Ti::BindingArray { base, size: _ } => work_list.push(base),
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
            Ti::Pointer {
                ref mut base,
                space: _,
            } => adjust(base),
            Ti::Array {
                ref mut base,
                size: _,
                stride: _,
            } => adjust(base),
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
