use super::{HandleSet, ModuleMap};
use crate::{Handle, UniqueArena};

pub struct TypeTracer<'a> {
    pub types: &'a UniqueArena<crate::Type>,
    pub types_used: &'a mut HandleSet<crate::Type>,
}

impl TypeTracer<'_> {
    /// Propagate usage through `self.types`, starting with `self.types_used`.
    ///
    /// Treat `self.types_used` as the initial set of "known
    /// live" types, and follow through to identify all
    /// transitively used types.
    pub fn trace_types(&mut self) {
        // We don't need recursion or a work list. Because an
        // expression may only refer to other expressions that precede
        // it in the arena, it suffices to make a single pass over the
        // arena from back to front, marking the referents of used
        // expressions as used themselves.
        for (handle, ty) in self.types.iter().rev() {
            // If this type isn't used, it doesn't matter what it uses.
            if !self.types_used.contains(handle) {
                continue;
            }

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
