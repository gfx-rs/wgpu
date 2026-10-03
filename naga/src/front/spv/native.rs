use super::{resolve_constant, BlockContext, Error, Frontend, LookupHelper as _};
use crate::{Expression, Handle, Span};

impl<I: Iterator<Item = u32>> Frontend<I> {
    pub(super) fn native_constant(&self, id: u32, module: &crate::Module) -> Result<u32, Error> {
        resolve_constant(module.to_ctx(), &self.lookup_constant.lookup(id)?.inner)
            .ok_or(Error::InvalidId(id))
    }

    pub(super) fn cooperative_layout(
        &self,
        id: u32,
        module: &crate::Module,
    ) -> Result<bool, Error> {
        match self.native_constant(id, module)? {
            0 => Ok(true),
            1 => Ok(false),
            _ => Err(Error::InvalidOperand),
        }
    }

    pub(super) fn parse_type_cooperative_matrix(
        &mut self,
        inst: super::Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        let start = self.data_offset;
        self.switch(super::ModuleState::Type, inst.op)?;
        inst.expect(7)?;
        let id = self.next()?;
        let component = self.next()?;
        let scope = self.next()?;
        let rows = self.next()?;
        let columns = self.next()?;
        let role = self.next()?;
        if self.native_scope(scope, module)? != crate::MemoryScope::Subgroup {
            return Err(Error::InvalidOperand);
        }
        let size = |id| match self.native_constant(id, module)? {
            8 => Ok(crate::CooperativeSize::Eight),
            16 => Ok(crate::CooperativeSize::Sixteen),
            _ => Err(Error::InvalidOperand),
        };
        let rows = size(rows)?;
        let columns = size(columns)?;
        let role = match self.native_constant(role, module)? {
            0 => crate::CooperativeRole::A,
            1 => crate::CooperativeRole::B,
            2 => crate::CooperativeRole::C,
            _ => return Err(Error::InvalidOperand),
        };
        let crate::TypeInner::Scalar(scalar) =
            module.types[self.lookup_type.lookup(component)?.handle].inner
        else {
            return Err(Error::InvalidInnerType(component));
        };
        let decor = self.future_decor.remove(&id);
        let handle = module.types.insert(
            crate::Type {
                name: decor.and_then(|d| d.name),
                inner: crate::TypeInner::CooperativeMatrix {
                    columns,
                    rows,
                    scalar,
                    role,
                },
            },
            self.span_from_with_op(start),
        );
        self.lookup_type.insert(
            id,
            super::LookupType {
                handle,
                base_id: Some(component),
            },
        );
        Ok(())
    }

    pub(super) fn pointer_cast(
        mut value: Handle<Expression>,
        source: Handle<crate::Type>,
        destination: Handle<crate::Type>,
        ctx: &mut BlockContext,
        emitter: &mut crate::proc::Emitter,
        block: &mut crate::Block,
        span: Span,
    ) -> Result<Handle<Expression>, Error> {
        use crate::{ScalarKind as K, TypeInner as T};
        let source_inner = ctx.module.types[source].inner.clone();
        let destination_inner = ctx.module.types[destination].inner.clone();
        let vector = matches!(source_inner, T::Vector { .. })
            || matches!(destination_inner, T::Vector { .. });
        let shift = if vector {
            block.extend(emitter.finish(ctx.expressions));
            let shift = ctx
                .expressions
                .append(Expression::Literal(crate::Literal::U32(32)), span);
            emitter.start(ctx.expressions);
            Some(shift)
        } else {
            None
        };
        let u64_ty = ctx.module.types.insert(
            crate::Type {
                name: None,
                inner: T::Scalar(crate::Scalar::U64),
            },
            span,
        );
        // Naga pointer casts go through u64, including SPIR-V two-word bitcasts.
        value = match source_inner {
            T::Pointer {
                space: crate::AddressSpace::PhysicalStorage,
                ..
            } => ctx.expressions.append(
                Expression::PointerCast {
                    expr: value,
                    ty: u64_ty,
                },
                span,
            ),
            T::Scalar(crate::Scalar {
                kind: K::Uint,
                width: 8,
            }) => value,
            T::Scalar(crate::Scalar {
                kind: K::Sint,
                width: 8,
            }) => ctx.expressions.append(
                Expression::As {
                    expr: value,
                    kind: K::Uint,
                    convert: None,
                },
                span,
            ),
            T::Vector {
                size: crate::VectorSize::Bi,
                scalar:
                    crate::Scalar {
                        kind: K::Uint | K::Sint,
                        width: 4,
                    },
            } => {
                let mut parts = [value; 2];
                for (index, part) in parts.iter_mut().enumerate() {
                    let word = ctx.expressions.append(
                        Expression::AccessIndex {
                            base: value,
                            index: index as u32,
                        },
                        span,
                    );
                    let word = ctx.expressions.append(
                        Expression::As {
                            expr: word,
                            kind: K::Uint,
                            convert: None,
                        },
                        span,
                    );
                    *part = ctx.expressions.append(
                        Expression::As {
                            expr: word,
                            kind: K::Uint,
                            convert: Some(8),
                        },
                        span,
                    );
                }
                let high = ctx.expressions.append(
                    Expression::Binary {
                        op: crate::BinaryOperator::ShiftLeft,
                        left: parts[1],
                        right: shift.unwrap(),
                    },
                    span,
                );
                ctx.expressions.append(
                    Expression::Binary {
                        op: crate::BinaryOperator::InclusiveOr,
                        left: parts[0],
                        right: high,
                    },
                    span,
                )
            }
            _ => return Err(Error::InvalidAsType(source)),
        };
        Ok(match destination_inner {
            T::Pointer {
                space: crate::AddressSpace::PhysicalStorage,
                ..
            } => ctx.expressions.append(
                Expression::PointerCast {
                    expr: value,
                    ty: destination,
                },
                span,
            ),
            T::Scalar(crate::Scalar {
                kind: K::Uint,
                width: 8,
            }) => value,
            T::Scalar(crate::Scalar {
                kind: K::Sint,
                width: 8,
            }) => ctx.expressions.append(
                Expression::As {
                    expr: value,
                    kind: K::Sint,
                    convert: None,
                },
                span,
            ),
            T::Vector {
                size: crate::VectorSize::Bi,
                scalar:
                    crate::Scalar {
                        kind: kind @ (K::Uint | K::Sint),
                        width: 4,
                    },
            } => {
                let high = ctx.expressions.append(
                    Expression::Binary {
                        op: crate::BinaryOperator::ShiftRight,
                        left: value,
                        right: shift.unwrap(),
                    },
                    span,
                );
                let mut components = alloc::vec::Vec::new();
                for word in [value, high] {
                    let word = ctx.expressions.append(
                        Expression::As {
                            expr: word,
                            kind: K::Uint,
                            convert: Some(4),
                        },
                        span,
                    );
                    components.push(ctx.expressions.append(
                        Expression::As {
                            expr: word,
                            kind,
                            convert: None,
                        },
                        span,
                    ));
                }
                ctx.expressions.append(
                    Expression::Compose {
                        ty: destination,
                        components,
                    },
                    span,
                )
            }
            _ => return Err(Error::InvalidAsType(destination)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn atomic_pointer(
        &self,
        pointer_id: u32,
        pointer: Handle<Expression>,
        scope_id: u32,
        order_id: u32,
        failure_id: Option<u32>,
        ctx: &mut BlockContext,
        span: Span,
    ) -> Result<Handle<Expression>, Error> {
        use crate::TypeInner as T;
        let pointer_ty = self
            .lookup_type
            .lookup(self.lookup_expression.lookup(pointer_id)?.type_id)?
            .handle;
        let T::Pointer {
            base,
            space: crate::AddressSpace::PhysicalStorage,
        } = ctx.module.types[pointer_ty].inner
        else {
            return Ok(pointer);
        };
        if self.native_scope(scope_id, ctx.module)? != crate::MemoryScope::Device {
            return Err(Error::InvalidOperand);
        }
        let decode = |id| {
            let bits = self.native_constant(id, ctx.module)?;
            Ok(
                match bits & !spirv::MemorySemantics::UNIFORM_MEMORY.bits() {
                    0 => crate::AtomicMemoryOrder::Relaxed,
                    2 => crate::AtomicMemoryOrder::Acquire,
                    4 => crate::AtomicMemoryOrder::Release,
                    8 => crate::AtomicMemoryOrder::AcquireRelease,
                    _ => return Err(Error::InvalidOperand),
                },
            )
        };
        let order = decode(order_id)?;
        let failure_order = failure_id.map(decode).transpose()?;
        let T::Scalar(scalar) = ctx.module.types[base].inner else {
            return Err(Error::InvalidInnerType(pointer_id));
        };
        // SPIR-V has scalar pointees; cast only atomic uses so ordinary aliases retain their types.
        let atomic = ctx.module.types.insert(
            crate::Type {
                name: None,
                inner: T::Atomic(scalar),
            },
            span,
        );
        let atomic_pointer = ctx.module.types.insert(
            crate::Type {
                name: None,
                inner: T::Pointer {
                    base: atomic,
                    space: crate::AddressSpace::PhysicalStorage,
                },
            },
            span,
        );
        let address_ty = ctx.module.types.insert(
            crate::Type {
                name: None,
                inner: T::Scalar(crate::Scalar::U64),
            },
            span,
        );
        let address = ctx.expressions.append(
            Expression::PointerCast {
                expr: pointer,
                ty: address_ty,
            },
            span,
        );
        let pointer = ctx.expressions.append(
            Expression::PointerCast {
                expr: address,
                ty: atomic_pointer,
            },
            span,
        );
        Ok(ctx.expressions.append(
            Expression::AtomicPointer {
                pointer,
                order,
                failure_order,
            },
            span,
        ))
    }

    pub(super) fn native_scope(
        &self,
        id: u32,
        module: &crate::Module,
    ) -> Result<crate::MemoryScope, Error> {
        let value = self.native_constant(id, module)?;
        Ok(match spirv::Scope::from_u32(value) {
            Some(spirv::Scope::Device) => crate::MemoryScope::Device,
            Some(spirv::Scope::QueueFamily) => crate::MemoryScope::QueueFamily,
            Some(spirv::Scope::Workgroup) => crate::MemoryScope::Workgroup,
            Some(spirv::Scope::Subgroup) => crate::MemoryScope::Subgroup,
            Some(spirv::Scope::Invocation) => crate::MemoryScope::Invocation,
            _ => return Err(Error::InvalidOperand),
        })
    }

    pub(super) fn memory_access(
        &mut self,
        mut pointer: Handle<Expression>,
        type_id: u32,
        words: u16,
        store: bool,
        ctx: &mut BlockContext,
        span: Span,
    ) -> Result<Handle<Expression>, Error> {
        use spirv::MemoryAccess as M;
        let physical = ctx.module.types[self.lookup_type.lookup(type_id)?.handle]
            .inner
            .pointer_space()
            == Some(crate::AddressSpace::PhysicalStorage);
        if words == 0 {
            return if physical {
                Err(Error::InvalidOperand)
            } else {
                Ok(pointer)
            };
        }
        let raw = self.next()?;
        let flags = M::from_bits(raw).ok_or(Error::InvalidOperand)?;
        let supported = M::ALIGNED
            | M::NONTEMPORAL
            | M::VOLATILE
            | M::MAKE_POINTER_AVAILABLE
            | M::MAKE_POINTER_VISIBLE
            | M::NON_PRIVATE_POINTER;
        if !(flags - supported).is_empty()
            || (physical && flags.intersects(M::VOLATILE | M::NONTEMPORAL))
        {
            return Err(Error::InvalidOperand);
        }
        let extra = u16::from(flags.contains(M::ALIGNED))
            + u16::from(flags.contains(M::MAKE_POINTER_AVAILABLE))
            + u16::from(flags.contains(M::MAKE_POINTER_VISIBLE));
        if words != 1 + extra {
            return Err(Error::InvalidWordCount);
        }
        let alignment = if flags.contains(M::ALIGNED) {
            Some(self.next()?)
        } else {
            None
        };
        let coherent = flags.intersects(M::MAKE_POINTER_AVAILABLE | M::MAKE_POINTER_VISIBLE);
        if coherent {
            if !physical
                || !self.vulkan_memory_model
                || !flags.contains(M::NON_PRIVATE_POINTER)
                || flags.contains(if store {
                    M::MAKE_POINTER_VISIBLE
                } else {
                    M::MAKE_POINTER_AVAILABLE
                })
            {
                return Err(Error::InvalidOperand);
            }
            let scope_id = self.next()?;
            let scope = self.native_scope(scope_id, ctx.module)?;
            pointer = ctx
                .expressions
                .append(Expression::CoherentPointer { pointer, scope }, span);
        } else if flags.contains(M::NON_PRIVATE_POINTER) {
            return Err(Error::InvalidOperand);
        }
        if physical {
            let alignment = alignment.ok_or(Error::InvalidOperand)?;
            if !alignment.is_power_of_two() {
                return Err(Error::InvalidOperand);
            }
            pointer = ctx
                .expressions
                .append(Expression::PointerAlignment { pointer, alignment }, span);
        }
        Ok(pointer)
    }
}
