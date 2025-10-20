use alloc::{boxed::Box, vec::Vec};

use crate::{
    front::wgsl::{
        error::Error,
        lower::{ExpressionContext, Lowerer, Result},
        parse::{ast, conv},
    },
    ir, Handle, Span,
};

/// Iterator over a template list.
///
/// All functions will attempt to consume an element in the list.
///
/// Function variants prefixed with "maybe" will not return an error if there
/// are no more elements left in the list.
pub struct TemplateListIter<'iter, 'source> {
    ident_span: Span,
    template_list: core::slice::Iter<'iter, Handle<ast::Expression<'source>>>,
}

impl<'iter, 'source> TemplateListIter<'iter, 'source> {
    pub fn new(ident_span: Span, template_list: &'iter [Handle<ast::Expression<'source>>]) -> Self {
        Self {
            ident_span,
            template_list: template_list.iter(),
        }
    }

    pub fn finish(self, ctx: &ExpressionContext<'source, '_, '_>) -> Result<'source, ()> {
        let unused_args: Vec<Span> = self
            .template_list
            .map(|expr| ctx.ast_expressions.get_span(*expr))
            .collect();
        if unused_args.is_empty() {
            Ok(())
        } else {
            Err(Box::new(Error::UnusedArgsForTemplate(unused_args)))
        }
    }

    fn expect_next(
        &mut self,
        arg: &'static str,
    ) -> Result<'source, Handle<ast::Expression<'source>>> {
        if let Some(expr) = self.template_list.next() {
            Ok(*expr)
        } else {
            Err(Box::new(Error::MissingTemplateArg {
                span: self.ident_span,
                arg,
            }))
        }
    }

    pub fn ty(
        &mut self,
        lowerer: &mut Lowerer<'source, '_>,
        ctx: &mut ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, Handle<ir::Type>> {
        let expr = self.expect_next("`T`, a type")?;
        lowerer.type_expression(expr, ctx)
    }

    pub fn ty_with_span(
        &mut self,
        lowerer: &mut Lowerer<'source, '_>,
        ctx: &mut ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, (Handle<ir::Type>, Span)> {
        let expr = self.expect_next("`T`, a type")?;
        let span = ctx.ast_expressions.get_span(expr);
        let ty = lowerer.type_expression(expr, ctx)?;
        Ok((ty, span))
    }

    pub fn scalar(
        &mut self,
        lowerer: &mut Lowerer<'source, '_>,
        ctx: &mut ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, (ir::Scalar, Span)> {
        let expr = self.expect_next("`T`, a scalar type")?;
        let ty = lowerer.type_expression(expr, ctx)?;
        let span = ctx.ast_expressions.get_span(expr);
        match ctx.module.types[ty].inner {
            ir::TypeInner::Scalar(scalar) => Ok((scalar, span)),
            _ => Err(Box::new(Error::UnknownScalarType(span))),
        }
    }

    pub fn maybe_array_size(
        &mut self,
        lowerer: &mut Lowerer<'source, '_>,
        ctx: &mut ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, ir::ArraySize> {
        if let Some(expr) = self.template_list.next() {
            lowerer.array_size2(*expr, ctx)
        } else {
            Ok(ir::ArraySize::Dynamic)
        }
    }

    pub fn address_space(
        &mut self,
        ctx: &ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, ir::AddressSpace> {
        let expr = self.expect_next("`AS`, an address space")?;
        let (enumerant, span) = Lowerer::enumerant(expr, ctx)?;
        conv::map_address_space(enumerant, span, &ctx.enable_extensions)
    }
    pub fn maybe_address_space(
        &mut self,
        ctx: &ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, Option<ir::AddressSpace>> {
        if let Some(expr) = self.template_list.next() {
            let (enumerant, span) = Lowerer::enumerant(*expr, ctx)?;
            Ok(Some(conv::map_address_space(
                enumerant,
                span,
                &ctx.enable_extensions,
            )?))
        } else {
            Ok(None)
        }
    }

    pub fn access_mode(
        &mut self,
        ctx: &ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, ir::StorageAccess> {
        let expr = self.expect_next("`Access`, an access mode")?;
        let (enumerant, span) = Lowerer::enumerant(expr, ctx)?;
        conv::map_access_mode(enumerant, span)
    }
    pub fn maybe_access_mode(
        &mut self,
        space: &mut ir::AddressSpace,
        ctx: &ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, ()> {
        if let &mut ir::AddressSpace::Storage { ref mut access } = space {
            if let Some(expr) = self.template_list.next() {
                let (enumerant, span) = Lowerer::enumerant(*expr, ctx)?;
                let access_mode = conv::map_access_mode(enumerant, span)?;
                *access = access_mode;
            } else {
                // defaulting to `read`
                *access = ir::StorageAccess::LOAD
            }
        }
        Ok(())
    }

    pub fn storage_format(
        &mut self,
        ctx: &ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, ir::StorageFormat> {
        let expr = self.expect_next("`Format`, a texel format")?;
        let (enumerant, span) = Lowerer::enumerant(expr, ctx)?;
        conv::map_storage_format(enumerant, span)
    }

    pub fn maybe_vertex_return(
        &mut self,
        ctx: &ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, bool> {
        if let Some(expr) = self.template_list.next() {
            let (enumerant, span) = Lowerer::enumerant(*expr, ctx)?;
            conv::map_ray_flag(enumerant, span)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn cooperative_role(
        &mut self,
        ctx: &ExpressionContext<'source, '_, '_>,
    ) -> Result<'source, crate::CooperativeRole> {
        let role_expr = self.expect_next("`Role`, a cooperative matrix role")?;
        let (enumerant, span) = Lowerer::enumerant(role_expr, ctx)?;
        let role = conv::map_cooperative_role(enumerant, span)?;
        Ok(role)
    }
}
