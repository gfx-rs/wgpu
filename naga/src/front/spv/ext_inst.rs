use super::{Error, LookupExpression, LookupHelper as _};

struct ExtInst {
    result_type_id: spirv::Word,
    result_id: spirv::Word,
    set_id: spirv::Word,
    inst_id: spirv::Word,
}

impl<I: Iterator<Item = u32>> super::Frontend<I> {
    pub(super) fn parse_ext_inst(
        &mut self,
        inst: super::Instruction,
        span: crate::Span,
        ctx: &mut super::BlockContext,
        emitter: &mut crate::proc::Emitter,
        block: &mut crate::Block,
        block_id: spirv::Word,
        body_idx: usize,
    ) -> Result<(), Error> {
        let base_wc = 5;
        inst.expect_at_least(base_wc)?;

        let ext_inst = ExtInst {
            result_type_id: self.next()?,
            result_id: self.next()?,
            set_id: self.next()?,
            inst_id: self.next()?,
        };
        let ext_name = if let Some(name) = self.ext_inst_imports.get(&ext_inst.set_id) {
            name
        } else {
            return Err(Error::UnsupportedExtInstSet(ext_inst.set_id));
        };

        match *ext_name {
            "GLSL.std.450" => self.parse_ext_inst_glsl_std(
                inst, ext_inst, span, ctx, emitter, block, block_id, body_idx,
            ),
            _ => {
                return Err(Error::UnsupportedExtInstSet(ext_inst.set_id));
            }
        }
    }
    fn parse_ext_inst_glsl_std(
        &mut self,
        inst: super::Instruction,
        ext_inst: ExtInst,
        span: crate::Span,
        ctx: &mut super::BlockContext,
        emitter: &mut crate::proc::Emitter,
        block: &mut crate::Block,
        block_id: spirv::Word,
        body_idx: usize,
    ) -> Result<(), Error> {
        use crate::MathFunction as Mf;
        use spirv::GLOp as Glo;

        let base_wc = 5;

        let gl_op =
            Glo::from_u32(ext_inst.inst_id).ok_or(Error::UnsupportedExtInst(ext_inst.inst_id))?;

        let fun = match gl_op {
            Glo::Round => Mf::Round,
            Glo::RoundEven => Mf::Round,
            Glo::Trunc => Mf::Trunc,
            Glo::FAbs | Glo::SAbs => Mf::Abs,
            Glo::FSign | Glo::SSign => Mf::Sign,
            Glo::Floor => Mf::Floor,
            Glo::Ceil => Mf::Ceil,
            Glo::Fract => Mf::Fract,
            Glo::Sin => Mf::Sin,
            Glo::Cos => Mf::Cos,
            Glo::Tan => Mf::Tan,
            Glo::Asin => Mf::Asin,
            Glo::Acos => Mf::Acos,
            Glo::Atan => Mf::Atan,
            Glo::Sinh => Mf::Sinh,
            Glo::Cosh => Mf::Cosh,
            Glo::Tanh => Mf::Tanh,
            Glo::Atan2 => Mf::Atan2,
            Glo::Asinh => Mf::Asinh,
            Glo::Acosh => Mf::Acosh,
            Glo::Atanh => Mf::Atanh,
            Glo::Radians => Mf::Radians,
            Glo::Degrees => Mf::Degrees,
            Glo::Pow => Mf::Pow,
            Glo::Exp => Mf::Exp,
            Glo::Log => Mf::Log,
            Glo::Exp2 => Mf::Exp2,
            Glo::Log2 => Mf::Log2,
            Glo::Sqrt => Mf::Sqrt,
            Glo::InverseSqrt => Mf::InverseSqrt,
            Glo::MatrixInverse => Mf::Inverse,
            Glo::Determinant => Mf::Determinant,
            Glo::ModfStruct => Mf::Modf,
            Glo::FMin | Glo::UMin | Glo::SMin | Glo::NMin => Mf::Min,
            Glo::FMax | Glo::UMax | Glo::SMax | Glo::NMax => Mf::Max,
            Glo::FClamp | Glo::UClamp | Glo::SClamp | Glo::NClamp => Mf::Clamp,
            Glo::FMix => Mf::Mix,
            Glo::Step => Mf::Step,
            Glo::SmoothStep => Mf::SmoothStep,
            Glo::Fma => Mf::Fma,
            Glo::FrexpStruct => Mf::Frexp,
            Glo::Ldexp => Mf::Ldexp,
            Glo::Length => Mf::Length,
            Glo::Distance => Mf::Distance,
            Glo::Cross => Mf::Cross,
            Glo::Normalize => Mf::Normalize,
            Glo::FaceForward => Mf::FaceForward,
            Glo::Reflect => Mf::Reflect,
            Glo::Refract => Mf::Refract,
            Glo::PackUnorm4x8 => Mf::Pack4x8unorm,
            Glo::PackSnorm4x8 => Mf::Pack4x8snorm,
            Glo::PackHalf2x16 => Mf::Pack2x16float,
            Glo::PackUnorm2x16 => Mf::Pack2x16unorm,
            Glo::PackSnorm2x16 => Mf::Pack2x16snorm,
            Glo::UnpackUnorm4x8 => Mf::Unpack4x8unorm,
            Glo::UnpackSnorm4x8 => Mf::Unpack4x8snorm,
            Glo::UnpackHalf2x16 => Mf::Unpack2x16float,
            Glo::UnpackUnorm2x16 => Mf::Unpack2x16unorm,
            Glo::UnpackSnorm2x16 => Mf::Unpack2x16snorm,
            Glo::FindILsb => Mf::FindLsb,
            Glo::FindUMsb | Glo::FindSMsb => Mf::FindMsb,
            // TODO: https://github.com/gfx-rs/naga/issues/2526
            Glo::Modf | Glo::Frexp => return Err(Error::UnsupportedExtInst(ext_inst.inst_id)),
            Glo::IMix
            | Glo::PackDouble2x32
            | Glo::UnpackDouble2x32
            | Glo::InterpolateAtCentroid
            | Glo::InterpolateAtSample
            | Glo::InterpolateAtOffset => return Err(Error::UnsupportedExtInst(ext_inst.inst_id)),
        };

        let arg_count = fun.argument_count();
        inst.expect(base_wc + arg_count as u16)?;
        let arg = {
            let arg_id = self.next()?;
            let lexp = self.lookup_expression.lookup(arg_id)?;
            self.get_expr_handle(arg_id, lexp, ctx, emitter, block, body_idx)
        };
        let arg1 = if arg_count > 1 {
            let arg_id = self.next()?;
            let lexp = self.lookup_expression.lookup(arg_id)?;
            Some(self.get_expr_handle(arg_id, lexp, ctx, emitter, block, body_idx))
        } else {
            None
        };
        let arg2 = if arg_count > 2 {
            let arg_id = self.next()?;
            let lexp = self.lookup_expression.lookup(arg_id)?;
            Some(self.get_expr_handle(arg_id, lexp, ctx, emitter, block, body_idx))
        } else {
            None
        };
        let arg3 = if arg_count > 3 {
            let arg_id = self.next()?;
            let lexp = self.lookup_expression.lookup(arg_id)?;
            Some(self.get_expr_handle(arg_id, lexp, ctx, emitter, block, body_idx))
        } else {
            None
        };

        let expr = crate::Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        };
        self.lookup_expression.insert(
            ext_inst.result_id,
            LookupExpression {
                handle: ctx.expressions.append(expr, span),
                type_id: ext_inst.result_type_id,
                block_id,
            },
        );
        Ok(())
    }
}
