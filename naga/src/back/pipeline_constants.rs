use super::PipelineConstants;
use crate::{Constant, Expression, Literal, Module, Scalar, Span, TypeInner};
use std::borrow::Cow;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum PipelineConstantError {
    #[error("Missing value for pipeline-overridable constant with identifier string: '{0}'")]
    MissingValue(String),
    #[error("Source f64 value needs to be finite (NaNs and Inifinites are not allowed) for number destinations")]
    SrcNeedsToBeFinite,
    #[error("Source f64 value doesn't fit in destination")]
    DstRangeTooSmall,
}

pub(super) fn process_overrides<'a>(
    module: &'a Module,
    pipeline_constants: &PipelineConstants,
) -> Result<Cow<'a, Module>, PipelineConstantError> {
    if module.overrides.is_empty() {
        return Ok(Cow::Borrowed(module));
    }

    let mut module = module.clone();

    for (_handle, override_, span) in module.overrides.drain() {
        let key = if let Some(id) = override_.id {
            Cow::Owned(id.to_string())
        } else if let Some(ref name) = override_.name {
            Cow::Borrowed(name)
        } else {
            unreachable!();
        };
        let init = if let Some(value) = pipeline_constants.get::<str>(&key) {
            let literal = match module.types[override_.ty].inner {
                TypeInner::Scalar(scalar) => map_value_to_literal(*value, scalar)?,
                _ => unreachable!(),
            };
            module
                .const_expressions
                .append(Expression::Literal(literal), Span::UNDEFINED)
        } else if let Some(init) = override_.init {
            init
        } else {
            return Err(PipelineConstantError::MissingValue(key.to_string()));
        };
        let constant = Constant {
            name: override_.name,
            ty: override_.ty,
            init,
        };
        module.constants.append(constant, span);
    }

    Ok(Cow::Owned(module))
}

fn map_value_to_literal(value: f64, scalar: Scalar) -> Result<Literal, PipelineConstantError> {
    // note that in rust 0.0 == -0.0
    match scalar {
        Scalar::BOOL => {
            // https://webidl.spec.whatwg.org/#js-boolean
            let value = value != 0.0 && !value.is_nan();
            Ok(Literal::Bool(value))
        }
        Scalar::I32 => {
            // https://webidl.spec.whatwg.org/#js-long
            if !value.is_finite() {
                return Err(PipelineConstantError::SrcNeedsToBeFinite);
            }

            let value = value.trunc();
            if value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
                return Err(PipelineConstantError::DstRangeTooSmall);
            }

            let value = value as i32;
            Ok(Literal::I32(value))
        }
        Scalar::U32 => {
            // https://webidl.spec.whatwg.org/#js-unsigned-long
            if !value.is_finite() {
                return Err(PipelineConstantError::SrcNeedsToBeFinite);
            }

            let value = value.trunc();
            if value < f64::from(u32::MIN) || value > f64::from(u32::MAX) {
                return Err(PipelineConstantError::DstRangeTooSmall);
            }

            let value = value as u32;
            Ok(Literal::U32(value))
        }
        Scalar::F32 => {
            // https://webidl.spec.whatwg.org/#js-float
            if !value.is_finite() {
                return Err(PipelineConstantError::SrcNeedsToBeFinite);
            }

            let value = value as f32;
            if !value.is_finite() {
                return Err(PipelineConstantError::DstRangeTooSmall);
            }

            Ok(Literal::F32(value))
        }
        Scalar::F64 => {
            // https://webidl.spec.whatwg.org/#js-double
            if !value.is_finite() {
                return Err(PipelineConstantError::SrcNeedsToBeFinite);
            }

            Ok(Literal::F64(value))
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_map_value_to_literal() {
    let bool_test_cases = [
        (0.0, false),
        (-0.0, false),
        (f64::NAN, false),
        (1.0, true),
        (f64::INFINITY, true),
        (f64::NEG_INFINITY, true),
    ];
    for (value, out) in bool_test_cases {
        let res = Ok(Literal::Bool(out));
        assert_eq!(map_value_to_literal(value, Scalar::BOOL), res);
    }

    for scalar in [Scalar::I32, Scalar::U32, Scalar::F32, Scalar::F64] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let res = Err(PipelineConstantError::SrcNeedsToBeFinite);
            assert_eq!(map_value_to_literal(value, scalar), res);
        }
    }

    // i32
    assert_eq!(
        map_value_to_literal(f64::from(i32::MIN), Scalar::I32),
        Ok(Literal::I32(i32::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::from(i32::MAX), Scalar::I32),
        Ok(Literal::I32(i32::MAX))
    );
    assert_eq!(
        map_value_to_literal(f64::from(i32::MIN) - 1.0, Scalar::I32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );
    assert_eq!(
        map_value_to_literal(f64::from(i32::MAX) + 1.0, Scalar::I32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );

    // u32
    assert_eq!(
        map_value_to_literal(f64::from(u32::MIN), Scalar::U32),
        Ok(Literal::U32(u32::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::from(u32::MAX), Scalar::U32),
        Ok(Literal::U32(u32::MAX))
    );
    assert_eq!(
        map_value_to_literal(f64::from(u32::MIN) - 1.0, Scalar::U32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );
    assert_eq!(
        map_value_to_literal(f64::from(u32::MAX) + 1.0, Scalar::U32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );

    // f32
    assert_eq!(
        map_value_to_literal(f64::from(f32::MIN), Scalar::F32),
        Ok(Literal::F32(f32::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::from(f32::MAX), Scalar::F32),
        Ok(Literal::F32(f32::MAX))
    );
    assert_eq!(
        map_value_to_literal(-f64::from_bits(0x47efffffefffffff), Scalar::F32),
        Ok(Literal::F32(f32::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::from_bits(0x47efffffefffffff), Scalar::F32),
        Ok(Literal::F32(f32::MAX))
    );
    assert_eq!(
        map_value_to_literal(-f64::from_bits(0x47effffff0000000), Scalar::F32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );
    assert_eq!(
        map_value_to_literal(f64::from_bits(0x47effffff0000000), Scalar::F32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );

    // f64
    assert_eq!(
        map_value_to_literal(f64::MIN, Scalar::F64),
        Ok(Literal::F64(f64::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::MAX, Scalar::F64),
        Ok(Literal::F64(f64::MAX))
    );
}
