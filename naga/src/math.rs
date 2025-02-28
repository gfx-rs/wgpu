#[cfg(all(not(feature = "libm"), not(feature = "std")))]
compile_error!("Either the `libm` feature or the `std` feature must be enabled.");

/// Provides standard math operations which can either use the standard library or `libm`.
pub(crate) trait Math: Sized {
    /// Raises a number to a floating point power.
    fn powf(x: Self, y: Self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    fn exp(x: Self) -> Self;

    /// Returns `2^(self)`.
    fn exp2(x: Self) -> Self;

    /// Returns the natural logarithm of the number.
    fn ln(x: Self) -> Self;

    /// Returns the base 2 logarithm of the number.
    fn log2(x: Self) -> Self;

    /// Computes the sine of a number (in radians).
    fn sin(x: Self) -> Self;

    /// Computes the cosine of a number (in radians).
    fn cos(x: Self) -> Self;

    /// Computes the tangent of a number (in radians).
    fn tan(x: Self) -> Self;

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    fn asin(x: Self) -> Self;

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    fn acos(x: Self) -> Self;

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    fn atan(x: Self) -> Self;

    /// Hyperbolic sine function.
    fn sinh(x: Self) -> Self;

    /// Hyperbolic cosine function.
    fn cosh(x: Self) -> Self;

    /// Hyperbolic tangent function.
    fn tanh(x: Self) -> Self;

    /// Inverse hyperbolic sine function.
    fn asinh(x: Self) -> Self;

    /// Inverse hyperbolic cosine function.
    fn acosh(x: Self) -> Self;

    /// Inverse hyperbolic tangent function.
    fn atanh(x: Self) -> Self;

    /// Fused multiply-add. Equivalent to `(x * a) + b`
    fn mul_add(x: Self, a: Self, b: Self) -> Self;

    /// Returns the square root of a number.
    fn sqrt(x: Self) -> Self;

    /// Returns the nearest integer to `x`. If a value is half-way between two integers, round away from `0.0`.
    fn round(x: Self) -> Self;

    /// Returns the largest integer less than or equal to `x`.
    fn floor(x: Self) -> Self;

    /// Returns the integer part of `x`.
    fn trunc(x: Self) -> Self;

    /// Returns the smallest integer greater than or equal to `x`.
    fn ceil(x: Self) -> Self;

    /// Returns the fractional part of `x`.
    #[cfg_attr(
        not(any(msl_out, all(test, feature = "wgsl-in"))),
        expect(dead_code, reason = "function only used with certain features")
    )]
    fn fract(x: Self) -> Self;
}

#[cfg(feature = "libm")]
impl Math for f32 {
    #[inline(always)]
    fn powf(x: f32, y: f32) -> f32 {
        libm::powf(x, y)
    }

    #[inline(always)]
    fn exp(x: f32) -> f32 {
        libm::expf(x)
    }

    #[inline(always)]
    fn exp2(x: f32) -> f32 {
        libm::exp2f(x)
    }

    #[inline(always)]
    fn ln(x: f32) -> f32 {
        // This isn't documented in `libm` but this is actually the base e logarithm.
        libm::logf(x)
    }

    #[inline(always)]
    fn log2(x: f32) -> f32 {
        libm::log2f(x)
    }

    #[inline(always)]
    fn sin(x: f32) -> f32 {
        libm::sinf(x)
    }

    #[inline(always)]
    fn cos(x: f32) -> f32 {
        libm::cosf(x)
    }

    #[inline(always)]
    fn tan(x: f32) -> f32 {
        libm::tanf(x)
    }

    #[inline(always)]
    fn asin(x: f32) -> f32 {
        libm::asinf(x)
    }

    #[inline(always)]
    fn acos(x: f32) -> f32 {
        libm::acosf(x)
    }

    #[inline(always)]
    fn atan(x: f32) -> f32 {
        libm::atanf(x)
    }

    #[inline(always)]
    fn sinh(x: f32) -> f32 {
        libm::sinhf(x)
    }

    #[inline(always)]
    fn cosh(x: f32) -> f32 {
        libm::coshf(x)
    }

    #[inline(always)]
    fn tanh(x: f32) -> f32 {
        libm::tanhf(x)
    }

    #[inline(always)]
    fn asinh(x: f32) -> f32 {
        libm::asinhf(x)
    }

    #[inline(always)]
    fn acosh(x: f32) -> f32 {
        libm::acoshf(x)
    }

    #[inline(always)]
    fn atanh(x: f32) -> f32 {
        libm::atanhf(x)
    }

    #[inline(always)]
    fn sqrt(x: f32) -> f32 {
        libm::sqrtf(x)
    }

    #[inline(always)]
    fn round(x: f32) -> f32 {
        libm::roundf(x)
    }

    #[inline(always)]
    fn floor(x: f32) -> f32 {
        libm::floorf(x)
    }

    #[inline(always)]
    fn mul_add(x: f32, a: f32, b: f32) -> f32 {
        libm::fmaf(x, a, b)
    }

    #[inline(always)]
    fn trunc(x: f32) -> f32 {
        libm::truncf(x)
    }

    #[inline(always)]
    fn ceil(x: f32) -> f32 {
        libm::ceilf(x)
    }

    #[inline(always)]
    fn fract(x: f32) -> f32 {
        libm::modff(x).0
    }
}

#[cfg(all(not(feature = "libm"), feature = "std"))]
impl Math for f32 {
    #[inline(always)]
    fn powf(x: f32, y: f32) -> f32 {
        f32::powf(x, y)
    }

    #[inline(always)]
    fn exp(x: f32) -> f32 {
        f32::exp(x)
    }

    #[inline(always)]
    fn exp2(x: f32) -> f32 {
        f32::exp2(x)
    }

    #[inline(always)]
    fn ln(x: f32) -> f32 {
        f32::ln(x)
    }

    #[inline(always)]
    fn log2(x: f32) -> f32 {
        f32::log2(x)
    }

    #[inline(always)]
    fn sin(x: f32) -> f32 {
        f32::sin(x)
    }

    #[inline(always)]
    fn cos(x: f32) -> f32 {
        f32::cos(x)
    }

    #[inline(always)]
    fn tan(x: f32) -> f32 {
        f32::tan(x)
    }

    #[inline(always)]
    fn asin(x: f32) -> f32 {
        f32::asin(x)
    }

    #[inline(always)]
    fn acos(x: f32) -> f32 {
        f32::acos(x)
    }

    #[inline(always)]
    fn atan(x: f32) -> f32 {
        f32::atan(x)
    }

    #[inline(always)]
    fn sinh(x: f32) -> f32 {
        f32::sinh(x)
    }

    #[inline(always)]
    fn cosh(x: f32) -> f32 {
        f32::cosh(x)
    }

    #[inline(always)]
    fn tanh(x: f32) -> f32 {
        f32::tanh(x)
    }

    #[inline(always)]
    fn asinh(x: f32) -> f32 {
        f32::asinh(x)
    }

    #[inline(always)]
    fn acosh(x: f32) -> f32 {
        f32::acosh(x)
    }

    #[inline(always)]
    fn atanh(x: f32) -> f32 {
        f32::atanh(x)
    }

    #[inline(always)]
    fn sqrt(x: f32) -> f32 {
        f32::sqrt(x)
    }

    #[inline(always)]
    fn round(x: f32) -> f32 {
        f32::round(x)
    }

    #[inline(always)]
    fn floor(x: f32) -> f32 {
        f32::floor(x)
    }

    #[inline(always)]
    fn mul_add(x: f32, a: f32, b: f32) -> f32 {
        f32::mul_add(x, a, b)
    }

    #[inline(always)]
    fn trunc(x: f32) -> f32 {
        f32::trunc(x)
    }

    #[inline(always)]
    fn ceil(x: f32) -> f32 {
        f32::ceil(x)
    }

    #[inline(always)]
    fn fract(x: f32) -> f32 {
        f32::fract(x)
    }
}

#[cfg(feature = "libm")]
impl Math for f64 {
    #[inline(always)]
    fn powf(x: f64, y: f64) -> f64 {
        libm::pow(x, y)
    }

    #[inline(always)]
    fn exp(x: f64) -> f64 {
        libm::exp(x)
    }

    #[inline(always)]
    fn exp2(x: f64) -> f64 {
        libm::exp2(x)
    }

    #[inline(always)]
    fn ln(x: f64) -> f64 {
        // This isn't documented in `libm` but this is actually the base e logarithm.
        libm::log(x)
    }

    #[inline(always)]
    fn log2(x: f64) -> f64 {
        libm::log2(x)
    }

    #[inline(always)]
    fn sin(x: f64) -> f64 {
        libm::sin(x)
    }

    #[inline(always)]
    fn cos(x: f64) -> f64 {
        libm::cos(x)
    }

    #[inline(always)]
    fn tan(x: f64) -> f64 {
        libm::tan(x)
    }

    #[inline(always)]
    fn asin(x: f64) -> f64 {
        libm::asin(x)
    }

    #[inline(always)]
    fn acos(x: f64) -> f64 {
        libm::acos(x)
    }

    #[inline(always)]
    fn atan(x: f64) -> f64 {
        libm::atan(x)
    }

    #[inline(always)]
    fn sinh(x: f64) -> f64 {
        libm::sinh(x)
    }

    #[inline(always)]
    fn cosh(x: f64) -> f64 {
        libm::cosh(x)
    }

    #[inline(always)]
    fn tanh(x: f64) -> f64 {
        libm::tanh(x)
    }

    #[inline(always)]
    fn asinh(x: f64) -> f64 {
        libm::asinh(x)
    }

    #[inline(always)]
    fn acosh(x: f64) -> f64 {
        libm::acosh(x)
    }

    #[inline(always)]
    fn atanh(x: f64) -> f64 {
        libm::atanh(x)
    }

    #[inline(always)]
    fn sqrt(x: f64) -> f64 {
        libm::sqrt(x)
    }

    #[inline(always)]
    fn round(x: f64) -> f64 {
        libm::round(x)
    }

    #[inline(always)]
    fn floor(x: f64) -> f64 {
        libm::floor(x)
    }

    #[inline(always)]
    fn mul_add(x: f64, a: f64, b: f64) -> f64 {
        libm::fma(x, a, b)
    }

    #[inline(always)]
    fn trunc(x: f64) -> f64 {
        libm::trunc(x)
    }

    #[inline(always)]
    fn ceil(x: f64) -> f64 {
        libm::ceil(x)
    }

    #[inline(always)]
    fn fract(x: f64) -> f64 {
        libm::modf(x).0
    }
}

#[cfg(all(not(feature = "libm"), feature = "std"))]
impl Math for f64 {
    #[inline(always)]
    fn powf(x: f64, y: f64) -> f64 {
        f64::powf(x, y)
    }

    #[inline(always)]
    fn exp(x: f64) -> f64 {
        f64::exp(x)
    }

    #[inline(always)]
    fn exp2(x: f64) -> f64 {
        f64::exp2(x)
    }

    #[inline(always)]
    fn ln(x: f64) -> f64 {
        f64::ln(x)
    }

    #[inline(always)]
    fn log2(x: f64) -> f64 {
        f64::log2(x)
    }

    #[inline(always)]
    fn sin(x: f64) -> f64 {
        f64::sin(x)
    }

    #[inline(always)]
    fn cos(x: f64) -> f64 {
        f64::cos(x)
    }

    #[inline(always)]
    fn tan(x: f64) -> f64 {
        f64::tan(x)
    }

    #[inline(always)]
    fn asin(x: f64) -> f64 {
        f64::asin(x)
    }

    #[inline(always)]
    fn acos(x: f64) -> f64 {
        f64::acos(x)
    }

    #[inline(always)]
    fn atan(x: f64) -> f64 {
        f64::atan(x)
    }

    #[inline(always)]
    fn sinh(x: f64) -> f64 {
        f64::sinh(x)
    }

    #[inline(always)]
    fn cosh(x: f64) -> f64 {
        f64::cosh(x)
    }

    #[inline(always)]
    fn tanh(x: f64) -> f64 {
        f64::tanh(x)
    }

    #[inline(always)]
    fn asinh(x: f64) -> f64 {
        f64::asinh(x)
    }

    #[inline(always)]
    fn acosh(x: f64) -> f64 {
        f64::acosh(x)
    }

    #[inline(always)]
    fn atanh(x: f64) -> f64 {
        f64::atanh(x)
    }

    #[inline(always)]
    fn sqrt(x: f64) -> f64 {
        f64::sqrt(x)
    }

    #[inline(always)]
    fn round(x: f64) -> f64 {
        f64::round(x)
    }

    #[inline(always)]
    fn floor(x: f64) -> f64 {
        f64::floor(x)
    }

    #[inline(always)]
    fn mul_add(x: f64, a: f64, b: f64) -> f64 {
        f64::mul_add(x, a, b)
    }

    #[inline(always)]
    fn trunc(x: f64) -> f64 {
        f64::trunc(x)
    }

    #[inline(always)]
    fn ceil(x: f64) -> f64 {
        f64::ceil(x)
    }

    #[inline(always)]
    fn fract(x: f64) -> f64 {
        f64::fract(x)
    }
}
