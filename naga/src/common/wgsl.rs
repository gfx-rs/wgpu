//! Code shared between the WGSL front and back ends.

use std::fmt::{self, Display, Formatter};

use crate::diagnostic_filter::{
    FilterableTriggeringRule, Severity, StandardFilterableTriggeringRule,
};

impl Severity {
    const ERROR: &'static str = "error";
    const WARNING: &'static str = "warning";
    const INFO: &'static str = "info";
    const OFF: &'static str = "off";

    /// Convert from a sentinel word in WGSL into its associated [`Severity`], if possible.
    pub fn from_wgsl_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::ERROR => Self::Error,
            Self::WARNING => Self::Warning,
            Self::INFO => Self::Info,
            Self::OFF => Self::Off,
            _ => return None,
        })
    }
}

struct DisplayFilterableTriggeringRule<'a>(&'a FilterableTriggeringRule);

impl Display for DisplayFilterableTriggeringRule<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let &Self(inner) = self;
        match *inner {
            FilterableTriggeringRule::Standard(rule) => write!(f, "{}", rule.to_wgsl_ident()),
            FilterableTriggeringRule::Unknown(ref rule) => write!(f, "{rule}"),
            FilterableTriggeringRule::User(ref rules) => {
                let &[ref seg1, ref seg2] = rules.as_ref();
                write!(f, "{seg1}.{seg2}")
            }
        }
    }
}

impl FilterableTriggeringRule {
    /// [`Display`] this rule's identifiers in WGSL.
    pub const fn display_wgsl_ident(&self) -> impl Display + '_ {
        DisplayFilterableTriggeringRule(self)
    }
}

impl StandardFilterableTriggeringRule {
    const DERIVATIVE_UNIFORMITY: &'static str = "derivative_uniformity";

    /// Convert from a sentinel word in WGSL into its associated
    /// [`StandardFilterableTriggeringRule`], if possible.
    pub fn from_wgsl_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::DERIVATIVE_UNIFORMITY => Self::DerivativeUniformity,
            _ => return None,
        })
    }

    /// Maps this [`StandardFilterableTriggeringRule`] into the sentinel word associated with it in
    /// WGSL.
    pub const fn to_wgsl_ident(self) -> &'static str {
        match self {
            Self::DerivativeUniformity => Self::DERIVATIVE_UNIFORMITY,
        }
    }
}

/// A table of all [`RayFlag`] values and their WGSL names.
///
/// [`RayFlag`]: crate::RayFlag
pub static RAYFLAG_NAMES: &[(crate::RayFlag, &str)] = &[
    (crate::RayFlag::FORCE_OPAQUE, "RAY_FLAG_FORCE_OPAQUE"),
    (crate::RayFlag::FORCE_NO_OPAQUE, "RAY_FLAG_FORCE_NO_OPAQUE"),
    (
        crate::RayFlag::TERMINATE_ON_FIRST_HIT,
        "RAY_FLAG_TERMINATE_ON_FIRST_HIT",
    ),
    (
        crate::RayFlag::SKIP_CLOSEST_HIT_SHADER,
        "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER",
    ),
    (
        crate::RayFlag::CULL_BACK_FACING,
        "RAY_FLAG_CULL_BACK_FACING",
    ),
    (
        crate::RayFlag::CULL_FRONT_FACING,
        "RAY_FLAG_CULL_FRONT_FACING",
    ),
    (crate::RayFlag::CULL_OPAQUE, "RAY_FLAG_CULL_OPAQUE"),
    (crate::RayFlag::CULL_NO_OPAQUE, "RAY_FLAG_CULL_NO_OPAQUE"),
    (crate::RayFlag::SKIP_TRIANGLES, "RAY_FLAG_SKIP_TRIANGLES"),
    (crate::RayFlag::SKIP_AABBS, "RAY_FLAG_SKIP_AABBS"),
];

/// A table of all [`RayQueryIntersection`] values and their WGSL names.
///
/// [`RayQueryIntersection`]: crate::RayQueryIntersection
pub static RAYQUERYINTERSECTION_NAMES: &[(crate::RayQueryIntersection, &str)] = &[
    (
        crate::RayQueryIntersection::None,
        "RAY_QUERY_INTERSECTION_NONE",
    ),
    (
        crate::RayQueryIntersection::Triangle,
        "RAY_QUERY_INTERSECTION_TRIANGLE",
    ),
    (
        crate::RayQueryIntersection::Generated,
        "RAY_QUERY_INTERSECTION_GENERATED",
    ),
    (
        crate::RayQueryIntersection::Aabb,
        "RAY_QUERY_INTERSECTION_AABB",
    ),
];
