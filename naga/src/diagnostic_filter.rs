//! [`DiagnosticFilter`]s and supporting functionality.

/// A severity set on a [`DiagnosticFilter`].
///
/// <https://www.w3.org/TR/WGSL/#diagnostic-severity>
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Severity {
    Off,
    Info,
    Warning,
    Error,
}

impl Severity {
    const ERROR: &'static str = "error";
    const WARNING: &'static str = "warning";
    const INFO: &'static str = "info";
    const OFF: &'static str = "off";

    /// Convert from a sentinel word in WGSL into its associated [`Severity`], if possible.
    pub fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::ERROR => Self::Error,
            Self::WARNING => Self::Warning,
            Self::INFO => Self::Info,
            Self::OFF => Self::Off,
            _ => return None,
        })
    }

    /// Checks whether this severity is [`Self::Error`].
    ///
    /// Naga does not yet support diagnostic items at lesser severities than
    /// [`Severity::Error`]. When this is implemented, this method should be deleted, and the
    /// severity should be used directly for reporting diagnostics.
    #[cfg(feature = "wgsl-in")]
    pub(crate) fn report_diag<E>(
        self,
        err: E,
        log_handler: impl FnOnce(E, log::Level),
    ) -> Result<(), E> {
        let log_level = match self {
            Severity::Off => return Ok(()),

            // NOTE: These severities are not yet reported.
            Severity::Info => log::Level::Info,
            Severity::Warning => log::Level::Warn,

            Severity::Error => return Err(err),
        };
        log_handler(err, log_level);
        Ok(())
    }
}

/// A filterable triggering rule in a [`DiagnosticFilter`].
///
/// <https://www.w3.org/TR/WGSL/#filterable-triggering-rules>
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FilterableTriggeringRule {
    DerivativeUniformity,
}

impl FilterableTriggeringRule {
    const DERIVATIVE_UNIFORMITY: &'static str = "derivative_uniformity";

    /// Convert from a sentinel word in WGSL into its associated [`FilterableTriggeringRule`], if possible.
    pub fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::DERIVATIVE_UNIFORMITY => Self::DerivativeUniformity,
            _ => return None,
        })
    }

    /// Maps this [`FilterableTriggeringRule`] into the sentinel word associated with it in WGSL.
    pub const fn to_ident(self) -> &'static str {
        match self {
            Self::DerivativeUniformity => Self::DERIVATIVE_UNIFORMITY,
        }
    }

    #[cfg(feature = "wgsl-in")]
    pub(crate) const fn tracking_issue_num(self) -> u16 {
        match self {
            FilterableTriggeringRule::DerivativeUniformity => 5320,
        }
    }
}

/// A filter that modifies how diagnostics are emitted for shaders.
///
/// <https://www.w3.org/TR/WGSL/#diagnostic-filter>
#[derive(Clone, Debug)]
pub struct DiagnosticFilter {
    pub new_severity: Severity,
    pub triggering_rule: FilterableTriggeringRule,
}
