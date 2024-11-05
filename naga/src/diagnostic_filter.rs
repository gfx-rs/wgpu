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

/// A filter that modifies how diagnostics are emitted for shaders.
///
/// <https://www.w3.org/TR/WGSL/#diagnostic-filter>
#[derive(Clone, Debug)]
pub struct DiagnosticFilter {
    pub new_severity: Severity,
    pub triggering_rule: FilterableTriggeringRule,
}
