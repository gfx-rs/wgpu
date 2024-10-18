//! Functionality for managing [`DiagnosticFilter`]s.
//!
//! This functionality is typically used by front ends via [`DiagnosticFilterMap`].

use std::str::FromStr;

use crate::{Handle, Span};

#[cfg(feature = "arbitrary")]
use arbitrary::Arbitrary;
use indexmap::IndexMap;
#[cfg(feature = "deserialize")]
use serde::Deserialize;
#[cfg(feature = "serialize")]
use serde::Serialize;

/// A severity set on a [`DiagnosticFilter`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
#[cfg_attr(test, derive(strum::EnumIter))]
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

    pub fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::ERROR => Self::Error,
            Self::WARNING => Self::Warning,
            Self::INFO => Self::Info,
            Self::OFF => Self::Off,
            _ => return None,
        })
    }

    #[cfg(test)]
    pub fn to_ident(self) -> &'static str {
        match self {
            Self::Error => Self::ERROR,
            Self::Warning => Self::WARNING,
            Self::Info => Self::INFO,
            Self::Off => Self::OFF,
        }
    }
}

impl FromStr for Severity {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_ident(s).ok_or(())
    }
}

/// The rule being configured in a [`DiagnosticFilter`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum DiagnosticTriggeringRule {
    DerivativeUniformity,
}

impl DiagnosticTriggeringRule {
    const DERIVATIVE_UNIFORMITY: &'static str = "derivative_uniformity";

    pub fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::DERIVATIVE_UNIFORMITY => Self::DerivativeUniformity,
            _ => return None,
        })
    }

    pub fn to_ident(self) -> &'static str {
        match self {
            Self::DerivativeUniformity => Self::DERIVATIVE_UNIFORMITY,
        }
    }
}

/// A filter that modifies how diagnostics are emitted for shaders.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct DiagnosticFilter {
    pub new_severity: Severity,
    pub triggering_rule: DiagnosticTriggeringRule,
}

/// A map of diagnostic filters parsed in a single parse site where filters can be specified in
/// source. Intended for front ends' first step into storing parsed [`DiagnosticFilter`]s.
#[derive(Clone, Debug, Default)]
pub(crate) struct DiagnosticFilterMap(IndexMap<DiagnosticTriggeringRule, (Severity, Span)>);

#[cfg(feature = "wgsl-in")]
impl DiagnosticFilterMap {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add the given `diagnostic_filter` parsed at the given `span` to this map.
    pub(crate) fn add(
        &mut self,
        diagnostic_filter: DiagnosticFilter,
        span: Span,
    ) -> Result<(), ConflictingDiagnosticRuleError> {
        use indexmap::map::Entry;

        let &mut Self(ref mut diagnostic_filters) = self;
        let DiagnosticFilter {
            new_severity,
            triggering_rule,
        } = diagnostic_filter;

        match diagnostic_filters.entry(triggering_rule) {
            Entry::Vacant(entry) => {
                entry.insert((new_severity, span));
                Ok(())
            }
            Entry::Occupied(entry) => {
                return Err(ConflictingDiagnosticRuleError {
                    triggering_rule,
                    triggering_rule_spans: [entry.get().1, span],
                })
            }
        }
    }
}

/// An error yielded by [`DiagnosticFilterMap::add`].
#[cfg(feature = "wgsl-in")]
impl IntoIterator for DiagnosticFilterMap {
    type Item = (DiagnosticTriggeringRule, (Severity, Span));

    type IntoIter = indexmap::map::IntoIter<DiagnosticTriggeringRule, (Severity, Span)>;

    fn into_iter(self) -> Self::IntoIter {
        let Self(this) = self;
        this.into_iter()
    }
}

#[cfg(feature = "wgsl-in")]
#[derive(Clone, Debug)]
pub(crate) struct ConflictingDiagnosticRuleError {
    pub triggering_rule: DiagnosticTriggeringRule,
    pub triggering_rule_spans: [Span; 2],
}

/// Represents a single link in a linked list of [`DiagnosticFilter`]s backed by a
/// [`crate::Arena`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct DiagnosticFilterNode {
    pub inner: DiagnosticFilter,
    pub parent: Option<Handle<DiagnosticFilterNode>>,
}

#[cfg(test)]
mod test {
    use crate::front::wgsl::assert_parse_err;

    use super::{DiagnosticTriggeringRule, Severity};

    use itertools::Itertools as _;
    use strum::IntoEnumIterator as _;

    #[test]
    fn basic() {}

    #[test]
    fn malformed() {
        assert_parse_err("directive;", snapshot);
        assert_parse_err("directive(off, asdf;", snapshot);
        assert_parse_err("directive();", snapshot);
    }

    #[test]
    fn severities() {}

    #[test]
    fn invalid_severity() {}

    #[test]
    fn triggering_rules() {}

    #[test]
    fn invalid_triggering_rule() {
        #[derive(Debug, Clone)]
        enum Rule {
            Valid(DiagnosticTriggeringRule),
            Invalid,
        }

        #[derive(Debug, Clone)]
        enum Sev {
            Valid(Severity),
            Invalid,
        }

        let cases = {
            let invalid_sev_cases = DiagnosticTriggeringRule::iter()
                .map(Rule::Valid)
                .cartesian_product([Sev::Invalid]);
            let invalid_rule_cases = [Rule::Invalid]
                .into_iter()
                .cartesian_product(Severity::iter().map(Sev::Valid));
            invalid_sev_cases.chain(invalid_rule_cases)
        };

        for (rule, severity) in cases {
            let rule = match rule {
                Rule::Valid(rule) => rule.to_ident(),
                Rule::Invalid => "totes_invalid_rule",
            };
            let severity = match severity {
                Sev::Valid(severity) => severity.to_ident(),
                Sev::Invalid => "totes_invalid_severity",
            };
            let shader = format!("diagnostic({severity},{rule});");
            let expected_msg = format!(
                "\
"
            );

            assert_parse_err(&shader, &expected_msg);
        }
    }
}
