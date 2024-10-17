use std::str::FromStr;

use crate::{Handle, Span};

#[cfg(feature = "arbitrary")]
use arbitrary::Arbitrary;
use indexmap::IndexMap;
#[cfg(feature = "deserialize")]
use serde::Deserialize;
#[cfg(feature = "serialize")]
use serde::Serialize;

// TODO: docs
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
    pub fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            "error" => Self::Error,
            "warning" => Self::Warning,
            "info" => Self::Info,
            "off" => Self::Off,
            _ => return None,
        })
    }
}

impl FromStr for Severity {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_ident(s).ok_or(())
    }
}

// TODO: docs
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum DiagnosticTriggeringRule {
    DerivativeUniformity,
}

// TODO: docs
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct DiagnosticFilter {
    pub new_severity: Severity,
    pub triggering_rule: DiagnosticTriggeringRule,
}

// TODO: docs
#[derive(Clone, Debug, Default)]
// TODO: `FastIndexMap` mebbe?
pub(crate) struct DiagnosticFilterMap(IndexMap<DiagnosticTriggeringRule, (Severity, Span)>);

#[cfg(feature = "wgsl-in")]
impl DiagnosticFilterMap {
    pub(crate) fn new() -> Self {
        Self::default()
    }

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

// TODO: docs
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
    fn malformed() {}

    #[test]
    fn severities() {}

    #[test]
    fn invalid_severity() {}

    #[test]
    fn triggering_rules() {}

    #[test]
    fn invalid_triggering_rule() {
        let cases = DiagnosticTriggeringRule::iter().cartesian_product(Severity::iter());

        for (rule, severity) in cases {
            let shader = "\
";

            assert_parse_err(
                shader, "\
    ",
            );
        }
    }
}
