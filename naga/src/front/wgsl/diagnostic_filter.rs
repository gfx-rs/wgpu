use crate::diagnostic_filter::{FilterableTriggeringRule, Severity};

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

impl FilterableTriggeringRule {
    const DERIVATIVE_UNIFORMITY: &'static str = "derivative_uniformity";

    /// Convert from a sentinel word in WGSL into its associated [`FilterableTriggeringRule`], if possible.
    pub fn from_wgsl_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::DERIVATIVE_UNIFORMITY => Self::DerivativeUniformity,
            _ => return None,
        })
    }

    /// Maps this [`FilterableTriggeringRule`] into the sentinel word associated with it in WGSL.
    pub const fn to_wgsl_ident(self) -> &'static str {
        match self {
            Self::DerivativeUniformity => Self::DERIVATIVE_UNIFORMITY,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::diagnostic_filter::{FilterableTriggeringRule, Severity};
    use crate::front::wgsl::assert_parse_err;

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
            Valid(FilterableTriggeringRule),
            Invalid,
        }

        #[derive(Debug, Clone)]
        enum Sev {
            Valid(Severity),
            Invalid,
        }

        let cases = {
            let invalid_sev_cases = FilterableTriggeringRule::iter()
                .map(Rule::Valid)
                .cartesian_product([Sev::Invalid]);
            let invalid_rule_cases = [Rule::Invalid]
                .into_iter()
                .cartesian_product(Severity::iter().map(Sev::Valid));
            invalid_sev_cases.chain(invalid_rule_cases)
        };

        for (rule, severity) in cases {
            let rule = match rule {
                Rule::Valid(rule) => rule.to_wgsl_ident(),
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
