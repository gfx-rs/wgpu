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

#[cfg(test)]
mod test {
    mod parse_sites_not_yet_supported {
        use crate::front::wgsl::assert_parse_err;

        #[test]
        fn user_rules() {
            let shader = "
fn myfunc() {
    if (true) @diagnostic(off, my.lint) {
        //    ^^^^^^^^^^^^^^^^^^^^^^^^^ not yet supported, should report an error
    }
}
";
            assert_parse_err(shader, "\
error: `@diagnostic(…)` attribute(s) not yet implemented
  ┌─ wgsl:3:15
  │
3 │     if (true) @diagnostic(off, my.lint) {
  │               ^^^^^^^^^^^^^^^^^^^^^^^^^ can't use this on compound statements (yet)
  │
  = note: Let Naga maintainers know that you ran into this at <https://github.com/gfx-rs/wgpu/issues/5320>, so they can prioritize it!

");
        }

        #[test]
        fn unknown_rules() {
            let shader = "
fn myfunc() {
	if (true) @diagnostic(off, wat_is_this) {
		//    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ should emit a warning
	}
}
";
            assert_parse_err(shader, "\
error: `@diagnostic(…)` attribute(s) not yet implemented
  ┌─ wgsl:3:12
  │
3 │     if (true) @diagnostic(off, wat_is_this) {
  │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ can't use this on compound statements (yet)
  │
  = note: Let Naga maintainers know that you ran into this at <https://github.com/gfx-rs/wgpu/issues/5320>, so they can prioritize it!

");
        }
    }

    mod directive_conflict {
        use crate::front::wgsl::assert_parse_err;

        #[test]
        fn user_rules() {
            let shader = "
diagnostic(off, my.lint);
diagnostic(warning, my.lint);
";
            assert_parse_err(shader, "\
error: found conflicting `diagnostic(…)` rule(s)
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^ first rule
3 │ diagnostic(warning, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ second rule
  │
  = note: Multiple `diagnostic(…)` rules with the same rule name conflict unless they are directives and the severity is the same.
  = note: You should delete the rule you don't want.

");
        }

        #[test]
        fn unknown_rules() {
            let shader = "
diagnostic(off, wat_is_this);
diagnostic(warning, wat_is_this);
";
            assert_parse_err(shader, "\
error: found conflicting `diagnostic(…)` rule(s)
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off, wat_is_this);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ first rule
3 │ diagnostic(warning, wat_is_this);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ second rule
  │
  = note: Multiple `diagnostic(…)` rules with the same rule name conflict unless they are directives and the severity is the same.
  = note: You should delete the rule you don't want.

");
        }
    }

    mod attribute_conflict {
        use crate::front::wgsl::assert_parse_err;

        #[test]
        fn user_rules() {
            let shader = "
diagnostic(off, my.lint);
diagnostic(warning, my.lint);
";
            assert_parse_err(shader, "\
error: found conflicting `diagnostic(…)` rule(s)
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^ first rule
3 │ diagnostic(warning, my.lint);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ second rule
  │
  = note: Multiple `diagnostic(…)` rules with the same rule name conflict unless they are directives and the severity is the same.
  = note: You should delete the rule you don't want.

");
        }

        #[test]
        fn unknown_rules() {
            let shader = "
diagnostic(off, wat_is_this);
diagnostic(warning, wat_is_this);
";
            assert_parse_err(shader, "\
error: found conflicting `diagnostic(…)` rule(s)
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off, wat_is_this);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ first rule
3 │ diagnostic(warning, wat_is_this);
  │ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ second rule
  │
  = note: Multiple `diagnostic(…)` rules with the same rule name conflict unless they are directives and the severity is the same.
  = note: You should delete the rule you don't want.

");
        }
    }
}
