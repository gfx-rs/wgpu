use std::{collections::BTreeMap, io::Write};

use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

use super::analyze::{AnalysisOutcome, TestLabel};

// Rendering is intentionally section-based and deterministic so humans can scan it
// quickly and tools can parse it with minimal heuristics.

struct Reporter {
    stream: StandardStream,
}

impl Reporter {
    fn new() -> Self {
        Self {
            stream: StandardStream::stdout(ColorChoice::Auto),
        }
    }

    fn line(&mut self, indent: usize, text: impl AsRef<str>, color: Option<Color>, bold: bool) {
        let mut style = ColorSpec::new();
        style.set_fg(color).set_bold(bold);
        let _ = self.stream.set_color(&style);
        let _ = write!(&mut self.stream, "{}", " ".repeat(indent));
        let _ = write!(&mut self.stream, "{}", text.as_ref());
        let _ = self.stream.reset();
        let _ = writeln!(&mut self.stream);
    }

    fn blank_line(&mut self) {
        let _ = writeln!(&mut self.stream);
    }

    fn general(&mut self, message: impl AsRef<str>) {
        self.line(2, message, None, false);
    }

    fn section(&mut self, message: impl AsRef<str>, color: Color) {
        self.line(2, message, Some(color), true);
    }

    fn test_item(&mut self, message: impl AsRef<str>) {
        self.line(4, message, Some(Color::Cyan), false);
    }

    fn adapter_heading(&mut self, message: impl AsRef<str>) {
        self.line(4, message, Some(Color::Cyan), true);
    }

    fn adapter_test(&mut self, message: impl AsRef<str>) {
        self.line(6, message, Some(Color::Cyan), false);
    }
}

pub(super) fn print_general(message: impl AsRef<str>) {
    let mut reporter = Reporter::new();
    reporter.general(message);
}

pub(super) fn print_section_line(message: impl AsRef<str>, color: Color) {
    let mut reporter = Reporter::new();
    reporter.section(message, color);
}

fn count_phrase(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("{count} {singular}")
    } else {
        format!("{count} {plural}")
    }
}

#[derive(Clone, Copy)]
struct CategoryPrintOptions<'a> {
    show_all: bool,
    show_items_by_default: bool,
    group_by_adapter: bool,
    limit: usize,
    more_label: &'a str,
}

fn render_limited_flat_items(
    reporter: &mut Reporter,
    items: &[String],
    show_all: bool,
    limit: usize,
    more_label: &str,
) {
    if show_all {
        for item in items {
            reporter.test_item(item);
        }
        return;
    }

    let shown = items.len().min(limit);
    for item in items.iter().take(limit) {
        reporter.test_item(item);
    }
    if shown < items.len() {
        reporter.test_item(format!("... and {} {}", items.len() - shown, more_label));
    }
}

fn render_label_items(
    reporter: &mut Reporter,
    items: &[TestLabel],
    show_all: bool,
    group_by_adapter: bool,
    limit: usize,
    more_label: &str,
) {
    if items.is_empty() {
        return;
    }

    if !group_by_adapter {
        let item_names = items
            .iter()
            .map(TestLabel::display_name)
            .collect::<Vec<_>>();
        render_limited_flat_items(reporter, &item_names, show_all, limit, more_label);
        return;
    }

    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ungrouped = Vec::new();
    for item in items {
        let test_name = item.display_name();
        if let Some(adapter) = item.adapter.as_ref() {
            grouped
                .entry(format!("{} ({})", adapter.name, adapter.backend))
                .or_default()
                .push(test_name);
        } else {
            ungrouped.push(test_name);
        }
    }

    // If no adapter metadata exists, keep the simple flat rendering.
    if grouped.is_empty() {
        let item_names = items
            .iter()
            .map(TestLabel::display_name)
            .collect::<Vec<_>>();
        render_limited_flat_items(reporter, &item_names, show_all, limit, more_label);
        return;
    }

    // Use usize::MAX to represent "show all items without limit".
    let mut remaining = if show_all { usize::MAX } else { limit };
    let mut shown = 0usize;
    for (adapter, tests) in grouped {
        if remaining == 0 {
            break;
        }
        reporter.adapter_heading(adapter);
        for test in tests {
            if remaining == 0 {
                break;
            }
            reporter.adapter_test(test);
            remaining -= 1;
            shown += 1;
        }
    }

    if !ungrouped.is_empty() && remaining > 0 {
        reporter.adapter_heading("Other");
        for test in ungrouped {
            if remaining == 0 {
                break;
            }
            reporter.adapter_test(test);
            remaining -= 1;
            shown += 1;
        }
    }

    if !show_all && shown < items.len() {
        reporter.adapter_test(format!("... and {} {}", items.len() - shown, more_label));
    }
}

fn print_label_category(
    reporter: &mut Reporter,
    title: String,
    color: Color,
    items: &[TestLabel],
    options: CategoryPrintOptions<'_>,
) {
    if items.is_empty() && !options.show_all {
        return;
    }

    reporter.section(title, color);
    if items.is_empty() {
        reporter.test_item("(none)");
        return;
    }

    if options.show_all || options.show_items_by_default {
        render_label_items(
            reporter,
            items,
            options.show_all,
            options.group_by_adapter,
            options.limit,
            options.more_label,
        );
    }
}

fn print_string_category(
    reporter: &mut Reporter,
    title: String,
    color: Color,
    items: &[String],
    options: CategoryPrintOptions<'_>,
) {
    if items.is_empty() && !options.show_all {
        return;
    }

    reporter.section(title, color);
    if items.is_empty() {
        reporter.test_item("(none)");
        return;
    }

    if options.show_all || options.show_items_by_default {
        render_limited_flat_items(
            reporter,
            items,
            options.show_all,
            options.limit,
            options.more_label,
        );
    }
}

pub(super) fn print_outcome(outcome: &AnalysisOutcome, show_all_categories: bool) {
    let mut reporter = Reporter::new();
    reporter.blank_line();
    reporter.line(
        0,
        "Expectation reconciliation summary",
        Some(Color::Blue),
        true,
    );

    print_label_category(
        &mut reporter,
        format!(
            "{} passed expectation checks.",
            count_phrase(outcome.passed_tests.len(), "test", "tests")
        ),
        Color::Green,
        &outcome.passed_tests,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: false,
            group_by_adapter: false,
            limit: 40,
            more_label: "more passed tests",
        },
    );
    print_label_category(
        &mut reporter,
        format!(
            "{} failed as expected.",
            count_phrase(outcome.known_failure_tests.len(), "test", "tests")
        ),
        Color::Yellow,
        &outcome.known_failure_tests,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: false,
            group_by_adapter: true,
            limit: 20,
            more_label: "more known-failure tests",
        },
    );
    print_label_category(
        &mut reporter,
        format!(
            "{} were skipped because required features are unsupported.",
            count_phrase(outcome.skipped_unsupported_tests.len(), "test", "tests")
        ),
        Color::Yellow,
        &outcome.skipped_unsupported_tests,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: false,
            group_by_adapter: true,
            limit: 20,
            more_label: "more unsupported-skipped tests",
        },
    );
    print_label_category(
        &mut reporter,
        format!(
            "{} were skipped due to inline expectations.",
            count_phrase(outcome.skipped_expected_tests.len(), "test", "tests")
        ),
        Color::Yellow,
        &outcome.skipped_expected_tests,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: false,
            group_by_adapter: true,
            limit: 20,
            more_label: "more expected-skipped tests",
        },
    );
    print_label_category(
        &mut reporter,
        format!(
            "{} expected to fail, but passed:",
            count_phrase(
                outcome.gpu_expected_to_fail_but_passed.len(),
                "test",
                "tests"
            )
        ),
        Color::Yellow,
        &outcome.gpu_expected_to_fail_but_passed,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: true,
            group_by_adapter: true,
            limit: 40,
            more_label: "more stale expectation cases",
        },
    );
    print_label_category(
        &mut reporter,
        format!(
            "{} failed with a different failure signature:",
            count_phrase(
                outcome.gpu_signature_mismatch_failures.len(),
                "test",
                "tests"
            )
        ),
        Color::Red,
        &outcome.gpu_signature_mismatch_failures,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: true,
            group_by_adapter: true,
            limit: 40,
            more_label: "more signature mismatch cases",
        },
    );
    print_label_category(
        &mut reporter,
        format!(
            "{} failed unexpectedly:",
            count_phrase(outcome.gpu_unexpected_failures.len(), "test", "tests")
        ),
        Color::Red,
        &outcome.gpu_unexpected_failures,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: true,
            group_by_adapter: true,
            limit: 40,
            more_label: "more unexpected failure cases",
        },
    );
    print_label_category(
        &mut reporter,
        format!(
            "{} failed in non-GPU suites:",
            count_phrase(outcome.non_gpu_failures.len(), "test", "tests")
        ),
        Color::Red,
        &outcome.non_gpu_failures,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: true,
            group_by_adapter: false,
            limit: 20,
            more_label: "more non-GPU failures",
        },
    );
    print_label_category(
        &mut reporter,
        format!(
            "{} were fixed vs baseline:",
            count_phrase(outcome.fixed.len(), "test", "tests")
        ),
        Color::Green,
        &outcome.fixed,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: true,
            group_by_adapter: true,
            limit: 10,
            more_label: "more fixed cases",
        },
    );

    let added_tests: &[String] = if outcome.baseline_present {
        &outcome.added_tests
    } else {
        &[]
    };
    let removed_tests: &[String] = if outcome.baseline_present {
        &outcome.removed_tests
    } else {
        &[]
    };
    let changed_tests: &[String] = if outcome.baseline_present {
        &outcome.changed
    } else {
        &[]
    };

    print_string_category(
        &mut reporter,
        format!(
            "{} were added to inventory:",
            count_phrase(added_tests.len(), "test", "tests")
        ),
        Color::Blue,
        added_tests,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: true,
            group_by_adapter: false,
            limit: 10,
            more_label: "more added tests",
        },
    );
    print_string_category(
        &mut reporter,
        format!(
            "{} were removed from inventory:",
            count_phrase(removed_tests.len(), "test", "tests")
        ),
        Color::Blue,
        removed_tests,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: true,
            group_by_adapter: false,
            limit: 10,
            more_label: "more removed tests",
        },
    );
    print_string_category(
        &mut reporter,
        format!(
            "{} additional outcomes changed vs baseline:",
            count_phrase(changed_tests.len(), "test", "tests")
        ),
        Color::Blue,
        changed_tests,
        CategoryPrintOptions {
            show_all: show_all_categories,
            show_items_by_default: false,
            group_by_adapter: false,
            limit: 20,
            more_label: "more changed outcomes",
        },
    );

    if !outcome.baseline_present {
        reporter.general("No baseline loaded yet; run with --save-baseline <name> to capture one.");
    }

    if !outcome.gpu_expected_to_fail_but_passed.is_empty() {
        reporter.section(
            "Hint: update inline test expectations (`expect_fail`, `skip`, or `FailureCase::crash`).",
            Color::Yellow,
        );
    }
    if outcome.baseline_present
        && (!outcome.added_tests.is_empty()
            || !outcome.removed_tests.is_empty()
            || !outcome.changed.is_empty())
    {
        reporter.section(
            "Hint: refresh your local machine baseline with `cargo xtask test --save-baseline <name>`.",
            Color::Yellow,
        );
    }
}
