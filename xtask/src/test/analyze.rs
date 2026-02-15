use std::collections::{BTreeMap, BTreeSet};

use quick_junit::{Report, TestCase, TestCaseStatus};
use serde::{Deserialize, Serialize};
use wgpu_test_metadata::{AdapterKey, EventPhase, FailureSignature, GpuHarnessEvent, GpuTestKey};

// Reconciliation logic lives here so reporting and orchestration can stay simple.
// This module is the single place that interprets:
// - inline GPU test expectations (from `expect_fail`, `skip`, etc.),
// - optional local baseline state (persisted per-machine outcomes),
// - and observed JUnit/harness outcomes (actual test results).
//
// The reconciliation performs a three-way merge:
// 1. Inline expectations define what the test author expects
// 2. Baseline captures what actually happens on this specific machine
// 3. Observed outcomes show what just happened in this run

// === Data model used for harness <-> xtask reconciliation ===
//
// `GpuHarnessEvent` is emitted from `tests/src/run.rs` and parsed out of JUnit.
// `ExpectationBaseline` is the optional per-machine baseline persisted by xtask.

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub(super) struct ExpectationBaseline {
    pub(super) version: u32,
    pub(super) inventory: Vec<String>,
    pub(super) gpu_cases: Vec<BaselineGpuCase>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct BaselineGpuCase {
    pub(super) key: GpuTestKey,
    pub(super) suite_name: String,
    pub(super) case_name: String,
    pub(super) actual_success: Option<bool>,
    pub(super) failure_signatures: Vec<FailureSignature>,
}

pub(super) struct AnalysisOutcome {
    pub(super) passed_tests: Vec<TestLabel>,
    pub(super) known_failure_tests: Vec<TestLabel>,
    pub(super) skipped_unsupported_tests: Vec<TestLabel>,
    pub(super) skipped_expected_tests: Vec<TestLabel>,
    pub(super) non_gpu_failures: Vec<TestLabel>,
    pub(super) gpu_expected_to_fail_but_passed: Vec<TestLabel>,
    pub(super) gpu_signature_mismatch_failures: Vec<TestLabel>,
    pub(super) gpu_unexpected_failures: Vec<TestLabel>,
    pub(super) changed: Vec<String>,
    pub(super) fixed: Vec<TestLabel>,
    pub(super) added_tests: Vec<String>,
    pub(super) removed_tests: Vec<String>,
    pub(super) baseline_present: bool,
    pub(super) current_baseline: ExpectationBaseline,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct TestLabel {
    pub(super) suite_name: String,
    pub(super) test_name: String,
    pub(super) adapter: Option<AdapterKey>,
}

impl TestLabel {
    pub(super) fn display_name(&self) -> String {
        format!("{}::{}", self.suite_name, self.test_name)
    }
}

impl AnalysisOutcome {
    pub(super) fn success(&self) -> bool {
        self.non_gpu_failures.is_empty()
            && self.gpu_expected_to_fail_but_passed.is_empty()
            && self.gpu_signature_mismatch_failures.is_empty()
            && self.gpu_unexpected_failures.is_empty()
    }
}

pub(super) fn analyze_report(
    report: Report,
    baseline: &ExpectationBaseline,
    inventory: BTreeSet<String>,
) -> AnalysisOutcome {
    // Baseline lookup is keyed by typed adapter+test identity so adapter ordering
    // changes between runs do not affect matching.
    let baseline_present = !baseline.inventory.is_empty() || !baseline.gpu_cases.is_empty();
    let mut baseline_map: BTreeMap<GpuTestKey, BaselineGpuCase> = BTreeMap::new();
    for case in &baseline.gpu_cases {
        baseline_map.insert(case.key.clone(), case.clone());
    }

    let mut non_gpu_failures = Vec::new();
    let mut gpu_expected_to_fail_but_passed = Vec::new();
    let mut gpu_signature_mismatch_failures = Vec::new();
    let mut gpu_unexpected_failures = Vec::new();
    let mut changed = Vec::new();
    let mut fixed = Vec::new();
    let mut passed_tests = Vec::new();
    let mut known_failure_tests = Vec::new();
    let mut skipped_unsupported_tests = Vec::new();
    let mut skipped_expected_tests = Vec::new();
    let mut current_gpu_cases = Vec::new();

    for suite in report.test_suites {
        let suite_name = suite.name.as_str().to_owned();
        for test_case in suite.test_cases {
            // We evaluate every junit testcase in three phases:
            // 1) extract harness events (if this is GPU/custom-harness),
            // 2) resolve baseline identity,
            // 3) reconcile baseline × inline expectation × actual.
            let case_name = test_case.name.as_str().to_owned();
            let junit_success = junit_success(&test_case.status);
            let junit_failure_signatures = junit_failure_signatures(&test_case.status);
            let (before, after, observed_failure_signatures) = extract_gpu_events(&test_case);
            let is_gpu_managed = before.is_some() || after.is_some() || is_gpu_suite(&suite_name);
            let case_label = make_case_label(&suite_name, &case_name, before.as_ref());

            if !is_gpu_managed {
                if junit_success == Some(false) {
                    non_gpu_failures.push(case_label);
                } else {
                    passed_tests.push(case_label);
                }
                continue;
            }

            let key = before.as_ref().map(|event| event.key.clone());
            let baseline_case = key.as_ref().and_then(|key| baseline_map.get(key));
            let evaluation = evaluate_gpu_case(
                &suite_name,
                &case_name,
                JunitObservation {
                    success: junit_success,
                    failure_signatures: &junit_failure_signatures,
                    observed_failure_signatures: &observed_failure_signatures,
                },
                before.as_ref(),
                after.as_ref(),
                baseline_case,
            );

            if let Some(failure_category) = evaluation.failure {
                match failure_category {
                    GpuFailureCategory::ExpectedToFailButPassed => {
                        gpu_expected_to_fail_but_passed.push(case_label.clone());
                    }
                    GpuFailureCategory::SignatureMismatch => {
                        gpu_signature_mismatch_failures.push(case_label.clone());
                    }
                    GpuFailureCategory::UnexpectedFailure => {
                        gpu_unexpected_failures.push(case_label.clone());
                    }
                }
            } else {
                passed_tests.push(case_label.clone());
                if evaluation.known_failure {
                    known_failure_tests.push(case_label.clone());
                }
            }

            if evaluation.skipped_unsupported {
                skipped_unsupported_tests.push(case_label.clone());
            }
            if evaluation.skipped_expected {
                skipped_expected_tests.push(case_label.clone());
            }

            if let Some(change) = evaluation.change {
                changed.push(change);
            }
            if evaluation.fixed == Some(true) {
                fixed.push(case_label.clone());
            }

            if let Some(key) = key {
                current_gpu_cases.push(BaselineGpuCase {
                    key,
                    suite_name: suite_name.clone(),
                    case_name: case_name.clone(),
                    actual_success: evaluation.current_actual_success,
                    failure_signatures: evaluation.current_failure_signatures,
                });
            }
        }
    }

    let current_inventory = inventory.iter().cloned().collect::<Vec<_>>();
    let current_inventory_set = inventory;
    let baseline_inventory_set = baseline
        .inventory
        .iter()
        .cloned()
        .collect::<BTreeSet<String>>();
    let added_tests = current_inventory_set
        .difference(&baseline_inventory_set)
        .cloned()
        .collect::<Vec<_>>();
    let removed_tests = baseline_inventory_set
        .difference(&current_inventory_set)
        .cloned()
        .collect::<Vec<_>>();

    AnalysisOutcome {
        passed_tests,
        known_failure_tests,
        skipped_unsupported_tests,
        skipped_expected_tests,
        non_gpu_failures,
        gpu_expected_to_fail_but_passed,
        gpu_signature_mismatch_failures,
        gpu_unexpected_failures,
        changed,
        fixed,
        added_tests,
        removed_tests,
        baseline_present,
        current_baseline: ExpectationBaseline {
            version: 1,
            inventory: current_inventory,
            gpu_cases: current_gpu_cases,
        },
    }
}

struct GpuCaseEvaluation {
    failure: Option<GpuFailureCategory>,
    change: Option<String>,
    fixed: Option<bool>,
    known_failure: bool,
    skipped_unsupported: bool,
    skipped_expected: bool,
    current_actual_success: Option<bool>,
    current_failure_signatures: Vec<FailureSignature>,
}

#[derive(Clone, Copy, Debug)]
enum GpuFailureCategory {
    ExpectedToFailButPassed,
    SignatureMismatch,
    UnexpectedFailure,
}

struct JunitObservation<'a> {
    /// Final JUnit verdict for the testcase (`None` when JUnit reports skipped).
    success: Option<bool>,
    /// Failure signatures from final JUnit status plus JUnit reruns/flaky runs.
    failure_signatures: &'a [FailureSignature],
    /// Failure signatures seen in harness `After` events from all observed reruns.
    observed_failure_signatures: &'a [FailureSignature],
}

/// Reconcile one testcase using the richest identity available.
///
/// If harness events exist, use adapter-aware event keys. Otherwise fall back to
/// plain JUnit status for non-custom-harness style tests.
fn evaluate_gpu_case(
    suite_name: &str,
    case_name: &str,
    junit: JunitObservation<'_>,
    before: Option<&GpuHarnessEvent>,
    after: Option<&GpuHarnessEvent>,
    baseline: Option<&BaselineGpuCase>,
) -> GpuCaseEvaluation {
    let label = make_case_label(suite_name, case_name, before);
    let label_text = label.display_name();
    let label_with_adapter = if let Some(adapter) = label.adapter.as_ref() {
        format!("{} [{} / {}]", label_text, adapter.backend, adapter.name)
    } else {
        label_text
    };
    let baseline_actual = baseline.and_then(|case| case.actual_success);

    if let Some(before) = before {
        return evaluate_gpu_case_with_harness(
            before,
            after,
            baseline,
            baseline_actual,
            junit,
            &label_with_adapter,
        );
    }

    evaluate_gpu_case_without_harness(
        baseline_actual,
        junit.success,
        junit.failure_signatures,
        &label_with_adapter,
    )
}

fn make_case_label(
    suite_name: &str,
    case_name: &str,
    before: Option<&GpuHarnessEvent>,
) -> TestLabel {
    // Prefer event key test path when available: this keeps names stable and avoids
    // relying on JUnit case-name formatting from custom harnesses.
    match before {
        Some(event) => TestLabel {
            suite_name: suite_name.to_owned(),
            test_name: event.key.test_path.clone(),
            adapter: Some(event.key.adapter.clone()),
        },
        None => TestLabel {
            suite_name: suite_name.to_owned(),
            test_name: case_name.to_owned(),
            adapter: None,
        },
    }
}

fn evaluate_gpu_case_with_harness(
    before: &GpuHarnessEvent,
    after: Option<&GpuHarnessEvent>,
    baseline: Option<&BaselineGpuCase>,
    baseline_actual: Option<bool>,
    junit: JunitObservation<'_>,
    label: &str,
) -> GpuCaseEvaluation {
    // A skip with no after event is suspicious: usually an event parsing/matching problem.
    if before.skip && after.is_none() && junit.success == Some(true) {
        return GpuCaseEvaluation {
            failure: Some(GpuFailureCategory::UnexpectedFailure),
            change: None,
            fixed: None,
            known_failure: false,
            skipped_unsupported: before.skip_due_to_unsupported,
            skipped_expected: before.skip_due_to_expectation || before.inline_expect_fail,
            current_actual_success: Some(true),
            current_failure_signatures: junit.failure_signatures.to_vec(),
        };
    }

    // JUnit is authoritative for process-level success/failure (including aborts).
    // Harness output can say "success" and still lose due to a late process crash.
    let actual_success = match junit.success {
        Some(true) => Some(true),
        Some(false) => Some(false),
        None => after.and_then(|event| event.actual_success),
    };

    // Merge failure evidence from every available channel:
    // - final harness `After` event
    // - earlier rerun `After` events
    // - JUnit final/rerun/flaky statuses
    let mut actual_failures = Vec::new();
    if let Some(after) = after {
        for signature in &after.actual_failure_signatures {
            if !actual_failures.contains(signature) {
                actual_failures.push(signature.clone());
            }
        }
    }
    for signature in junit.observed_failure_signatures {
        if !actual_failures.contains(signature) {
            actual_failures.push(signature.clone());
        }
    }
    for signature in junit.failure_signatures {
        if !actual_failures.contains(signature) {
            actual_failures.push(signature.clone());
        }
    }

    // Skipped test: no actual pass/fail bit from harness.
    if actual_success.is_none() {
        return GpuCaseEvaluation {
            failure: None,
            // Mark as changed if baseline had any concrete outcome but now it's skipped.
            change: baseline_actual
                .is_some()
                .then(|| format!("changed (skipped): {label}")),
            fixed: None,
            known_failure: false,
            skipped_unsupported: before.skip_due_to_unsupported,
            skipped_expected: before.skip_due_to_expectation || before.inline_expect_fail,
            current_actual_success: None,
            current_failure_signatures: Vec::new(),
        };
    }

    // From this point onward we only handle concrete pass/fail outcomes.
    let actual_success = actual_success.unwrap_or(false);
    let expectation_success = !before.inline_expect_fail;
    // Failures are permitted if inline expectations allow them or if the machine
    // baseline says this test is a known local failure.
    let allow_failure = !expectation_success || baseline_actual == Some(false);
    let signature_match =
        failure_signatures_match(before, baseline, expectation_success, &actual_failures);
    // A final pass with failed flaky reruns should still count as an observed expected
    // failure; otherwise flaky expected-fail tests become stale-success false positives.
    let has_flaky_retry_failures =
        junit.success == Some(true) && !junit.failure_signatures.is_empty();
    let observed_expected_failure = actual_success
        && allow_failure
        && !actual_failures.is_empty()
        && (signature_match || has_flaky_retry_failures);
    // Treat "pass after expected flaky failure" as effective failure for reconciliation.
    // This prevents flaky expected-fail tests from being reported as stale expectations
    // that should be removed, since they did fail on at least one retry.
    let effective_actual_success = if observed_expected_failure {
        false
    } else {
        actual_success
    };

    // Matrix interpretation:
    // - Final success passes if inline expectations currently say this test should pass.
    // - Final failure passes only when either inline expectation or baseline permits failure,
    //   and the failure signature matches (or this is a flaky retry failure).
    // Final matrix:
    // - effective success passes only when inline expectations currently expect success.
    // - effective failure passes only when failure is allowed and the observed signature
    //   is consistent with either inline or baseline known signatures.
    let pass = if effective_actual_success {
        expectation_success
    } else {
        allow_failure && (signature_match || observed_expected_failure)
    };

    let known_failure = pass && allow_failure && !effective_actual_success;
    let current_actual = Some(effective_actual_success);
    let changed = baseline_actual != current_actual;
    let failure = if pass {
        None
    } else if effective_actual_success && !expectation_success {
        Some(GpuFailureCategory::ExpectedToFailButPassed)
    } else if !effective_actual_success && !signature_match && !observed_expected_failure {
        Some(GpuFailureCategory::SignatureMismatch)
    } else {
        Some(GpuFailureCategory::UnexpectedFailure)
    };

    GpuCaseEvaluation {
        failure,
        change: changed
            .then(|| format!("changed: {label} ({baseline_actual:?} -> {current_actual:?})")),
        fixed: (baseline_actual == Some(false) && current_actual == Some(true)).then_some(true),
        known_failure,
        skipped_unsupported: false,
        skipped_expected: false,
        current_actual_success: current_actual,
        current_failure_signatures: actual_failures,
    }
}

fn evaluate_gpu_case_without_harness(
    baseline_actual: Option<bool>,
    junit_success: Option<bool>,
    junit_failure_signatures: &[FailureSignature],
    label: &str,
) -> GpuCaseEvaluation {
    // Fallback for suites with no harness events: we can only trust JUnit status and
    // baseline pass/fail transitions, without signature-aware expectation matching.
    let current_actual = junit_success;
    let changed = baseline_actual != current_actual;
    let failure_signatures = junit_failure_signatures.to_vec();
    let failure = if junit_success == Some(false) {
        Some(GpuFailureCategory::UnexpectedFailure)
    } else {
        None
    };

    GpuCaseEvaluation {
        failure,
        change: changed.then(|| {
            format!("changed (fallback): {label} ({baseline_actual:?} -> {current_actual:?})")
        }),
        fixed: (baseline_actual == Some(false) && current_actual == Some(true)).then_some(true),
        known_failure: false,
        skipped_unsupported: false,
        skipped_expected: false,
        current_actual_success: current_actual,
        current_failure_signatures: failure_signatures,
    }
}

/// Check whether observed failure signatures match expected patterns.
///
/// Patterns can come from two sources:
/// - Inline expectations: if the test has `expect_fail` with specific failure reasons,
///   those define the expected signatures for this run.
/// - Baseline: if this machine's baseline shows this test failed with specific signatures,
///   those are also accepted as known patterns (even if inline expectations changed).
///
/// Returns `true` if any pattern matches any observed failure.
fn failure_signatures_match(
    before: &GpuHarnessEvent,
    baseline: Option<&BaselineGpuCase>,
    expectation_success: bool,
    actual_failures: &[FailureSignature],
) -> bool {
    // For explicit crash expectations, we cannot rely on in-process signature text.
    // Any observed failure is acceptable as long as process failure occurred.
    if before.inline_expect_crash {
        return true;
    }

    // A failure can be accepted by either:
    // - the inline expectation signatures currently attached to the test, or
    // - baseline signatures recorded for this machine when baseline says it failed.
    let mut patterns = Vec::new();
    if !expectation_success {
        patterns.extend(before.expected_failure_signatures.clone());
    }
    if let Some(baseline) = baseline {
        if baseline.actual_success == Some(false) {
            patterns.extend(baseline.failure_signatures.clone());
        }
    }

    signatures_match(&patterns, actual_failures)
}

fn signatures_match(patterns: &[FailureSignature], actual_failures: &[FailureSignature]) -> bool {
    // Any expected pattern matching any observed failure is sufficient.
    for pattern in patterns {
        for actual in actual_failures {
            if signature_matches(pattern, actual) {
                return true;
            }
        }
    }
    false
}

fn signature_matches(pattern: &FailureSignature, actual: &FailureSignature) -> bool {
    // Kind must match exactly when specified; message is case-insensitive substring.
    let kind_matches = pattern.kind.as_ref().is_none_or(|pattern_kind| {
        actual
            .kind
            .as_ref()
            .is_some_and(|actual_kind| actual_kind == pattern_kind)
    });
    let message_matches = pattern.message.as_ref().is_none_or(|pattern_message| {
        actual.message.as_ref().is_some_and(|actual_message| {
            actual_message
                .to_ascii_lowercase()
                .contains(&pattern_message.to_ascii_lowercase())
        })
    });
    kind_matches && message_matches
}

fn extract_gpu_events(
    test_case: &TestCase,
) -> (
    Option<GpuHarnessEvent>,
    Option<GpuHarnessEvent>,
    Vec<FailureSignature>,
) {
    // Trust typed event identity from payload instead of parsing JUnit case names.
    // Once the first key is selected, ignore events for other keys in the same output.
    let mut before = None;
    let mut after = None;
    let mut observed_failure_signatures = Vec::new();
    let mut selected_key: Option<GpuTestKey> = None;
    for chunk in test_output_chunks(test_case) {
        for line in chunk.lines() {
            let Some(start) = line.find(wgpu_test_metadata::GPU_TEST_EVENT_PREFIX) else {
                continue;
            };
            let json = &line[start + wgpu_test_metadata::GPU_TEST_EVENT_PREFIX.len()..];
            let Ok(event) = serde_json::from_str::<GpuHarnessEvent>(json) else {
                continue;
            };
            if let Some(selected_key) = selected_key.as_ref() {
                if selected_key != &event.key {
                    continue;
                }
            } else {
                selected_key = Some(event.key.clone());
            }
            match event.phase {
                EventPhase::Before => before = Some(event),
                EventPhase::After => {
                    if event.actual_success == Some(false) {
                        for signature in &event.actual_failure_signatures {
                            if !observed_failure_signatures.contains(signature) {
                                observed_failure_signatures.push(signature.clone());
                            }
                        }
                    }
                    after = Some(event);
                }
            }
        }
    }
    (before, after, observed_failure_signatures)
}

fn test_output_chunks(test_case: &TestCase) -> Vec<&str> {
    // Include base output and rerun/flaky output so reconciliation sees all signals.
    let mut chunks = Vec::new();
    if let Some(system_out) = test_case.system_out.as_ref() {
        chunks.push(system_out.as_str());
    }
    if let Some(system_err) = test_case.system_err.as_ref() {
        chunks.push(system_err.as_str());
    }
    match &test_case.status {
        TestCaseStatus::NonSuccess { reruns, .. } => {
            for rerun in reruns {
                if let Some(system_out) = rerun.system_out.as_ref() {
                    chunks.push(system_out.as_str());
                }
                if let Some(system_err) = rerun.system_err.as_ref() {
                    chunks.push(system_err.as_str());
                }
            }
        }
        TestCaseStatus::Success { flaky_runs } => {
            for rerun in flaky_runs {
                if let Some(system_out) = rerun.system_out.as_ref() {
                    chunks.push(system_out.as_str());
                }
                if let Some(system_err) = rerun.system_err.as_ref() {
                    chunks.push(system_err.as_str());
                }
            }
        }
        TestCaseStatus::Skipped { .. } => {}
    }
    chunks
}

fn junit_success(status: &TestCaseStatus) -> Option<bool> {
    match status {
        TestCaseStatus::Success { .. } => Some(true),
        TestCaseStatus::NonSuccess { .. } => Some(false),
        TestCaseStatus::Skipped { .. } => None,
    }
}

fn junit_failure_signatures(status: &TestCaseStatus) -> Vec<FailureSignature> {
    // Normalize JUnit failure/failure-rerun/flaky-rerun status into the same
    // signature shape used by harness and baseline matching.
    fn non_success_signature(
        kind: quick_junit::NonSuccessKind,
        message: Option<&str>,
    ) -> FailureSignature {
        FailureSignature {
            kind: Some(match kind {
                quick_junit::NonSuccessKind::Error => "error".to_owned(),
                quick_junit::NonSuccessKind::Failure => "failure".to_owned(),
            }),
            message: message.map(ToOwned::to_owned),
        }
    }

    match status {
        TestCaseStatus::Success { flaky_runs } => {
            let mut signatures = Vec::new();
            for rerun in flaky_runs {
                signatures.push(non_success_signature(
                    rerun.kind,
                    rerun
                        .message
                        .as_ref()
                        .map(|value| value.as_str())
                        .or_else(|| rerun.description.as_ref().map(|value| value.as_str())),
                ));
            }
            signatures
        }
        TestCaseStatus::Skipped { .. } => Vec::new(),
        TestCaseStatus::NonSuccess {
            kind,
            message,
            description,
            reruns,
            ..
        } => {
            let mut signatures = vec![non_success_signature(
                *kind,
                message
                    .as_ref()
                    .map(|value| value.as_str())
                    .or_else(|| description.as_ref().map(|value| value.as_str())),
            )];
            signatures.extend(reruns.iter().map(|rerun| {
                non_success_signature(
                    rerun.kind,
                    rerun
                        .message
                        .as_ref()
                        .map(|value| value.as_str())
                        .or_else(|| rerun.description.as_ref().map(|value| value.as_str())),
                )
            }));
            signatures
        }
    }
}

fn is_gpu_suite(suite_name: &str) -> bool {
    // Fallback classification for GPU suites that don't emit harness events.
    suite_name.contains("wgpu-gpu") || suite_name.contains("wgpu-examples")
}

#[cfg(test)]
mod tests {
    use wgpu_test_metadata::AdapterKey;

    use super::*;

    fn adapter() -> AdapterKey {
        AdapterKey {
            backend: "Dx12".to_owned(),
            vendor: 0x10de,
            device: 0x1234,
            name: "Adapter".to_owned(),
            driver: "Driver".to_owned(),
        }
    }

    fn key(test_path: &str) -> GpuTestKey {
        GpuTestKey {
            test_path: test_path.to_owned(),
            adapter: adapter(),
        }
    }

    fn event_before(test_path: &str) -> GpuHarnessEvent {
        GpuHarnessEvent {
            version: 1,
            phase: EventPhase::Before,
            key: key(test_path),
            inline_expect_fail: false,
            inline_expect_crash: false,
            skip: false,
            skip_due_to_unsupported: false,
            skip_due_to_expectation: false,
            expected_failure_signatures: Vec::new(),
            actual_success: None,
            actual_failure_signatures: Vec::new(),
            expectation_verdict: None,
        }
    }

    fn event_after(test_path: &str, actual_success: Option<bool>) -> GpuHarnessEvent {
        GpuHarnessEvent {
            version: 1,
            phase: EventPhase::After,
            key: key(test_path),
            inline_expect_fail: false,
            inline_expect_crash: false,
            skip: false,
            skip_due_to_unsupported: false,
            skip_due_to_expectation: false,
            expected_failure_signatures: Vec::new(),
            actual_success,
            actual_failure_signatures: Vec::new(),
            expectation_verdict: Some("complete".to_owned()),
        }
    }

    fn baseline_case(test_path: &str, actual_success: Option<bool>) -> BaselineGpuCase {
        BaselineGpuCase {
            key: key(test_path),
            suite_name: "s".to_owned(),
            case_name: "c".to_owned(),
            actual_success,
            failure_signatures: Vec::new(),
        }
    }

    #[test]
    fn signature_matching_is_case_insensitive_substring() {
        let pattern = FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("Some String".to_owned()),
        };
        let actual = FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("contains sOmE sTrInG text".to_owned()),
        };
        assert!(signature_matches(&pattern, &actual));
    }

    #[test]
    fn baseline_local_failure_allows_expected_pass() {
        let before = event_before("t");
        let mut after = event_after("t", Some(false));
        after.actual_failure_signatures.push(FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("known failure".to_owned()),
        });
        let mut baseline = baseline_case("t", Some(false));
        baseline.failure_signatures.push(FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("known".to_owned()),
        });

        let evaluation = evaluate_gpu_case(
            "suite",
            "case",
            JunitObservation {
                success: Some(false),
                failure_signatures: &after.actual_failure_signatures,
                observed_failure_signatures: &Vec::new(),
            },
            Some(&before),
            Some(&after),
            Some(&baseline),
        );
        assert!(evaluation.failure.is_none());
        assert!(evaluation.known_failure);
    }

    #[test]
    fn stale_expectation_fails() {
        let mut before = event_before("t");
        before.inline_expect_fail = true;
        before.expected_failure_signatures.push(FailureSignature {
            kind: None,
            message: None,
        });
        let mut after = event_after("t", Some(true));
        after.inline_expect_fail = true;
        after.expectation_verdict = Some("panic".to_owned());

        let evaluation = evaluate_gpu_case(
            "suite",
            "case",
            JunitObservation {
                success: Some(true),
                failure_signatures: &Vec::new(),
                observed_failure_signatures: &Vec::new(),
            },
            Some(&before),
            Some(&after),
            None,
        );
        assert!(evaluation.failure.is_some());
    }

    #[test]
    fn flaky_rerun_failure_avoids_stale_expectation() {
        let mut before = event_before("t");
        before.inline_expect_fail = true;
        before.expected_failure_signatures.push(FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("expected".to_owned()),
        });
        let mut after = event_after("t", Some(true));
        after.inline_expect_fail = true;

        let evaluation = evaluate_gpu_case(
            "suite",
            "case",
            JunitObservation {
                success: Some(true),
                failure_signatures: &[FailureSignature {
                    kind: Some("failure".to_owned()),
                    message: Some("retry failed".to_owned()),
                }],
                observed_failure_signatures: &Vec::new(),
            },
            Some(&before),
            Some(&after),
            None,
        );
        assert!(evaluation.failure.is_none());
        assert!(evaluation.known_failure);
    }

    #[test]
    fn unexpected_failure_without_baseline_fails() {
        let before = event_before("t");
        let mut after = event_after("t", Some(false));
        after.actual_failure_signatures.push(FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("boom".to_owned()),
        });
        after.expectation_verdict = Some("panic".to_owned());

        let evaluation = evaluate_gpu_case(
            "suite",
            "case",
            JunitObservation {
                success: Some(false),
                failure_signatures: &after.actual_failure_signatures,
                observed_failure_signatures: &Vec::new(),
            },
            Some(&before),
            Some(&after),
            None,
        );
        assert!(evaluation.failure.is_some());
        assert!(!evaluation.known_failure);
    }

    #[test]
    fn local_baseline_requires_signature_match() {
        let before = event_before("t");
        let mut after = event_after("t", Some(false));
        after.actual_failure_signatures.push(FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("new message".to_owned()),
        });
        after.expectation_verdict = Some("panic".to_owned());
        let mut baseline = baseline_case("t", Some(false));
        baseline.failure_signatures.push(FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("different".to_owned()),
        });

        let evaluation = evaluate_gpu_case(
            "suite",
            "case",
            JunitObservation {
                success: Some(false),
                failure_signatures: &after.actual_failure_signatures,
                observed_failure_signatures: &Vec::new(),
            },
            Some(&before),
            Some(&after),
            Some(&baseline),
        );
        assert!(evaluation.failure.is_some());
    }

    #[test]
    fn baseline_known_failure_and_actual_success_is_fixed() {
        let before = event_before("t");
        let after = event_after("t", Some(true));
        let mut baseline = baseline_case("t", Some(false));
        baseline.failure_signatures.push(FailureSignature {
            kind: Some("panic".to_owned()),
            message: Some("known".to_owned()),
        });

        let evaluation = evaluate_gpu_case(
            "suite",
            "case",
            JunitObservation {
                success: Some(true),
                failure_signatures: &Vec::new(),
                observed_failure_signatures: &Vec::new(),
            },
            Some(&before),
            Some(&after),
            Some(&baseline),
        );
        assert!(evaluation.failure.is_none());
        assert!(!evaluation.known_failure);
        assert_eq!(evaluation.fixed, Some(true));
    }

    #[test]
    fn missing_after_event_for_skipped_case_is_flagged() {
        let mut before = event_before("t");
        before.inline_expect_fail = true;
        before.skip = true;
        before.skip_due_to_expectation = true;

        let evaluation = evaluate_gpu_case(
            "suite",
            "case",
            JunitObservation {
                success: Some(true),
                failure_signatures: &Vec::new(),
                observed_failure_signatures: &Vec::new(),
            },
            Some(&before),
            None,
            None,
        );
        assert!(evaluation.failure.is_some());
    }

    #[test]
    fn junit_failure_overrides_harness_success_event() {
        let before = event_before("t");
        let after = event_after("t", Some(true));
        let junit_failure = FailureSignature {
            kind: Some("test abort".to_owned()),
            message: Some("stack-based buffer overrun".to_owned()),
        };

        let evaluation = evaluate_gpu_case(
            "suite",
            "case",
            JunitObservation {
                success: Some(false),
                failure_signatures: &[junit_failure],
                observed_failure_signatures: &Vec::new(),
            },
            Some(&before),
            Some(&after),
            None,
        );
        assert!(evaluation.failure.is_some());
        assert_eq!(evaluation.current_actual_success, Some(false));
    }
}
