//! Tests for reading `trace.ron` files with `player::parse_trace`.

#![cfg(not(target_arch = "wasm32"))]

extern crate wgpu_core as wgc;

use wgc::device::trace::{Action, DiskTrace, Trace, FILE_NAME};

const CREATE_BUFFER: &str = r#"CreateBuffer(PointerId(0x10),(label:Some("dummy"),size:16,usage:"MAP_READ | COPY_DST | VERTEX",mapped_at_creation:false))"#;
const SUBMIT: &str = "Submit(1,[])";

fn assert_create_then_submit(actions: &[Action<'_, wgc::command::PointerReferences>]) {
    assert_eq!(actions.len(), 2);
    assert!(matches!(actions[0], Action::CreateBuffer(..)));
    assert!(matches!(actions[1], Action::Submit(..)));
}

#[test]
fn one_action_per_line() {
    let content = format!("{CREATE_BUFFER}\n{SUBMIT}\n");
    assert_create_then_submit(&player::parse_trace(&content));
}

#[test]
fn incomplete_last_action_is_skipped() {
    // A trace cut short while the last action was being written.
    let partial = &CREATE_BUFFER[..CREATE_BUFFER.len() / 2];
    let content = format!("{CREATE_BUFFER}\n{SUBMIT}\n{partial}");
    assert_create_then_submit(&player::parse_trace(&content));
}

#[test]
fn list_format() {
    let content = format!("[\n{CREATE_BUFFER},\n{SUBMIT},\n]");
    assert_create_then_submit(&player::parse_trace(&content));
}

#[test]
fn list_format_without_closing_bracket() {
    let content = format!("[\n{CREATE_BUFFER},\n{SUBMIT},\n");
    assert_create_then_submit(&player::parse_trace(&content));
}

#[test]
fn disk_trace_round_trip() {
    let dir = std::env::temp_dir().join(format!("wgpu-trace-file-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let source = format!("{CREATE_BUFFER}\n{SUBMIT}\n");
    let mut trace = DiskTrace::new(dir.clone()).unwrap();
    for action in player::parse_trace(&source) {
        trace.add(action);
    }
    // Read the file while the trace is still alive, as if the program had
    // crashed: no closing `]` is needed.
    let written = std::fs::read_to_string(dir.join(FILE_NAME)).unwrap();
    drop(trace);
    std::fs::remove_dir_all(&dir).unwrap();

    assert_eq!(written.lines().count(), 2);
    assert_create_then_submit(&player::parse_trace(&written));
}
