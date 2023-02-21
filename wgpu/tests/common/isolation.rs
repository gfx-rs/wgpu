use std::sync::atomic::{AtomicBool, Ordering};

/// True if a test is in progress somewhere in the process, false otherwise.
static TEST_ACTIVE_IN_PROCESS: AtomicBool = AtomicBool::new(false);

const OTHER_TEST_IN_PROGRESS_ERROR: &str = "TEST ISOLATION ERROR:

wgpu's test harness requires that no more than one test is running per process.

The best way to facilitate this is by using cargo-nextest which runs each test in its own process
and has a very good testing UI:

cargo install cargo-nextest
cargo nextest run

Alternatively, you can run tests in single threaded mode (much slower).

cargo test -- --test-threads=1

Calling std::process::abort()...
";

/// When this guard is active, enforces that there is only a single test running in the process
/// at any one time. If there are multiple processes, creating the guard hard terminates the process.
pub struct OneTestPerProcessGuard(());

impl OneTestPerProcessGuard {
    pub fn new() -> Self {
        let other_tests_in_flight = TEST_ACTIVE_IN_PROCESS.swap(true, Ordering::SeqCst);

        // We never abort if we're on wasm. Wasm tests are inherently single threaded, and panics cannot
        // unwind the stack and trigger all the guards, so we don't actually need to check.
        if other_tests_in_flight && !cfg!(target_arch = "wasm32") {
            log::error!("{}", OTHER_TEST_IN_PROGRESS_ERROR);
            // Hard exit to call attention to the error
            std::process::abort();
        }
        OneTestPerProcessGuard(())
    }
}

impl Drop for OneTestPerProcessGuard {
    fn drop(&mut self) {
        TEST_ACTIVE_IN_PROCESS.store(false, Ordering::SeqCst);
    }
}
