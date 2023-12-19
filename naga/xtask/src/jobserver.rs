//! Running jobs in parallel, with a controlled degree of concurrency.

use std::sync::OnceLock;

use jobserver::Client;

static JOB_SERVER: OnceLock<Client> = OnceLock::new();

pub fn init() {
    JOB_SERVER.get_or_init(|| {
        // Try to connect to a jobserver inherited from our parent.
        if let Some(client) = unsafe { Client::from_env() } {
            log::debug!("connected to inherited jobserver client");
            client
        } else {
            // Otherwise, start our own jobserver.
            log::debug!("no inherited jobserver client; creating a new jobserver");
            Client::new(num_cpus::get()).expect("failed to create jobserver")
        }
    });
}

/// Wait until it is okay to start a new job, and then spawn a thread running `body`.
pub fn start_job_thread<F>(body: F) -> anyhow::Result<()>
where
    F: FnOnce() + Send + 'static,
{
    let acquired = JOB_SERVER.get().unwrap().acquire()?;
    std::thread::spawn(move || {
        body();
        drop(acquired);
    });
    Ok(())
}
