use crate::TestingContext;

impl TestingContext {
    /// Utility to allow wait polling on devices that don't support wait polling.
    ///
    /// # Notes
    ///
    /// - On WebGL and WebGPU, this will ignore the submission index argument and wait
    ///   for all submitted work to complete.
    /// - On WebGL uses a incremental backoff with setInterval to do repeated polling.
    #[cfg(target_arch = "wasm32")]
    pub async fn async_poll(&self, poll_info: wgpu::PollInfo) -> wgpu::SubmissionStatus {
        use std::time::Duration;
        use web_time::Instant;

        let timeout = match poll_info.effective_wait_duration() {
            Some(timeout) => timeout,
            None => return self.device.poll(poll_info),
        };

        let (sender, receiver) = flume::unbounded();

        let sender_clone = sender.clone();
        self.queue.on_submitted_work_done(move || {
            sender_clone
                .send(wgpu::SubmissionStatus::QueueEmpty)
                .unwrap()
        });

        if self.adapter_info.backend == wgpu::Backend::BrowserWebGpu {
            return receiver.recv_async().await.unwrap();
        }

        let start = Instant::now();
        // We use exponential backoff.
        let mut step_duration = Duration::from_millis(5);
        loop {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return wgpu::SubmissionStatus::Incomplete;
            };

            // Either the step direction or the remaining time, whichever is smaller
            let remaining_duration = timeout - elapsed;
            let sleep_duration = step_duration.min(remaining_duration);

            log::trace!(
                "Waiting for {sleep_duration:?}. Remaining in wait: {remaining_duration:?}."
            );
            gloo_timers::future::sleep(sleep_duration).await;

            // This wait is silly as it will always timeout. This is however required to make firefox make forward progress.
            let _ = self.device.poll(wgpu::PollInfo::poll());

            if let Ok(result) = receiver.try_recv() {
                return result;
            }

            step_duration *= 2;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn async_poll(&self, maintain: wgpu::PollInfo) -> wgpu::SubmissionStatus {
        self.device.poll(maintain)
    }
}
