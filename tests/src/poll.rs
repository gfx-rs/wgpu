use crate::TestingContext;

impl TestingContext {
    /// Utility to allow future asynchronous polling.
    pub async fn async_poll(&self, maintain: wgpu::PollType) -> wgpu::PollStatus {
        self.device.poll(maintain)
    }
}
