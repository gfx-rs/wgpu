use crate::TestingContext;

impl TestingContext {
    /// Utility to allow future asynchronous polling.
    pub async fn async_poll(
        &self,
        maintain: wgpu::PollType,
    ) -> Result<wgpu::PollStatus, wgpu::PollError> {
        self.device.poll(maintain)
    }
}
