use crate::TestingContext;

impl TestingContext {
    /// Utility to allow future asynchronous polling.
    pub async fn async_poll(
        &self,
        poll_type: wgpu::PollType,
    ) -> Result<wgpu::PollStatus, wgpu::PollError> {
        self.device.poll(poll_type)
    }
}
