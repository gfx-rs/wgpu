use windows::Win32::Foundation::{CloseHandle, HANDLE};

/// An owned Win32 NT handle that closes itself on drop.
pub struct OwnedHandle(pub HANDLE);

// The handle is just an opaque OS resource; sharing it across threads is sound.
unsafe impl Send for OwnedHandle {}
unsafe impl Sync for OwnedHandle {}

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            let _ = unsafe { CloseHandle(self.0) };
        }
    }
}
