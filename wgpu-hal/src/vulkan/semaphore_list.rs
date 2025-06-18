//! Definition of the [`SemaphoreList`] type.

use alloc::vec::Vec;
use ash::vk;
use core::mem::MaybeUninit;

/// A list of Vulkan semaphores to signal.
///
/// This represents a list of binary or timeline semaphores, together
/// with values for the timeline semaphores.
///
/// This type ensures that the array of semaphores to be signaled
/// stays aligned with the array of values for timeline semaphores
/// appearing in that list. The [`add_to_submit`] method prepares the
/// `vkQueueSubmit` arguments appropriately for whatever semaphores we
/// actually have.
///
/// [`add_to_submit`]: SemaphoreList::add_to_submit
#[derive(Debug, Default)]
pub struct SemaphoreList {
    /// Semaphores to signal.
    ///
    /// This can be a mix of binary and timeline semaphores.
    semaphores: Vec<vk::Semaphore>,

    /// Values for timeline semaphores.
    ///
    /// If no timeline semaphores are present in [`semaphores`], this
    /// is empty. If any timeline semaphores are present, then this
    /// has the same length as [`semaphores`], with dummy !0 values
    /// in the elements corresponding to binary semaphores, since
    /// Vulkan ignores these.
    ///
    /// [`semaphores`]: Self::semaphores
    values: Vec<u64>,
}

impl SemaphoreList {
    pub fn is_empty(&self) -> bool {
        self.semaphores.is_empty()
    }

    /// Add this list to the semaphores to be signalled by a `vkQueueSubmit` call.
    ///
    /// - Set `submit_info`'s `pSignalSemaphores` list to this list's
    ///   semaphores.
    ///
    /// - If this list contains any timeline semaphores, then initialize
    ///   `timeline_info`, set its `pSignalSemaphoreValues` to this
    ///   list's values, and add it to `submit_info`s extension chain.
    ///
    /// Return the revised `submit_info` value.
    pub fn add_to_submit<'i, 's: 'i>(
        &'s self,
        submit_info: vk::SubmitInfo<'i>,
        timeline_info: &'i mut MaybeUninit<vk::TimelineSemaphoreSubmitInfo<'i>>,
    ) -> vk::SubmitInfo<'i> {
        self.check();
        let mut submit_info = submit_info.signal_semaphores(&self.semaphores);
        if !self.values.is_empty() {
            let timeline_info = timeline_info.write(
                vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&self.values),
            );
            submit_info = submit_info.push_next(timeline_info);
        }
        submit_info
    }

    /// Add a binary semaphore to this list.
    pub fn push_binary(&mut self, semaphore: vk::Semaphore) {
        self.semaphores.push(semaphore);
        // Push a dummy value if necessary.
        if !self.values.is_empty() {
            self.values.push(!0);
        }
        self.check();
    }

    /// Add a timeline semaphore to this list, to be signalled with
    /// `value`.
    pub fn push_timeline(&mut self, semaphore: vk::Semaphore, value: u64) {
        self.pad_values();
        self.semaphores.push(semaphore);
        self.values.push(value);
        self.check();
    }

    /// Append `other` to `self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        // If we're about to receive values, ensure we're aligned first.
        if !other.values.is_empty() {
            self.pad_values();
        }
        self.semaphores.append(&mut other.semaphores);
        self.values.append(&mut other.values);
        // If we had values, but `other` did not, re-align.
        if !self.values.is_empty() {
            self.pad_values();
        }
        self.check();
    }

    /// Pad `self.values` with dummy values for binary semaphores,
    /// in preparation for adding a timeline semaphore value.
    ///
    /// This is a no-op if we already have values.
    fn pad_values(&mut self) {
        self.values.resize(self.semaphores.len(), !0);
    }

    #[track_caller]
    fn check(&self) {
        debug_assert!(self.values.is_empty() || self.values.len() == self.semaphores.len());
    }
}
