//! A mock [`BlockBackend`] for tests: tracks live blocks, and can be configured to
//! fail block creation on demand (the Nth call, or when total live bytes would exceed
//! a cap) so the pool's failure-halving retry and error paths are exercised.

use alloc::vec::Vec;

use crate::{BlockBackend, BlockId};

/// The opaque per-block value the mock hands the pool: its id and size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MockBlock {
    pub id: BlockId,
    pub size: u64,
}

/// A record of one create/destroy event, for assertions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Create { id: BlockId, size: u64 },
    Destroy { id: BlockId, size: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MockError;

impl core::fmt::Display for MockError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("mock backend failed to create a block")
    }
}

/// A Vec-based failure-injectable backend.
#[derive(Debug, Default)]
pub(crate) struct MockBackend {
    /// Total number of `create_block` calls (including failures), for fail-Nth.
    pub create_calls: usize,
    /// If set, every `create_block` call with an index in this list fails.
    pub fail_create_indices: Vec<usize>,
    /// If set, a create that would push total live bytes above this cap fails.
    pub byte_cap: Option<u64>,
    /// Current total live block bytes.
    pub live_bytes: u64,
    /// Number of live blocks.
    pub live_blocks: usize,
    /// Event log (create/destroy), in order.
    pub events: Vec<Event>,
}

impl MockBackend {
    pub(crate) fn new() -> Self {
        MockBackend::default()
    }

    /// Fail the create calls whose 0-based index is in `indices`.
    pub(crate) fn failing_at(indices: &[usize]) -> Self {
        MockBackend {
            fail_create_indices: indices.to_vec(),
            ..MockBackend::default()
        }
    }

    /// Fail any create that would push total live bytes above `cap`.
    // Only used by the proptests, which are gated off on wasm (see `Cargo.toml`).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_byte_cap(cap: u64) -> Self {
        MockBackend {
            byte_cap: Some(cap),
            ..MockBackend::default()
        }
    }

    pub(crate) fn create_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, Event::Create { .. }))
            .count()
    }

    pub(crate) fn destroy_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, Event::Destroy { .. }))
            .count()
    }
}

impl BlockBackend for MockBackend {
    type Block = MockBlock;
    type Error = MockError;

    fn create_block(&mut self, size: u64, block_id: BlockId) -> Result<Self::Block, Self::Error> {
        let index = self.create_calls;
        self.create_calls += 1;

        if self.fail_create_indices.contains(&index) {
            return Err(MockError);
        }
        if let Some(cap) = self.byte_cap {
            if self.live_bytes.saturating_add(size) > cap {
                return Err(MockError);
            }
        }

        self.live_bytes += size;
        self.live_blocks += 1;
        self.events.push(Event::Create { id: block_id, size });
        Ok(MockBlock { id: block_id, size })
    }

    fn destroy_block(&mut self, block: Self::Block, block_id: BlockId) {
        assert_eq!(block.id, block_id, "destroy_block id mismatch");
        self.live_bytes -= block.size;
        self.live_blocks -= 1;
        self.events.push(Event::Destroy {
            id: block_id,
            size: block.size,
        });
    }
}
