//! Ranks for `wgpu-core` locks, restricting acquisition order.
//!
//! See [`LockRank`].

/// The rank of a lock.
///
/// Each [`Mutex`], [`RwLock`], and [`SnatchLock`] in `wgpu-core` has been
/// assigned a *rank*: a node in the DAG defined at the bottom of
/// `wgpu-core/src/lock/rank.rs`. The rank of the most recently
/// acquired lock you are still holding determines which locks you may
/// attempt to acquire next.
///
/// When you create a lock in `wgpu-core`, you must specify its rank
/// by passing in a [`LockRank`] value. This module declares a
/// pre-defined set of ranks to cover everything in `wgpu-core`, named
/// after the type in which they occur, and the name of the type's
/// field that is a lock. For example, [`CommandBuffer::data`] is a
/// `Mutex`, and its rank here is the constant
/// [`COMMAND_BUFFER_DATA`].
///
/// [`Mutex`]: parking_lot::Mutex
/// [`RwLock`]: parking_lot::RwLock
/// [`SnatchLock`]: crate::snatch::SnatchLock
/// [`CommandBuffer::data`]: crate::command::CommandBuffer::data
#[derive(Debug, Copy, Clone)]
pub struct LockRank {
    /// The bit representing this lock.
    ///
    /// There should only be a single bit set in this value.
    pub(super) bit: LockRankSet,

    /// A bitmask of permitted successor ranks.
    ///
    /// If `rank` is the rank of the most recently acquired lock we
    /// are still holding, then `rank.followers` is the mask of
    /// locks we are allowed to acquire next.
    ///
    /// The `define_lock_ranks!` macro ensures that there are no
    /// cycles in the graph of lock ranks and their followers.
    pub(super) followers: LockRankSet,
}

/// Define a set of lock ranks, and each rank's permitted successors.
macro_rules! define_lock_ranks {
    {
        $(
            $( #[ $attr:meta ] )*
            rank $name:ident $member:literal followed by { $( $follower:ident ),* $(,)? }
        )*
    } => {
        // An enum that assigns a unique number to each rank.
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        enum LockRankNumber { $( $name, )* }

        bitflags::bitflags! {
            #[derive(Debug, Copy, Clone, Eq, PartialEq)]
            /// A bitflags type representing a set of lock ranks.
            pub struct LockRankSet: u64 {
                $(
                    const $name = 1 << (LockRankNumber:: $name as u64);
                )*
            }
        }

        impl LockRankSet {
            pub fn name(self) -> &'static str {
                match self {
                    $(
                        LockRankSet:: $name => $member,
                    )*
                    _ => "<unrecognized LockRankSet bit>",
                }
            }
        }

        $(
            // If there is any cycle in the ranking, the initializers
            // for `followers` will be cyclic, and rustc will give us
            // an error message explaining the cycle.
            $( #[ $attr ] )*
            pub const $name: LockRank = LockRank {
                bit: LockRankSet:: $name,
                followers: LockRankSet::empty() $( .union($follower.bit) )*,
            };
        )*
    }
}

define_lock_ranks! {
    rank DEVICE_TEMP_SUSPECTED "Device::temp_suspected" followed by {
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER,
        COMMAND_BUFFER_DATA,
        DEVICE_TRACKERS,
    }
    rank COMMAND_BUFFER_DATA "CommandBuffer::data" followed by {
        DEVICE_SNATCHABLE_LOCK,
        DEVICE_USAGE_SCOPES,
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER,
        BUFFER_BIND_GROUP_STATE_BUFFERS,
        TEXTURE_BIND_GROUP_STATE_TEXTURES,
        BUFFER_MAP_STATE,
        STATELESS_BIND_GROUP_STATE_RESOURCES,
    }
    rank DEVICE_SNATCHABLE_LOCK "Device::snatchable_lock" followed by {
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER,
        DEVICE_TRACE,
        BUFFER_MAP_STATE,
        BUFFER_BIND_GROUP_STATE_BUFFERS,
        TEXTURE_BIND_GROUP_STATE_TEXTURES,
        STATELESS_BIND_GROUP_STATE_RESOURCES,
        // Uncomment this to see an interesting cycle.
        // COMMAND_BUFFER_DATA,
    }
    rank BUFFER_MAP_STATE "Buffer::map_state" followed by {
        DEVICE_PENDING_WRITES,
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER,
        DEVICE_TRACE,
    }
    rank DEVICE_PENDING_WRITES "Device::pending_writes" followed by {
        COMMAND_ALLOCATOR_FREE_ENCODERS,
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER,
        DEVICE_LIFE_TRACKER,
    }
    rank DEVICE_LIFE_TRACKER "Device::life_tracker" followed by {
        COMMAND_ALLOCATOR_FREE_ENCODERS,
        // Uncomment this to see an interesting cycle.
        // DEVICE_TEMP_SUSPECTED,
        DEVICE_TRACE,
    }
    rank COMMAND_ALLOCATOR_FREE_ENCODERS "CommandAllocator::free_encoders" followed by {
        SHARED_TRACKER_INDEX_ALLOCATOR_INNER,
    }

    rank BUFFER_BIND_GROUPS "Buffer::bind_groups" followed by { }
    rank BUFFER_BIND_GROUP_STATE_BUFFERS "BufferBindGroupState::buffers" followed by { }
    rank BUFFER_INITIALIZATION_STATUS "Buffer::initialization_status" followed by { }
    rank BUFFER_SYNC_MAPPED_WRITES "Buffer::sync_mapped_writes" followed by { }
    rank DEVICE_DEFERRED_DESTROY "Device::deferred_destroy" followed by { }
    rank DEVICE_FENCE "Device::fence" followed by { }
    #[allow(dead_code)]
    rank DEVICE_TRACE "Device::trace" followed by { }
    rank DEVICE_TRACKERS "Device::trackers" followed by { }
    rank DEVICE_USAGE_SCOPES "Device::usage_scopes" followed by { }
    rank IDENTITY_MANAGER_VALUES "IdentityManager::values" followed by { }
    rank REGISTRY_STORAGE "Registry::storage" followed by { }
    rank RENDER_BUNDLE_SCOPE_BUFFERS "RenderBundleScope::buffers" followed by { }
    rank RENDER_BUNDLE_SCOPE_TEXTURES "RenderBundleScope::textures" followed by { }
    rank RENDER_BUNDLE_SCOPE_BIND_GROUPS "RenderBundleScope::bind_groups" followed by { }
    rank RENDER_BUNDLE_SCOPE_RENDER_PIPELINES "RenderBundleScope::render_pipelines" followed by { }
    rank RENDER_BUNDLE_SCOPE_QUERY_SETS "RenderBundleScope::query_sets" followed by { }
    rank RESOURCE_POOL_INNER "ResourcePool::inner" followed by { }
    rank SHARED_TRACKER_INDEX_ALLOCATOR_INNER "SharedTrackerIndexAllocator::inner" followed by { }
    rank STAGING_BUFFER_RAW "StagingBuffer::raw" followed by { }
    rank STATELESS_BIND_GROUP_STATE_RESOURCES "StatelessBindGroupState::resources" followed by { }
    rank SURFACE_PRESENTATION "Surface::presentation" followed by { }
    rank TEXTURE_BIND_GROUPS "Texture::bind_groups" followed by { }
    rank TEXTURE_BIND_GROUP_STATE_TEXTURES "TextureBindGroupState::textures" followed by { }
    rank TEXTURE_INITIALIZATION_STATUS "Texture::initialization_status" followed by { }
    rank TEXTURE_CLEAR_MODE "Texture::clear_mode" followed by { }
    rank TEXTURE_VIEWS "Texture::views" followed by { }

    #[cfg(test)]
    rank PAWN "pawn" followed by { ROOK, BISHOP }
    #[cfg(test)]
    rank ROOK "rook" followed by { KNIGHT }
    #[cfg(test)]
    rank KNIGHT "knight" followed by { }
    #[cfg(test)]
    rank BISHOP "bishop" followed by { }
}
