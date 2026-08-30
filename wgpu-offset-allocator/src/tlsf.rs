//! Two-Level Segregated Fit allocator.
//!
//! Port of `VmaBlockMetadata_TLSF` (vk_mem_alloc.h) and D3D12MA's
//! `BlockMetadata_TLSF`. The two C++ implementations are identical for the parts we
//! port; where they use different terms they are footnoted below.
//!
//! # Arena of indices
//!
//! The C++ code threads two intrusive doubly linked lists through heap-allocated
//! `Block` nodes: a physical-address-ordered list (`prevPhysical`/`nextPhysical`)
//! and a per-size-class free list (`prevFree`/`nextFree`). It uses `prevFree == self`
//! as the "taken" marker and a `union` overlaying `nextFree` with `userData`.
//!
//! To stay in safe Rust we keep the nodes in a [`Vec`] arena addressed by [`u32`]
//! indices ([`NodeIndex`]), with a free-index recycling list ([`Nodes::recycle`]).
//! The taken/free distinction and the free-list links are separate, explicit fields
//! ([`NodeState`]) rather than a pointer union.
//!
//! # Bitmap widths
//!
//! VMA/D3D12MA store the top-level "which memory classes are non-empty" bitmap in a
//! `uint32_t`, which only holds for block sizes below ~512 GiB (memory class stays
//! `< 32`). To support the full [`u64`] size range without any undefined shifts, we
//! widen the top-level bitmap to [`u64`] ([`is_free_bitmap`](Tlsf::is_free_bitmap)).
//! The inner (second-level) bitmaps remain [`u32`], because there are always exactly
//! `2^SLI == 32` second-level lists. See the crate docs' fidelity notes.

use alloc::vec;
use alloc::vec::Vec;

use crate::granularity::BufferImageGranularity;
use crate::math::{
    align_up, bit_scan_lsb_u32, bit_scan_lsb_u64, bit_scan_msb_u64, is_pow2, shl_all_ones_u32,
    shl_all_ones_u64,
};
use crate::statistics::{DetailedStatistics, Statistics};
use crate::{
    AllocationDesc, AllocationError, AllocationHandle, AllocationInfo, AllocationRequest,
    AllocationType, CreateError, HandleError, RequestPayload, Strategy, Suballocator,
};

const SECOND_LEVEL_INDEX: u32 = 5;
const SMALL_BUFFER_SIZE: u64 = 256;
const MEMORY_CLASS_SHIFT: u32 = 7;
/// Number of inner (second-level) lists per non-small memory class: `2^SLI`.
const SECOND_LEVEL_COUNT: u32 = 1 << SECOND_LEVEL_INDEX; // 32
/// Initial arena capacity, mirroring VMA's `INITIAL_BLOCK_ALLOC_COUNT`.
const INITIAL_BLOCK_ALLOC_COUNT: usize = 16;

/// Index of a block node in the arena. `u32::MAX` is the null sentinel (there is no
/// pointer to be null in safe Rust).
type NodeIndex = u32;
const NULL_NODE: NodeIndex = u32::MAX;

/// Free-list linkage / taken marker for a block node.
///
/// Replaces the C++ `prevFree == self` union trick with an explicit enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeState {
    /// The block is a live allocation. Carries the user-data slot index (see
    /// [`Tlsf::user_data`]); `u32::MAX` when there is no user data yet.
    Taken { user_data: u32 },
    /// The block is free and linked into a per-size-class free list.
    Free {
        prev_free: NodeIndex,
        next_free: NodeIndex,
    },
}

/// A single block node: a physical run of the address space, either free or taken.
#[derive(Clone, Copy, Debug)]
struct Block {
    offset: u64,
    size: u64,
    /// Address-ordered doubly linked list.
    prev_physical: NodeIndex,
    next_physical: NodeIndex,
    state: NodeState,
    /// Per-slot generation, bumped every time the slot is freed/recycled/cleared.
    /// Packed into the high 32 bits of an [`AllocationHandle`] so a stale handle to a
    /// recycled slot is caught by a mismatch (see [`handle_of`]/[`node_of`]).
    generation: u32,
    /// `true` when this free node is a debug-margin filler inserted by
    /// [`alloc_impl`](Tlsf::alloc_impl). Margin fillers must never merge with a
    /// neighbouring free block (they belong to their owning allocation until it is
    /// freed); genuine free blocks always may. Replaces the fragile
    /// `size == debug_margin` heuristic that stranded real free blocks whose size
    /// happened to equal the margin. Meaningful only while the node is free.
    is_margin_filler: bool,
}

impl Block {
    #[inline]
    fn is_free(&self) -> bool {
        matches!(self.state, NodeState::Free { .. })
    }
}

/// The arena of block nodes plus a recycling free-index list.
#[derive(Debug)]
struct Nodes {
    slots: Vec<Block>,
    /// Indices of `slots` entries that are unused and available for reuse.
    recycle: Vec<NodeIndex>,
}

impl Nodes {
    fn new() -> Self {
        Nodes {
            slots: Vec::with_capacity(INITIAL_BLOCK_ALLOC_COUNT),
            recycle: Vec::new(),
        }
    }

    /// Allocates a node slot, returning its index. Reuses a recycled slot if any.
    ///
    /// A recycled slot keeps the generation it was bumped to when it was freed (see
    /// [`free`](Self::free)); this method preserves that generation while overwriting
    /// the rest of the block, so a stale handle to the slot's previous occupant no
    /// longer validates. Fresh slots start at generation 0.
    fn alloc(&mut self, mut block: Block) -> NodeIndex {
        if let Some(index) = self.recycle.pop() {
            // Preserve the recycled slot's (already-bumped) generation.
            block.generation = self.slots[index as usize].generation;
            self.slots[index as usize] = block;
            index
        } else {
            let index = self.slots.len();
            // A block can never exceed u32::MAX nodes: each node covers at least one
            // unit and NULL_NODE reserves the top index. In practice the arena stays
            // tiny. Guard the invariant rather than risk a silent wrap.
            assert!(
                index < NULL_NODE as usize,
                "TLSF arena exhausted (too many nodes)"
            );
            block.generation = 0;
            self.slots.push(block);
            index as NodeIndex
        }
    }

    /// Returns a node slot to the recycling list, bumping its generation so any
    /// outstanding handle to the slot's current occupant is invalidated.
    fn free(&mut self, index: NodeIndex) {
        self.bump_generation(index);
        self.recycle.push(index);
    }

    /// Bumps a slot's generation (wrapping). Used on free/recycle and on `clear` so
    /// stale handles fail the generation check.
    #[inline]
    fn bump_generation(&mut self, index: NodeIndex) {
        let g = &mut self.slots[index as usize].generation;
        *g = g.wrapping_add(1);
    }

    #[inline]
    fn get(&self, index: NodeIndex) -> &Block {
        &self.slots[index as usize]
    }

    #[inline]
    fn get_mut(&mut self, index: NodeIndex) -> &mut Block {
        &mut self.slots[index as usize]
    }
}

/// A Two-Level Segregated Fit suballocator.
///
/// See the crate-level documentation for the algorithm overview and the
/// arena-of-indices design.
#[derive(Debug)]
pub struct Tlsf<T> {
    size: u64,
    is_virtual: bool,
    debug_margin: u64,
    granularity: BufferImageGranularity,

    nodes: Nodes,
    /// User-data slab, indexed by the `user_data` field of [`NodeState::Taken`]. Kept
    /// separate from the arena so a taken block need only store a `u32` index.
    user_data: Vec<Option<T>>,
    user_data_recycle: Vec<u32>,

    alloc_count: usize,
    /// Number of free blocks *excluding* the null block.
    blocks_free_count: usize,
    /// Total size of free blocks *excluding* the null block.
    blocks_free_size: u64,

    /// Top-level bitmap: bit `c` set iff memory class `c` has a non-empty free list.
    /// Widened to `u64` versus VMA's `u32` (see module docs).
    is_free_bitmap: u64,
    /// Per-memory-class inner bitmaps: bit `s` of `inner[c]` set iff `free_list` for
    /// `(c, s)` is non-empty. `2^SLI == 32` lists, so `u32` suffices.
    inner_is_free_bitmap: Vec<u32>,
    /// `memoryClass + 2` at construction; the number of valid entries of
    /// `inner_is_free_bitmap` and the top-level bitmap.
    memory_classes: u32,
    /// Number of entries in `free_list`.
    lists_count: u32,
    /// Free-list heads, one per `(memory_class, second_index)` list. `NULL_NODE` when
    /// empty.
    free_list: Vec<NodeIndex>,

    /// The distinguished trailing free block. Never on any free list, always
    /// physically last.
    null_block: NodeIndex,
}

impl<T> Tlsf<T> {
    // --- size-class math (ports of SizeToMemoryClass / SizeToSecondIndex / GetListIndex) ---

    /// `SizeToMemoryClass`: size <= 256 -> class 0, else `MSB(size) - 7`.
    fn size_to_memory_class(size: u64) -> u32 {
        if size > SMALL_BUFFER_SIZE {
            bit_scan_msb_u64(size) - MEMORY_CLASS_SHIFT
        } else {
            0
        }
    }

    /// `SizeToSecondIndex`. Class 0 uses 32 buckets (virtual) or 4 buckets (real);
    /// higher classes split by the `SLI` bits below the MSB.
    fn size_to_second_index(&self, size: u64, memory_class: u32) -> u32 {
        if memory_class == 0 {
            if self.is_virtual {
                ((size - 1) / 8) as u32
            } else {
                ((size - 1) / 64) as u32
            }
        } else {
            // (size >> (memoryClass + 7 - 5)) ^ 32
            ((size >> (memory_class + MEMORY_CLASS_SHIFT - SECOND_LEVEL_INDEX)) as u32)
                ^ SECOND_LEVEL_COUNT
        }
    }

    /// `GetListIndex(memoryClass, secondIndex)` with the +4 (real) / +32 (virtual)
    /// offset for the small-buffer lists at class 0.
    fn list_index_from_class(&self, memory_class: u32, second_index: u32) -> u32 {
        if memory_class == 0 {
            return second_index;
        }
        let index = (memory_class - 1) * SECOND_LEVEL_COUNT + second_index;
        if self.is_virtual {
            index + SECOND_LEVEL_COUNT
        } else {
            index + 4
        }
    }

    /// `GetListIndex(size)`.
    fn list_index_from_size(&self, size: u64) -> u32 {
        let mc = Self::size_to_memory_class(size);
        self.list_index_from_class(mc, self.size_to_second_index(size, mc))
    }

    // --- free-list maintenance (ports of RemoveFreeBlock / InsertFreeBlock / MergeBlock) ---

    /// Removes `block` from its free list and marks it taken. Port of `RemoveFreeBlock`.
    fn remove_free_block(&mut self, block: NodeIndex) {
        debug_assert!(block != self.null_block);
        debug_assert!(self.nodes.get(block).is_free());

        let (prev_free, next_free, size) = match self.nodes.get(block).state {
            NodeState::Free {
                prev_free,
                next_free,
            } => (prev_free, next_free, self.nodes.get(block).size),
            NodeState::Taken { .. } => unreachable!("remove_free_block on taken block"),
        };
        self.splice_out_free(block, prev_free, next_free, size);
    }

    /// Helper for [`remove_free_block`](Self::remove_free_block): performs the actual
    /// free-list unlink and bitmap update.
    fn splice_out_free(
        &mut self,
        block: NodeIndex,
        prev_free: NodeIndex,
        next_free: NodeIndex,
        size: u64,
    ) {
        if next_free != NULL_NODE {
            set_prev_free(self.nodes.get_mut(next_free), prev_free);
        }
        if prev_free != NULL_NODE {
            set_next_free(self.nodes.get_mut(prev_free), next_free);
        } else {
            let mem_class = Self::size_to_memory_class(size);
            let second_index = self.size_to_second_index(size, mem_class);
            let index = self.list_index_from_class(mem_class, second_index);
            debug_assert_eq!(self.free_list[index as usize], block);
            self.free_list[index as usize] = next_free;
            if next_free == NULL_NODE {
                self.inner_is_free_bitmap[mem_class as usize] &= !(1u32 << second_index);
                if self.inner_is_free_bitmap[mem_class as usize] == 0 {
                    self.is_free_bitmap &= !(1u64 << mem_class);
                }
            }
        }
        // Mark taken with no user data yet.
        self.nodes.get_mut(block).state = NodeState::Taken {
            user_data: u32::MAX,
        };
        self.blocks_free_count -= 1;
        self.blocks_free_size -= size;
    }

    /// Inserts `block` at the head of its size-class free list. Port of `InsertFreeBlock`.
    fn insert_free_block(&mut self, block: NodeIndex) {
        debug_assert!(block != self.null_block);
        debug_assert!(
            !self.nodes.get(block).is_free(),
            "cannot insert block twice"
        );

        let size = self.nodes.get(block).size;
        let mem_class = Self::size_to_memory_class(size);
        let second_index = self.size_to_second_index(size, mem_class);
        let index = self.list_index_from_class(mem_class, second_index);
        debug_assert!((index as usize) < self.free_list.len());

        let old_head = self.free_list[index as usize];
        self.nodes.get_mut(block).state = NodeState::Free {
            prev_free: NULL_NODE,
            next_free: old_head,
        };
        self.free_list[index as usize] = block;
        if old_head != NULL_NODE {
            set_prev_free(self.nodes.get_mut(old_head), block);
        } else {
            self.inner_is_free_bitmap[mem_class as usize] |= 1u32 << second_index;
            self.is_free_bitmap |= 1u64 << mem_class;
        }
        self.blocks_free_count += 1;
        self.blocks_free_size += size;
    }

    /// Merges free/taken `prev` (physically before `block`) into `block`, freeing
    /// `prev`'s node slot. Port of `MergeBlock`. `prev` must already be removed from
    /// any free list.
    fn merge_block(&mut self, block: NodeIndex, prev: NodeIndex) {
        debug_assert_eq!(self.nodes.get(block).prev_physical, prev);
        debug_assert!(
            !self.nodes.get(prev).is_free(),
            "merging a free-listed block"
        );

        let prev_offset = self.nodes.get(prev).offset;
        let prev_size = self.nodes.get(prev).size;
        let prev_prev = self.nodes.get(prev).prev_physical;

        let b = self.nodes.get_mut(block);
        b.offset = prev_offset;
        b.size += prev_size;
        b.prev_physical = prev_prev;
        // A coalesced region is never a pure margin filler: clear the marker so the
        // survivor is treated as a genuine free block by later merge guards. This
        // covers the case where the survivor is the just-absorbed margin node.
        b.is_margin_filler = false;
        if prev_prev != NULL_NODE {
            self.nodes.get_mut(prev_prev).next_physical = block;
        }
        self.nodes.free(prev);
    }

    // --- search (ports of FindFreeBlock / CheckBlock) ---

    /// `FindFreeBlock`: two-level bitmap scan for the lowest free block of at least
    /// `size`. Returns `(block, list_index)` or `None`.
    fn find_free_block(&self, size: u64) -> Option<(NodeIndex, u32)> {
        let mut memory_class = Self::size_to_memory_class(size);
        let second_index = self.size_to_second_index(size, memory_class);
        let mut inner_free_map =
            self.inner_is_free_bitmap[memory_class as usize] & shl_all_ones_u32(second_index);
        if inner_free_map == 0 {
            // Check higher memory classes.
            let free_map = self.is_free_bitmap & shl_all_ones_u64(memory_class + 1);
            if free_map == 0 {
                return None;
            }
            memory_class = bit_scan_lsb_u64(free_map);
            inner_free_map = self.inner_is_free_bitmap[memory_class as usize];
            debug_assert!(inner_free_map != 0);
        }
        let list_index = self.list_index_from_class(memory_class, bit_scan_lsb_u32(inner_free_map));
        let head = self.free_list[list_index as usize];
        debug_assert!(head != NULL_NODE);
        Some((head, list_index))
    }

    /// `CheckBlock`: does `block` admit an allocation of `alloc_size`/`alloc_alignment`
    /// with `alloc_type`? If so, records the placement in an [`AllocationRequest`],
    /// moves `block` to the head of its free list (unless it is the null block), and
    /// returns `Some`.
    ///
    /// `alloc_size` here already includes the debug margin (as in VMA).
    fn check_block(
        &mut self,
        block: NodeIndex,
        list_index: u32,
        alloc_size: u64,
        alloc_alignment: u64,
        alloc_type: AllocationType,
    ) -> Option<AllocationRequest> {
        debug_assert!(self.nodes.get(block).is_free());

        let block_offset = self.nodes.get(block).offset;
        let block_size = self.nodes.get(block).size;

        let mut aligned_offset = align_up(block_offset, alloc_alignment);
        // Overflow-safe form of `block.size < allocSize + alignedOffset - block.offset`.
        // padding = alignedOffset - block.offset (align_up never decreases the value,
        // and saturation only occurs beyond u64 range, which also fails the fit test).
        let padding = aligned_offset - block_offset;
        if padding > block_size || block_size - padding < alloc_size {
            return None;
        }

        // Granularity conflict check (only for real blocks with page tracking).
        if !self.is_virtual
            && self.granularity.check_conflict_and_align_up(
                &mut aligned_offset,
                alloc_size,
                block_offset,
                block_size,
                alloc_type,
            )
        {
            return None;
        }

        // Move a normal (non-null) block to the head of its list, so subsequent
        // searches find it first. This is the sole "mutation" allowed in the query
        // phase; it preserves every invariant.
        if list_index != self.lists_count {
            self.move_block_to_list_head(block, list_index);
        }

        Some(AllocationRequest {
            offset: aligned_offset,
            size: alloc_size - self.debug_margin,
            payload: RequestPayload::Tlsf { block, alloc_type },
        })
    }

    /// Moves an already-free `block` to the head of free list `list_index`.
    fn move_block_to_list_head(&mut self, block: NodeIndex, list_index: u32) {
        let (prev_free, next_free) = match self.nodes.get(block).state {
            NodeState::Free {
                prev_free,
                next_free,
            } => (prev_free, next_free),
            NodeState::Taken { .. } => return,
        };
        // Only reorder if the block is not already the head (i.e. has a predecessor).
        if prev_free == NULL_NODE {
            return;
        }
        // Unlink.
        set_next_free(self.nodes.get_mut(prev_free), next_free);
        if next_free != NULL_NODE {
            set_prev_free(self.nodes.get_mut(next_free), prev_free);
        }
        // Relink at head.
        let old_head = self.free_list[list_index as usize];
        self.nodes.get_mut(block).state = NodeState::Free {
            prev_free: NULL_NODE,
            next_free: old_head,
        };
        self.free_list[list_index as usize] = block;
        if old_head != NULL_NODE {
            set_prev_free(self.nodes.get_mut(old_head), block);
        }
    }

    // --- user data slab ---

    fn store_user_data(&mut self, user_data: T) -> u32 {
        if let Some(index) = self.user_data_recycle.pop() {
            self.user_data[index as usize] = Some(user_data);
            index
        } else {
            let index = self.user_data.len() as u32;
            self.user_data.push(Some(user_data));
            index
        }
    }

    fn take_user_data(&mut self, index: u32) -> Option<T> {
        if index == u32::MAX {
            return None;
        }
        let value = self.user_data[index as usize].take();
        self.user_data_recycle.push(index);
        value
    }

    /// Returns the user-data slab index of a taken block, or panics in debug if free.
    fn taken_user_data_index(&self, handle: NodeIndex) -> u32 {
        match self.nodes.get(handle).state {
            NodeState::Taken { user_data } => user_data,
            NodeState::Free { .. } => {
                debug_assert!(false, "operation on a freed TLSF handle");
                u32::MAX
            }
        }
    }

    /// Validates a public [`AllocationHandle`] and returns the live node index it
    /// identifies, or [`HandleError`] if it is stale / double-freed / foreign.
    ///
    /// Checks, in order: the node index is within the arena, the packed generation
    /// matches the slot's current generation, and the node is currently a live
    /// (taken) allocation and not the trailing null block. This is the single choke
    /// point that keeps the internal `unreachable!`/`debug_assert!` guards genuinely
    /// unreachable through the public API.
    fn resolve_handle(&self, handle: AllocationHandle) -> Result<NodeIndex, HandleError> {
        let node = node_of(handle);
        let index = node as usize;
        if index >= self.nodes.slots.len() {
            return Err(HandleError::InvalidHandle);
        }
        let block = self.nodes.get(node);
        if block.generation != generation_of(handle) {
            return Err(HandleError::InvalidHandle);
        }
        // Free nodes (including the null block) and internal filler/margin nodes are
        // never live allocations. Only Taken nodes are valid handle targets, and the
        // null block is Free, so it is excluded here too.
        if block.is_free() {
            return Err(HandleError::InvalidHandle);
        }
        Ok(node)
    }

    /// Builds the public handle for a live node, packing its current generation.
    #[inline]
    fn handle_for(&self, node: NodeIndex) -> AllocationHandle {
        handle_of(node, self.nodes.get(node).generation)
    }
}

// Free-standing helpers that poke a single node's free-list links, kept outside the
// impl to avoid borrow-checker friction with `Nodes::get_mut`.
#[inline]
fn set_prev_free(block: &mut Block, value: NodeIndex) {
    if let NodeState::Free {
        ref mut prev_free, ..
    } = block.state
    {
        *prev_free = value;
    } else {
        debug_assert!(false, "set_prev_free on taken block");
    }
}

#[inline]
fn set_next_free(block: &mut Block, value: NodeIndex) {
    if let NodeState::Free {
        ref mut next_free, ..
    } = block.state
    {
        *next_free = value;
    } else {
        debug_assert!(false, "set_next_free on taken block");
    }
}

/// A TLSF handle packs the `u32` node index in the low 32 bits and the node's `u32`
/// generation in the high 32 bits of the crate-wide `u64` [`AllocationHandle`]. The
/// generation lets us detect stale/double-free/foreign handles (see [`HandleError`]).
#[inline]
fn handle_of(node: NodeIndex, generation: u32) -> AllocationHandle {
    AllocationHandle(((generation as u64) << 32) | (node as u64))
}

#[inline]
fn node_of(handle: AllocationHandle) -> NodeIndex {
    (handle.0 & 0xFFFF_FFFF) as NodeIndex
}

#[inline]
fn generation_of(handle: AllocationHandle) -> u32 {
    (handle.0 >> 32) as u32
}

impl<T> Suballocator<T> for Tlsf<T> {
    fn new(
        size: u64,
        granularity: u64,
        is_virtual: bool,
        debug_margin: u64,
    ) -> Result<Self, CreateError> {
        validate_create_args(size, granularity, debug_margin)?;
        // Debug margin is forced off for virtual blocks (matches GetDebugMargin).
        let debug_margin = if is_virtual { 0 } else { debug_margin };

        let mut this = Tlsf {
            size,
            is_virtual,
            debug_margin,
            granularity: BufferImageGranularity::new(granularity, size, is_virtual)?,
            nodes: Nodes::new(),
            user_data: Vec::new(),
            user_data_recycle: Vec::new(),
            alloc_count: 0,
            blocks_free_count: 0,
            blocks_free_size: 0,
            is_free_bitmap: 0,
            inner_is_free_bitmap: Vec::new(),
            memory_classes: 0,
            lists_count: 0,
            free_list: Vec::new(),
            null_block: NULL_NODE,
        };
        this.init();
        Ok(this)
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn is_virtual(&self) -> bool {
        self.is_virtual
    }

    fn validate(&self) -> Result<(), &'static str> {
        self.validate_impl()
    }

    fn allocation_count(&self) -> usize {
        self.alloc_count
    }

    fn free_regions_count(&self) -> usize {
        self.blocks_free_count + 1
    }

    fn sum_free_size(&self) -> u64 {
        self.blocks_free_size + self.nodes.get(self.null_block).size
    }

    fn is_empty(&self) -> bool {
        self.nodes.get(self.null_block).offset == 0
    }

    fn allocation_offset(&self, handle: AllocationHandle) -> Result<u64, HandleError> {
        let node = self.resolve_handle(handle)?;
        Ok(self.nodes.get(node).offset)
    }

    fn allocation_info(&self, handle: AllocationHandle) -> Result<AllocationInfo<T>, HandleError>
    where
        T: Clone,
    {
        let node = self.resolve_handle(handle)?;
        let block = self.nodes.get(node);
        let user_data_index = self.taken_user_data_index(node);
        let user_data = if user_data_index == u32::MAX {
            None
        } else {
            self.user_data[user_data_index as usize].clone()
        };
        Ok(AllocationInfo {
            offset: block.offset,
            size: block.size,
            user_data: user_data.expect("TLSF allocation info requires stored user data"),
        })
    }

    fn set_user_data(&mut self, handle: AllocationHandle, user_data: T) -> Result<(), HandleError> {
        let node = self.resolve_handle(handle)?;
        let index = self.taken_user_data_index(node);
        if index == u32::MAX {
            let new_index = self.store_user_data(user_data);
            self.nodes.get_mut(node).state = NodeState::Taken {
                user_data: new_index,
            };
        } else {
            self.user_data[index as usize] = Some(user_data);
        }
        Ok(())
    }

    fn allocation_list_begin(&self) -> Option<AllocationHandle> {
        if self.alloc_count == 0 {
            return None;
        }
        let mut block = self.nodes.get(self.null_block).prev_physical;
        while block != NULL_NODE {
            if !self.nodes.get(block).is_free() {
                return Some(self.handle_for(block));
            }
            block = self.nodes.get(block).prev_physical;
        }
        None
    }

    fn next_allocation(&self, prev: AllocationHandle) -> Option<AllocationHandle> {
        // Validate through the same choke point as the other handle methods: a stale /
        // foreign handle whose low-32 index is out of the arena would otherwise index
        // `self.nodes.slots` out of bounds and panic. On an invalid handle report the
        // end of iteration (`None`), matching this method's failure shape.
        let node = self.resolve_handle(prev).ok()?;
        let mut block = self.nodes.get(node).prev_physical;
        while block != NULL_NODE {
            if !self.nodes.get(block).is_free() {
                return Some(self.handle_for(block));
            }
            block = self.nodes.get(block).prev_physical;
        }
        None
    }

    fn next_free_region_size(&self, alloc: AllocationHandle) -> u64 {
        // Validate the handle first (see `next_allocation`); an invalid handle reports
        // "no free region" (0), matching this method's failure shape.
        let Ok(node) = self.resolve_handle(alloc) else {
            return 0;
        };
        let prev = self.nodes.get(node).prev_physical;
        if prev != NULL_NODE && self.nodes.get(prev).is_free() {
            self.nodes.get(prev).size
        } else {
            0
        }
    }

    fn add_statistics(&self, stats: &mut Statistics) {
        stats.block_count += 1;
        stats.allocation_count += self.alloc_count as u32;
        stats.block_bytes += self.size;
        stats.allocation_bytes += self.size - self.sum_free_size();
    }

    fn add_detailed_statistics(&self, stats: &mut DetailedStatistics) {
        stats.statistics.block_count += 1;
        stats.statistics.block_bytes += self.size;
        let null_size = self.nodes.get(self.null_block).size;
        if null_size > 0 {
            stats.add_unused_range(null_size);
        }
        let mut block = self.nodes.get(self.null_block).prev_physical;
        while block != NULL_NODE {
            let b = self.nodes.get(block);
            if b.is_free() {
                stats.add_unused_range(b.size);
            } else {
                stats.add_allocation(b.size);
            }
            block = b.prev_physical;
        }
    }

    fn create_allocation_request(
        &mut self,
        desc: AllocationDesc,
    ) -> Result<AllocationRequest, AllocationError> {
        let AllocationDesc {
            size,
            alignment,
            alloc_type,
            upper_address,
            strategy,
        } = desc;
        if size == 0 {
            return Err(AllocationError::InvalidSize);
        }
        if !is_pow2(alignment) {
            return Err(AllocationError::InvalidAlignment);
        }
        if upper_address {
            return Err(AllocationError::UpperAddressUnsupported);
        }
        self.create_allocation_request_impl(size, alignment, alloc_type, strategy)
            .ok_or(AllocationError::OutOfSpace)
    }

    fn alloc(&mut self, request: AllocationRequest, user_data: T) -> AllocationHandle {
        let RequestPayload::Tlsf { block, alloc_type } = request.payload;
        self.alloc_impl(block, request.offset, request.size, alloc_type, user_data)
    }

    fn free(&mut self, handle: AllocationHandle) -> Result<(), HandleError> {
        let node = self.resolve_handle(handle)?;
        self.free_impl(node);
        Ok(())
    }

    fn clear(&mut self) {
        self.clear_impl();
    }
}

fn validate_create_args(size: u64, granularity: u64, debug_margin: u64) -> Result<(), CreateError> {
    if size == 0 {
        return Err(CreateError::ZeroSize);
    }
    if granularity == 0 {
        return Err(CreateError::ZeroGranularity);
    }
    if granularity > 1 && !is_pow2(granularity) {
        return Err(CreateError::GranularityNotPowerOfTwo);
    }
    if debug_margin != 0 && !debug_margin.is_multiple_of(4) {
        return Err(CreateError::DebugMarginNotMultipleOfFour);
    }
    Ok(())
}

impl<T> Tlsf<T> {
    /// Port of `Init`: sets up the single null block spanning the whole block and
    /// sizes the free-list / bitmap arrays.
    fn init(&mut self) {
        let size = self.size;
        let null_block = self.nodes.alloc(Block {
            offset: 0,
            size,
            prev_physical: NULL_NODE,
            next_physical: NULL_NODE,
            state: NodeState::Free {
                prev_free: NULL_NODE,
                next_free: NULL_NODE,
            },
            generation: 0,
            is_margin_filler: false,
        });
        self.null_block = null_block;

        let memory_class = Self::size_to_memory_class(size);
        let sli = self.size_to_second_index(size, memory_class);
        let mut lists_count = if memory_class == 0 {
            0
        } else {
            (memory_class - 1) * SECOND_LEVEL_COUNT + sli
        } + 1;
        lists_count += if self.is_virtual {
            SECOND_LEVEL_COUNT
        } else {
            4
        };

        self.memory_classes = memory_class + 2;
        self.lists_count = lists_count;
        self.inner_is_free_bitmap = vec![0u32; self.memory_classes as usize];
        self.free_list = vec![NULL_NODE; lists_count as usize];
    }

    fn create_allocation_request_impl(
        &mut self,
        size: u64,
        alignment: u64,
        alloc_type: AllocationType,
        strategy: Strategy,
    ) -> Option<AllocationRequest> {
        // For low granularity, bump the request up (port of RoundupAllocRequest).
        let (mut alloc_size, alignment) = if !self.is_virtual {
            self.granularity
                .roundup_alloc_request(alloc_type, size, alignment)
        } else {
            (size, alignment)
        };

        // Add debug margin. Overflow would otherwise silently accept a smaller
        // usable allocation than requested.
        alloc_size = alloc_size.checked_add(self.debug_margin)?;

        // Quick reject.
        if alloc_size > self.sum_free_size() {
            return None;
        }

        let lists_count = self.lists_count;

        // If there are no free blocks besides the null block, only it can serve.
        if self.blocks_free_count == 0 {
            return self.check_block(
                self.null_block,
                lists_count,
                alloc_size,
                alignment,
                alloc_type,
            );
        }

        // Round up to the next size-class boundary.
        let size_for_next_list = self.size_for_next_list(alloc_size);

        match strategy {
            Strategy::MinTime => {
                // Larger bucket first.
                if let Some((mut block, mut list_index)) = self.find_free_block(size_for_next_list)
                {
                    if let Some(req) =
                        self.check_block(block, list_index, alloc_size, alignment, alloc_type)
                    {
                        return Some(req);
                    }
                    // Null block.
                    if let Some(req) = self.check_block(
                        self.null_block,
                        lists_count,
                        alloc_size,
                        alignment,
                        alloc_type,
                    ) {
                        return Some(req);
                    }
                    // Walk the larger bucket chain.
                    loop {
                        block = self.next_free(block);
                        if block == NULL_NODE {
                            break;
                        }
                        if let Some(req) =
                            self.check_block(block, list_index, alloc_size, alignment, alloc_type)
                        {
                            return Some(req);
                        }
                    }
                    let _ = &mut list_index;
                } else {
                    // No larger bucket; try the null block before best-fit.
                    if let Some(req) = self.check_block(
                        self.null_block,
                        lists_count,
                        alloc_size,
                        alignment,
                        alloc_type,
                    ) {
                        return Some(req);
                    }
                }
                // Best-fit bucket.
                if let Some(req) = self.search_bucket_chain(alloc_size, alignment, alloc_type) {
                    return Some(req);
                }
            }
            Strategy::MinMemory => {
                // Best-fit bucket.
                if let Some(req) = self.search_bucket_chain(alloc_size, alignment, alloc_type) {
                    return Some(req);
                }
                // Null block.
                if let Some(req) = self.check_block(
                    self.null_block,
                    lists_count,
                    alloc_size,
                    alignment,
                    alloc_type,
                ) {
                    return Some(req);
                }
                // Larger bucket.
                if let Some(req) = self.search_bucket_chain_from(
                    size_for_next_list,
                    alloc_size,
                    alignment,
                    alloc_type,
                ) {
                    return Some(req);
                }
            }
            Strategy::MinOffset => {
                // Gather all fitting free blocks (excluding the null block) sorted by
                // ascending offset, and try them in order.
                let mut candidates: Vec<NodeIndex> = Vec::with_capacity(self.blocks_free_count);
                let mut block = self.nodes.get(self.null_block).prev_physical;
                while block != NULL_NODE {
                    let b = self.nodes.get(block);
                    if b.is_free() && b.size >= alloc_size {
                        candidates.push(block);
                    }
                    block = b.prev_physical;
                }
                // We walked from high offset to low; reverse for ascending offset.
                candidates.reverse();
                for block in candidates {
                    let list_index = self.list_index_from_size(self.nodes.get(block).size);
                    if let Some(req) =
                        self.check_block(block, list_index, alloc_size, alignment, alloc_type)
                    {
                        return Some(req);
                    }
                }
                // Null block last.
                return self.check_block(
                    self.null_block,
                    lists_count,
                    alloc_size,
                    alignment,
                    alloc_type,
                );
            }
            Strategy::Balanced => {
                // Larger bucket.
                if let Some(req) = self.search_bucket_chain_from(
                    size_for_next_list,
                    alloc_size,
                    alignment,
                    alloc_type,
                ) {
                    return Some(req);
                }
                // Null block.
                if let Some(req) = self.check_block(
                    self.null_block,
                    lists_count,
                    alloc_size,
                    alignment,
                    alloc_type,
                ) {
                    return Some(req);
                }
                // Best-fit bucket.
                if let Some(req) = self.search_bucket_chain(alloc_size, alignment, alloc_type) {
                    return Some(req);
                }
            }
        }

        // Worst-case full linear sweep of all higher lists.
        self.full_sweep(alloc_size, alignment, alloc_type)
    }

    /// `sizeForNextList`: rounds `alloc_size` up to the next bucket boundary.
    fn size_for_next_list(&self, alloc_size: u64) -> u64 {
        let small_size_step = SMALL_BUFFER_SIZE / if self.is_virtual { 32 } else { 4 };
        if alloc_size > SMALL_BUFFER_SIZE {
            alloc_size.saturating_add(1u64 << (bit_scan_msb_u64(alloc_size) - SECOND_LEVEL_INDEX))
        } else if alloc_size > SMALL_BUFFER_SIZE - small_size_step {
            SMALL_BUFFER_SIZE + 1
        } else {
            alloc_size + small_size_step
        }
    }

    /// Runs `find_free_block(alloc_size)` then walks the returned bucket chain,
    /// checking each block. Used for the best-fit bucket phase.
    fn search_bucket_chain(
        &mut self,
        alloc_size: u64,
        alignment: u64,
        alloc_type: AllocationType,
    ) -> Option<AllocationRequest> {
        self.search_bucket_chain_from(alloc_size, alloc_size, alignment, alloc_type)
    }

    /// Like [`search_bucket_chain`](Self::search_bucket_chain) but seeds the bitmap
    /// scan with `search_size` (which may be the next-larger-list size), while
    /// checking fit against `alloc_size`.
    fn search_bucket_chain_from(
        &mut self,
        search_size: u64,
        alloc_size: u64,
        alignment: u64,
        alloc_type: AllocationType,
    ) -> Option<AllocationRequest> {
        let (mut block, list_index) = self.find_free_block(search_size)?;
        loop {
            if let Some(req) =
                self.check_block(block, list_index, alloc_size, alignment, alloc_type)
            {
                return Some(req);
            }
            block = self.next_free(block);
            if block == NULL_NODE {
                return None;
            }
        }
    }

    /// The final fallback: linearly sweep every list above the one that
    /// [`find_free_block`](Self::find_free_block) would start at. Mirrors the C++
    /// `while (++nextListIndex < m_ListsCount)` loop, but since our earlier phases may
    /// not have set `nextListIndex`, we conservatively sweep all lists.
    fn full_sweep(
        &mut self,
        alloc_size: u64,
        alignment: u64,
        alloc_type: AllocationType,
    ) -> Option<AllocationRequest> {
        for list_index in 0..self.lists_count {
            let mut block = self.free_list[list_index as usize];
            while block != NULL_NODE {
                if let Some(req) =
                    self.check_block(block, list_index, alloc_size, alignment, alloc_type)
                {
                    return Some(req);
                }
                block = self.next_free(block);
            }
        }
        None
    }

    /// Returns the `next_free` link of a free block, or `NULL_NODE`.
    #[inline]
    fn next_free(&self, block: NodeIndex) -> NodeIndex {
        match self.nodes.get(block).state {
            NodeState::Free { next_free, .. } => next_free,
            NodeState::Taken { .. } => NULL_NODE,
        }
    }

    /// Port of `Alloc`: commit an allocation into free `block` at `offset` with usable
    /// `size`.
    fn alloc_impl(
        &mut self,
        block: NodeIndex,
        offset: u64,
        size: u64,
        alloc_type: AllocationType,
        user_data: T,
    ) -> AllocationHandle {
        debug_assert!(self.nodes.get(block).offset <= offset);

        let current = block;
        if current != self.null_block {
            self.remove_free_block(current);
        }

        let debug_margin = self.debug_margin;
        let missing_alignment = offset - self.nodes.get(current).offset;

        // Absorb the alignment padding. Either grow a free previous physical block, or
        // materialize the padding as a fresh *free* block on the free list.
        //
        // Fidelity note: VMA/D3D12MA write this filler as `MarkTaken()` immediately
        // followed by `InsertFreeBlock()`. Because `InsertFreeBlock` overwrites
        // `PrevFree()` (the same field the taken marker uses) with null, the node ends
        // up *free*. The `MarkTaken()` there only exists to satisfy an internal
        // `!IsFree()` assert. We model the true end state directly: a free block.
        if missing_alignment != 0 {
            let prev = self.nodes.get(current).prev_physical;
            debug_assert!(prev != NULL_NODE, "missing alignment at offset 0");

            let prev_is_free = self.nodes.get(prev).is_free();
            let prev_is_margin = self.nodes.get(prev).is_margin_filler;
            let prev_size = self.nodes.get(prev).size;
            // Grow the previous free block into the padding, unless it is a debug-margin
            // filler (which must stay attached to its owning allocation). Test the
            // explicit marker, not `size == debug_margin`: a genuine free block
            // whose size merely equals the margin must still absorb the padding.
            if prev_is_free && !prev_is_margin {
                let old_list = self.list_index_from_size(prev_size);
                let new_size = prev_size + missing_alignment;
                if old_list != self.list_index_from_size(new_size) {
                    // Re-bucket: remove at old size, grow, reinsert at new size.
                    self.remove_free_block(prev);
                    self.nodes.get_mut(prev).size = new_size;
                    self.insert_free_block(prev);
                } else {
                    self.nodes.get_mut(prev).size = new_size;
                    self.blocks_free_size += missing_alignment;
                }
            } else {
                // New free filler node before `current`.
                let cur_offset = self.nodes.get(current).offset;
                let new_block = self.nodes.alloc(Block {
                    offset: cur_offset,
                    size: missing_alignment,
                    prev_physical: prev,
                    next_physical: current,
                    state: NodeState::Taken {
                        user_data: u32::MAX,
                    },
                    generation: 0,
                    is_margin_filler: false,
                });
                self.nodes.get_mut(current).prev_physical = new_block;
                self.nodes.get_mut(prev).next_physical = new_block;
                self.insert_free_block(new_block);
            }

            let c = self.nodes.get_mut(current);
            c.size -= missing_alignment;
            c.offset += missing_alignment;
        }

        let total = size + debug_margin;
        let current_size = self.nodes.get(current).size;
        if current_size == total {
            if current == self.null_block {
                // The null block was consumed exactly: spawn a fresh empty null block
                // after it.
                let cur = self.nodes.get(current);
                let new_null = self.nodes.alloc(Block {
                    offset: cur.offset + total,
                    size: 0,
                    prev_physical: current,
                    next_physical: NULL_NODE,
                    state: NodeState::Free {
                        prev_free: NULL_NODE,
                        next_free: NULL_NODE,
                    },
                    generation: 0,
                    is_margin_filler: false,
                });
                self.nodes.get_mut(current).next_physical = new_null;
                self.nodes.get_mut(current).state = NodeState::Taken {
                    user_data: u32::MAX,
                };
                self.null_block = new_null;
            }
            // else: exact fit of a non-null free block; it is already removed from the
            // free list and marked taken.
        } else {
            debug_assert!(current_size > total, "found block smaller than request");
            // Split: create trailing free block.
            let cur = *self.nodes.get(current);
            let new_block = self.nodes.alloc(Block {
                offset: cur.offset + total,
                size: current_size - total,
                prev_physical: current,
                next_physical: cur.next_physical,
                state: NodeState::Taken {
                    user_data: u32::MAX,
                },
                generation: 0,
                is_margin_filler: false,
            });
            self.nodes.get_mut(current).next_physical = new_block;
            self.nodes.get_mut(current).size = total;

            if current == self.null_block {
                self.nodes.get_mut(new_block).state = NodeState::Free {
                    prev_free: NULL_NODE,
                    next_free: NULL_NODE,
                };
                self.null_block = new_block;
                self.nodes.get_mut(current).state = NodeState::Taken {
                    user_data: u32::MAX,
                };
            } else {
                let next = self.nodes.get(new_block).next_physical;
                if next != NULL_NODE {
                    self.nodes.get_mut(next).prev_physical = new_block;
                }
                self.insert_free_block(new_block);
            }
        }

        // Store user data on the now-taken current block. Clear any stale
        // margin-filler marker: `current` may itself have been a margin filler that the
        // search selected to satisfy this request, and it is now a real allocation.
        let ud_index = self.store_user_data(user_data);
        let cur = self.nodes.get_mut(current);
        cur.state = NodeState::Taken {
            user_data: ud_index,
        };
        cur.is_margin_filler = false;

        // Debug margin: shrink the allocation and add a debug-margin block after it.
        // As with the alignment filler, VMA's `MarkTaken()`+`InsertFreeBlock()` leaves
        // this node *free* (see the fidelity note above), so we insert it as free. It
        // is flagged `is_margin_filler` so the merge guards keep it attached to its
        // owning allocation until that allocation is freed.
        if debug_margin > 0 {
            let cur = *self.nodes.get(current);
            self.nodes.get_mut(current).size = cur.size - debug_margin;
            let margin_offset = self.nodes.get(current).offset + self.nodes.get(current).size;
            let margin_block = self.nodes.alloc(Block {
                offset: margin_offset,
                size: debug_margin,
                prev_physical: current,
                next_physical: cur.next_physical,
                state: NodeState::Taken {
                    user_data: u32::MAX,
                },
                generation: 0,
                is_margin_filler: true,
            });
            let next = cur.next_physical;
            if next != NULL_NODE {
                self.nodes.get_mut(next).prev_physical = margin_block;
            }
            self.nodes.get_mut(current).next_physical = margin_block;
            self.insert_free_block(margin_block);
        }

        // Register the granularity pages.
        if !self.is_virtual {
            let cur = self.nodes.get(current);
            self.granularity
                .alloc_pages(alloc_type, cur.offset, cur.size);
        }
        self.alloc_count += 1;

        let generation = self.nodes.get(current).generation;
        handle_of(current, generation)
    }

    /// Port of `Free`. `handle` must already have been validated by
    /// [`resolve_handle`](Self::resolve_handle); it names a live (taken) node.
    fn free_impl(&mut self, handle: NodeIndex) {
        let mut block = handle;
        debug_assert!(!self.nodes.get(block).is_free(), "double free");

        // Invalidate any outstanding handle to this allocation immediately: bump the
        // node's generation before it is recycled or re-inserted as free. A second
        // free of the same handle (double free) then fails `resolve_handle`'s
        // generation check rather than corrupting state.
        self.nodes.bump_generation(block);

        // Release the user-data slot.
        let ud_index = self.taken_user_data_index(block);
        let _ = self.take_user_data(ud_index);

        if !self.is_virtual {
            let b = self.nodes.get(block);
            self.granularity.free_pages(b.offset, b.size);
        }
        self.alloc_count -= 1;

        let debug_margin = self.debug_margin;
        let mut next = self.nodes.get(block).next_physical;

        // Merge the trailing debug-margin block first. It is a free block (see
        // alloc_impl), so remove it from its free list before merging. Port of
        // `RemoveFreeBlock(next); MergeBlock(next, block)`.
        if debug_margin > 0 {
            let margin = next;
            self.remove_free_block(margin);
            self.merge_block(margin, block); // margin.prev_physical == block
            block = margin;
            next = self.nodes.get(block).next_physical;
        }

        // Merge with previous physical if it is free (and not a debug-margin filler,
        // which belongs to the allocation *before* it). Test the explicit marker, not
        // `size == debug_margin`.
        let prev = self.nodes.get(block).prev_physical;
        if prev != NULL_NODE {
            let prev_is_free = self.nodes.get(prev).is_free();
            let prev_is_margin = self.nodes.get(prev).is_margin_filler;
            if prev_is_free && !prev_is_margin {
                self.remove_free_block(prev);
                self.merge_block(block, prev);
            }
        }

        // Merge with next.
        if next != NULL_NODE && !self.nodes.get(next).is_free() {
            self.insert_free_block(block);
        } else if next == self.null_block {
            self.merge_block(self.null_block, block);
        } else {
            // next is a free (non-null) block.
            self.remove_free_block(next);
            self.merge_block(next, block);
            self.insert_free_block(next);
        }
    }

    /// Port of `Clear`.
    fn clear_impl(&mut self) {
        self.alloc_count = 0;
        self.blocks_free_count = 0;
        self.blocks_free_size = 0;
        self.is_free_bitmap = 0;

        // Free every node except the null block, then reset the null block.
        let mut block = self.nodes.get(self.null_block).prev_physical;
        self.nodes.get_mut(self.null_block).prev_physical = NULL_NODE;
        while block != NULL_NODE {
            let prev = self.nodes.get(block).prev_physical;
            self.nodes.free(block);
            block = prev;
        }
        let null = self.nodes.get_mut(self.null_block);
        null.offset = 0;
        null.size = self.size;
        null.state = NodeState::Free {
            prev_free: NULL_NODE,
            next_free: NULL_NODE,
        };

        for h in &mut self.free_list {
            *h = NULL_NODE;
        }
        for b in &mut self.inner_is_free_bitmap {
            *b = 0;
        }
        // Reset user data slab.
        self.user_data.clear();
        self.user_data_recycle.clear();
        self.granularity.clear();
    }

    /// Port of `Validate`.
    fn validate_impl(&self) -> Result<(), &'static str> {
        macro_rules! check {
            ($cond:expr, $msg:literal) => {
                if !($cond) {
                    return Err($msg);
                }
            };
        }

        check!(
            self.sum_free_size() <= self.size,
            "sum free size exceeds block size"
        );

        let null = self.nodes.get(self.null_block);
        let mut calculated_size = null.size;
        let mut calculated_free_size = null.size;
        let mut alloc_count = 0usize;
        let mut free_count = 0usize;

        // Free-list integrity.
        for &head in &self.free_list {
            let mut block = head;
            if block != NULL_NODE {
                check!(self.nodes.get(block).is_free(), "free-list head is taken");
                match self.nodes.get(block).state {
                    NodeState::Free { prev_free, .. } => {
                        check!(prev_free == NULL_NODE, "free-list head has a predecessor")
                    }
                    NodeState::Taken { .. } => unreachable!(),
                }
                loop {
                    let next = self.next_free(block);
                    if next == NULL_NODE {
                        break;
                    }
                    check!(
                        self.nodes.get(next).is_free(),
                        "free-list link to taken block"
                    );
                    match self.nodes.get(next).state {
                        NodeState::Free { prev_free, .. } => {
                            check!(prev_free == block, "free-list back-link mismatch")
                        }
                        NodeState::Taken { .. } => unreachable!(),
                    }
                    block = next;
                }
            }
        }

        // Physical chain, from the null block backwards.
        check!(
            null.next_physical == NULL_NODE,
            "null block has a next physical"
        );
        if null.prev_physical != NULL_NODE {
            check!(
                self.nodes.get(null.prev_physical).next_physical == self.null_block,
                "null block prev/next mismatch"
            );
        }

        let mut next_offset = null.offset;
        let mut page_allocs = self.granularity.start_validation();

        let mut prev = null.prev_physical;
        while prev != NULL_NODE {
            let b = self.nodes.get(prev);
            check!(
                b.offset + b.size == next_offset,
                "physical chain offset gap"
            );
            next_offset = b.offset;
            calculated_size += b.size;

            let list_index = self.list_index_from_size(b.size);
            if b.is_free() {
                free_count += 1;
                // Belongs to the correct free list.
                let mut found = false;
                let mut fb = self.free_list[list_index as usize];
                check!(fb != NULL_NODE, "free block's list is empty");
                while fb != NULL_NODE {
                    if fb == prev {
                        found = true;
                        break;
                    }
                    fb = self.next_free(fb);
                }
                check!(found, "free block not found on its list");
                calculated_free_size += b.size;
            } else {
                alloc_count += 1;
                // Not on any free list.
                let mut fb = self.free_list[list_index as usize];
                while fb != NULL_NODE {
                    check!(fb != prev, "taken block found on a free list");
                    fb = self.next_free(fb);
                }
                if !self.is_virtual {
                    self.granularity
                        .validate(&mut page_allocs, b.offset, b.size)?;
                }
            }

            if b.prev_physical != NULL_NODE {
                check!(
                    self.nodes.get(b.prev_physical).next_physical == prev,
                    "physical back-link mismatch"
                );
            }
            prev = b.prev_physical;
        }

        if !self.is_virtual {
            self.granularity.finish_validation(&page_allocs)?;
        }

        check!(next_offset == 0, "physical chain does not start at 0");
        check!(calculated_size == self.size, "physical chain size mismatch");
        check!(
            calculated_free_size == self.sum_free_size(),
            "free size accounting mismatch"
        );
        check!(alloc_count == self.alloc_count, "alloc count mismatch");
        check!(free_count == self.blocks_free_count, "free count mismatch");

        // Bitmap consistency: for every (memory class, second index) list, the
        // inner bitmap bit must be set iff that list's head is non-empty; and each
        // top-level class bit must be set iff any of its inner bits are set.
        //
        // The number of valid second indices differs for the small-buffer class 0
        // (4 real / 32 virtual buckets) versus higher classes (always 32); a class-0
        // loop over the full 32 would alias higher lists via `list_index_from_class`.
        let class0_second_count = if self.is_virtual {
            SECOND_LEVEL_COUNT
        } else {
            4
        };
        for mem_class in 0..self.memory_classes {
            let inner = self.inner_is_free_bitmap[mem_class as usize];
            let second_count = if mem_class == 0 {
                class0_second_count
            } else {
                SECOND_LEVEL_COUNT
            };
            for second_index in 0..second_count {
                let list_index = self.list_index_from_class(mem_class, second_index);
                let bit_set = (inner & (1u32 << second_index)) != 0;
                if (list_index as usize) >= self.free_list.len() {
                    // This (class, second index) has no backing list (the highest
                    // memory class is only partially populated). Its bit must be clear.
                    check!(!bit_set, "TLSF inner bitmap bit set for a nonexistent list");
                    continue;
                }
                let head_nonempty = self.free_list[list_index as usize] != NULL_NODE;
                check!(
                    head_nonempty == bit_set,
                    "TLSF inner bitmap bit disagrees with free-list head"
                );
            }
            let top_bit_set = (self.is_free_bitmap & (1u64 << mem_class)) != 0;
            check!(
                top_bit_set == (inner != 0),
                "TLSF top-level bitmap bit disagrees with inner bitmap"
            );
        }

        let _ = &mut page_allocs;
        Ok(())
    }
}
