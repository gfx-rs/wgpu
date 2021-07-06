use bit_set::BitSet;
use std::fmt;

const HEAP_SIZE_FIXED: usize = 64;

#[derive(Copy, Clone)]
pub(super) struct DualHandle {
    cpu: native::CpuDescriptor,
    gpu: native::GpuDescriptor,
    /// How large the block allocated to this handle is.
    size: u64,
}

type DescriptorIndex = u64;

struct LinearHeap {
    raw: native::DescriptorHeap,
    handle_size: u64,
    total_handles: u64,
    start: DualHandle,
}

impl LinearHeap {
    pub(super) fn at(&self, index: DescriptorIndex, size: u64) -> DualHandle {
        assert!(index < self.total_handles);
        DualHandle {
            cpu: self.cpu_descriptor_at(index),
            gpu: self.gpu_descriptor_at(index),
            size,
        }
    }

    pub(super) fn cpu_descriptor_at(&self, index: u64) -> native::CpuDescriptor {
        native::CpuDescriptor {
            ptr: self.start.cpu.ptr + (self.handle_size * index) as usize,
        }
    }

    pub(super) fn gpu_descriptor_at(&self, index: u64) -> native::GpuDescriptor {
        native::GpuDescriptor {
            ptr: self.start.gpu.ptr + self.handle_size * index,
        }
    }
}

/// Fixed-size free-list allocator for CPU descriptors.
struct FixedSizeHeap {
    raw: native::DescriptorHeap,
    /// Bit flag representation of available handles in the heap.
    ///
    ///  0 - Occupied
    ///  1 - free
    availability: u64,
    handle_size: usize,
    start: native::CpuDescriptor,
}

impl FixedSizeHeap {
    fn new(device: native::Device, ty: native::DescriptorHeapType) -> Self {
        let (heap, _hr) = device.create_descriptor_heap(
            HEAP_SIZE_FIXED as _,
            ty,
            native::DescriptorHeapFlags::empty(),
            0,
        );

        Self {
            handle_size: device.get_descriptor_increment_size(ty) as _,
            availability: !0, // all free!
            start: heap.start_cpu_descriptor(),
            raw: heap,
        }
    }

    fn alloc_handle(&mut self) -> native::CpuDescriptor {
        // Find first free slot.
        let slot = self.availability.trailing_zeros() as usize;
        assert!(slot < HEAP_SIZE_FIXED);
        // Set the slot as occupied.
        self.availability ^= 1 << slot;

        native::CpuDescriptor {
            ptr: self.start.ptr + self.handle_size * slot,
        }
    }

    fn free_handle(&mut self, handle: native::CpuDescriptor) {
        let slot = (handle.ptr - self.start.ptr) / self.handle_size;
        assert!(slot < HEAP_SIZE_FIXED);
        assert_eq!(self.availability & (1 << slot), 0);
        self.availability ^= 1 << slot;
    }

    fn is_full(&self) -> bool {
        self.availability == 0
    }

    unsafe fn destroy(&self) {
        self.raw.destroy();
    }
}

#[derive(Clone, Copy)]
pub(super) struct Handle {
    pub raw: native::CpuDescriptor,
    heap_index: usize,
}

impl fmt::Debug for Handle {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Handle")
            .field("ptr", &self.raw.ptr)
            .field("heap_index", &self.heap_index)
            .finish()
    }
}

pub(super) struct CpuPool {
    device: native::Device,
    ty: native::DescriptorHeapType,
    heaps: Vec<FixedSizeHeap>,
    avaliable_heap_indices: BitSet,
}

impl CpuPool {
    pub(super) fn new(device: native::Device, ty: native::DescriptorHeapType) -> Self {
        Self {
            device,
            ty,
            heaps: Vec::new(),
            avaliable_heap_indices: BitSet::new(),
        }
    }

    pub(super) fn alloc_handle(&mut self) -> Handle {
        let heap_index = self
            .avaliable_heap_indices
            .iter()
            .next()
            .unwrap_or_else(|| {
                // Allocate a new heap
                let id = self.heaps.len();
                self.heaps.push(FixedSizeHeap::new(self.device, self.ty));
                self.avaliable_heap_indices.insert(id);
                id
            });

        let heap = &mut self.heaps[heap_index];
        let handle = Handle {
            raw: heap.alloc_handle(),
            heap_index,
        };
        if heap.is_full() {
            self.avaliable_heap_indices.remove(heap_index);
        }

        handle
    }

    pub(super) fn free_handle(&mut self, handle: Handle) {
        self.heaps[handle.heap_index].free_handle(handle.raw);
        self.avaliable_heap_indices.insert(handle.heap_index);
    }

    pub(super) unsafe fn destroy(&self) {
        for heap in &self.heaps {
            heap.destroy();
        }
    }
}
