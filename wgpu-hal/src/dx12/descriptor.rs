use crate::auxil::dxgi::result::HResult as _;
use bit_set::BitSet;
use parking_lot::Mutex;
use range_alloc::RangeAllocator;
use std::fmt;

const HEAP_SIZE_FIXED: usize = 64;

#[derive(Copy, Clone)]
pub(super) struct DualHandle {
    cpu: d3d12::CpuDescriptor,
    pub gpu: d3d12::GpuDescriptor,
    /// How large the block allocated to this handle is.
    count: u64,
}

impl fmt::Debug for DualHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DualHandle")
            .field("cpu", &self.cpu.ptr)
            .field("gpu", &self.gpu.ptr)
            .field("count", &self.count)
            .finish()
    }
}

type DescriptorIndex = u64;

pub(super) struct GeneralHeap {
    pub raw: d3d12::DescriptorHeap,
    ty: d3d12::DescriptorHeapType,
    handle_size: u64,
    total_handles: u64,
    start: DualHandle,
    ranges: Mutex<RangeAllocator<DescriptorIndex>>,
}

impl GeneralHeap {
    pub(super) fn new(
        device: d3d12::Device,
        ty: d3d12::DescriptorHeapType,
        total_handles: u64,
    ) -> Result<Self, crate::DeviceError> {
        let raw = {
            profiling::scope!("ID3D12Device::CreateDescriptorHeap");
            device
                .create_descriptor_heap(
                    total_handles as u32,
                    ty,
                    d3d12::DescriptorHeapFlags::SHADER_VISIBLE,
                    0,
                )
                .into_device_result("Descriptor heap creation")?
        };

        Ok(Self {
            raw: raw.clone(),
            ty,
            handle_size: device.get_descriptor_increment_size(ty) as u64,
            total_handles,
            start: DualHandle {
                cpu: raw.start_cpu_descriptor(),
                gpu: raw.start_gpu_descriptor(),
                count: 0,
            },
            ranges: Mutex::new(RangeAllocator::new(0..total_handles)),
        })
    }

    pub(super) fn at(&self, index: DescriptorIndex, count: u64) -> DualHandle {
        assert!(index < self.total_handles);
        DualHandle {
            cpu: self.cpu_descriptor_at(index),
            gpu: self.gpu_descriptor_at(index),
            count,
        }
    }

    fn cpu_descriptor_at(&self, index: u64) -> d3d12::CpuDescriptor {
        d3d12::CpuDescriptor {
            ptr: self.start.cpu.ptr + (self.handle_size * index) as usize,
        }
    }

    fn gpu_descriptor_at(&self, index: u64) -> d3d12::GpuDescriptor {
        d3d12::GpuDescriptor {
            ptr: self.start.gpu.ptr + self.handle_size * index,
        }
    }

    pub(super) fn allocate_slice(&self, count: u64) -> Result<DescriptorIndex, crate::DeviceError> {
        let range = self.ranges.lock().allocate_range(count).map_err(|err| {
            log::error!("Unable to allocate descriptors: {:?}", err);
            crate::DeviceError::OutOfMemory
        })?;
        Ok(range.start)
    }

    /// Free handles previously given out by this `DescriptorHeapSlice`.
    /// Do not use this with handles not given out by this `DescriptorHeapSlice`.
    pub(crate) fn free_slice(&self, handle: DualHandle) {
        let start = (handle.gpu.ptr - self.start.gpu.ptr) / self.handle_size;
        self.ranges.lock().free_range(start..start + handle.count);
    }
}

/// Fixed-size free-list allocator for CPU descriptors.
struct FixedSizeHeap {
    _raw: d3d12::DescriptorHeap,
    /// Bit flag representation of available handles in the heap.
    ///
    ///  0 - Occupied
    ///  1 - free
    availability: u64,
    handle_size: usize,
    start: d3d12::CpuDescriptor,
}

impl FixedSizeHeap {
    fn new(device: &d3d12::Device, ty: d3d12::DescriptorHeapType) -> Self {
        let (heap, _hr) = device.create_descriptor_heap(
            HEAP_SIZE_FIXED as _,
            ty,
            d3d12::DescriptorHeapFlags::empty(),
            0,
        );

        Self {
            handle_size: device.get_descriptor_increment_size(ty) as _,
            availability: !0, // all free!
            start: heap.start_cpu_descriptor(),
            _raw: heap,
        }
    }

    fn alloc_handle(&mut self) -> d3d12::CpuDescriptor {
        // Find first free slot.
        let slot = self.availability.trailing_zeros() as usize;
        assert!(slot < HEAP_SIZE_FIXED);
        // Set the slot as occupied.
        self.availability ^= 1 << slot;

        d3d12::CpuDescriptor {
            ptr: self.start.ptr + self.handle_size * slot,
        }
    }

    fn free_handle(&mut self, handle: d3d12::CpuDescriptor) {
        let slot = (handle.ptr - self.start.ptr) / self.handle_size;
        assert!(slot < HEAP_SIZE_FIXED);
        assert_eq!(self.availability & (1 << slot), 0);
        self.availability ^= 1 << slot;
    }

    fn is_full(&self) -> bool {
        self.availability == 0
    }
}

#[derive(Clone, Copy)]
pub(super) struct Handle {
    pub raw: d3d12::CpuDescriptor,
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
    device: d3d12::Device,
    ty: d3d12::DescriptorHeapType,
    heaps: Vec<FixedSizeHeap>,
    avaliable_heap_indices: BitSet,
}

impl CpuPool {
    pub(super) fn new(device: d3d12::Device, ty: d3d12::DescriptorHeapType) -> Self {
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
                self.heaps.push(FixedSizeHeap::new(&self.device, self.ty));
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
}

pub(super) struct CpuHeapInner {
    pub _raw: d3d12::DescriptorHeap,
    pub stage: Vec<d3d12::CpuDescriptor>,
}

pub(super) struct CpuHeap {
    pub inner: Mutex<CpuHeapInner>,
    start: d3d12::CpuDescriptor,
    handle_size: u32,
    total: u32,
}

unsafe impl Send for CpuHeap {}
unsafe impl Sync for CpuHeap {}

impl CpuHeap {
    pub(super) fn new(
        device: d3d12::Device,
        ty: d3d12::DescriptorHeapType,
        total: u32,
    ) -> Result<Self, crate::DeviceError> {
        let handle_size = device.get_descriptor_increment_size(ty);
        let raw = device
            .create_descriptor_heap(total, ty, d3d12::DescriptorHeapFlags::empty(), 0)
            .into_device_result("CPU descriptor heap creation")?;

        Ok(Self {
            inner: Mutex::new(CpuHeapInner {
                _raw: raw.clone(),
                stage: Vec::new(),
            }),
            start: raw.start_cpu_descriptor(),
            handle_size,
            total,
        })
    }

    pub(super) fn at(&self, index: u32) -> d3d12::CpuDescriptor {
        d3d12::CpuDescriptor {
            ptr: self.start.ptr + (self.handle_size * index) as usize,
        }
    }
}

impl fmt::Debug for CpuHeap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuHeap")
            .field("start", &self.start.ptr)
            .field("handle_size", &self.handle_size)
            .field("total", &self.total)
            .finish()
    }
}

pub(super) unsafe fn upload(
    device: d3d12::Device,
    src: &CpuHeapInner,
    dst: &GeneralHeap,
    dummy_copy_counts: &[u32],
) -> Result<DualHandle, crate::DeviceError> {
    let count = src.stage.len() as u32;
    let index = dst.allocate_slice(count as u64)?;
    unsafe {
        device.CopyDescriptors(
            1,
            &dst.cpu_descriptor_at(index),
            &count,
            count,
            src.stage.as_ptr(),
            dummy_copy_counts.as_ptr(),
            dst.ty as u32,
        )
    };
    Ok(dst.at(index, count as u64))
}
