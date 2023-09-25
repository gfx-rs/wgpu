//! GPU Resource

use crate::{com::ComPtr, D3DResult, Rect};
use std::{ops::Range, ptr};
use winapi::um::d3d12;

pub type Subresource = u32;

pub struct DiscardRegion<'a> {
    pub rects: &'a [Rect],
    pub subregions: Range<Subresource>,
}

pub type Resource = ComPtr<d3d12::ID3D12Resource>;

impl Resource {
    ///
    pub fn map(
        &self,
        subresource: Subresource,
        read_range: Option<Range<usize>>,
    ) -> D3DResult<*mut ()> {
        let mut ptr = ptr::null_mut();
        let read_range = read_range.map(|r| d3d12::D3D12_RANGE {
            Begin: r.start,
            End: r.end,
        });
        let read = match read_range {
            Some(ref r) => r as *const _,
            None => ptr::null(),
        };
        let hr = unsafe { self.Map(subresource, read, &mut ptr) };

        (ptr as _, hr)
    }

    pub fn unmap(&self, subresource: Subresource, write_range: Option<Range<usize>>) {
        let write_range = write_range.map(|r| d3d12::D3D12_RANGE {
            Begin: r.start,
            End: r.end,
        });
        let write = match write_range {
            Some(ref r) => r as *const _,
            None => ptr::null(),
        };

        unsafe { self.Unmap(subresource, write) };
    }

    pub fn gpu_virtual_address(&self) -> u64 {
        unsafe { self.GetGPUVirtualAddress() }
    }
}
