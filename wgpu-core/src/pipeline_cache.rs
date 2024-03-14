use std::io::Write;

use wgt::AdapterInfo;

const MAGIC: [u8; 8] = *b"WGPUPLCH";
const HEADER_VERSION: u32 = 1;
const ABI: u32 = std::mem::size_of::<*const ()>() as u32;

#[repr(C)]
struct PipelineCacheHeader {
    /// The magic header to ensure that we have the right file format
    /// Has a value of MAGIC, as above
    magic: [u8; 8],
    // /// The total size of this header, in bytes
    // header_size: u32,
    /// The version of this wgpu header
    /// Should be equal to HEADER_VERSION above
    ///
    /// This must always be the second item, after the value above
    header_version: u32,
    /// The number of bytes in the pointers of this ABI, because some drivers
    /// have previously not distinguished between their 32 bit and 64 bit drivers
    /// leading to Vulkan data corruption
    cache_abi: u32,
    /// The id for the backend in use, from [wgt::Backend]
    backend: u8,
    /// The hash key which identifiers the device/adapter.
    /// This is used to validate that this pipeline cache (probably) was produced for
    /// the expected device.
    /// On Vulkan: it is a combination of vendor ID and device ID
    adapter_key: [u8; 15],
    /// A key used to validate that this device is still compatible with the cache
    ///
    /// This should e.g. contain driver version and/or intermediate compiler versions
    device_key: [u8; 16],
    /// The length of the data which is sent to/recieved from the backend
    data_size: u64,
    /// The hash of the data which is sent to/recieved from the backend, and which
    /// follows this header. That should be the remainder of the memory
    data_hash: u64,
}

pub enum PipelineCacheValidationError {
    Truncated,
    Corrupted,
    Outdated,
    WrongDevice,
    Unsupported,
}

impl PipelineCacheValidationError {
    /// Could the error have been avoided?
    pub fn was_avoidable(&self) -> bool {
        match self {
            PipelineCacheValidationError::WrongDevice
            | PipelineCacheValidationError::Unsupported => true,
            PipelineCacheValidationError::Truncated
            | PipelineCacheValidationError::Outdated
            | PipelineCacheValidationError::Corrupted => false,
        }
    }
}

/// Validate the data in a pipeline cache
pub fn validate_pipeline_cache<'d>(
    cache_data: &'d [u8],
    adapter: &AdapterInfo,
    device_key: [u8; 16],
) -> Result<&'d [u8], PipelineCacheValidationError> {
    let adapter_key = adapter_key(adapter)?;
    let Some((header, remaining_data)) = read_header(cache_data) else {
        return Err(PipelineCacheValidationError::Truncated);
    };
    if header.magic != MAGIC {
        return Err(PipelineCacheValidationError::Corrupted);
    }
    if header.header_version != HEADER_VERSION {
        return Err(PipelineCacheValidationError::Outdated);
    }
    if header.cache_abi != ABI {
        return Err(PipelineCacheValidationError::Outdated);
    }
    if header.backend != adapter.backend as u8 {
        return Err(PipelineCacheValidationError::WrongDevice);
    }
    if header.adapter_key != adapter_key {
        return Err(PipelineCacheValidationError::WrongDevice);
    }
    if header.device_key != device_key {
        return Err(PipelineCacheValidationError::WrongDevice);
    }
    let data_size: usize = header
        .data_size
        .try_into()
        // If the data was previously more than 4GiB, and we're on the same size of system (ABI, above)l
        // Then the data must be corrupted
        .map_err(|_| PipelineCacheValidationError::Corrupted)?;
    if data_size != remaining_data.len() {
        return Err(PipelineCacheValidationError::WrongDevice);
    }
    Ok(remaining_data)
}

fn adapter_key(adapter: &AdapterInfo) -> Result<[u8; 15], PipelineCacheValidationError> {
    match adapter.backend {
        wgt::Backend::Vulkan => {
            // If these change size, the header format needs to change
            // We set the type explicitly so this won't compile in that case
            let v: [u8; 4] = adapter.vendor.to_be_bytes();
            let d: [u8; 4] = adapter.device.to_be_bytes();
            let adapter = [
                255, 255, 255, v[0], v[1], v[2], v[3], d[0], d[1], d[2], d[3], 255, 255, 255, 255,
            ];
            Ok(adapter)
        }
        _ => Err(PipelineCacheValidationError::Unsupported),
    }
}

pub fn write_pipeline_cache(writer: &mut dyn Write, data: &[u8], adpater: &AdapterInfo) {}

fn read_header(data: &[u8]) -> Option<(PipelineCacheHeader, &[u8])> {
    let mut reader = Reader {
        data,
        total_read: 0,
    };
    let magic = reader.read_array()?;
    let header_version = reader.read_u32()?;
    let cache_abi = reader.read_u32()?;
    let backend = reader.read_byte()?;
    let adapter_key = reader.read_array()?;
    let device_key = reader.read_array()?;
    let data_size = reader.read_u64()?;
    let data_hash = reader.read_u64()?;

    assert_eq!(
        reader.total_read,
        std::mem::size_of::<PipelineCacheHeader>()
    );

    Some((
        PipelineCacheHeader {
            magic,
            header_version,
            cache_abi,
            backend,
            adapter_key,
            device_key,
            data_size,
            data_hash,
        },
        reader.data,
    ))
}

fn write_header(header: PipelineCacheHeader, writer: &mut dyn Write) {}

struct Reader<'a> {
    data: &'a [u8],
    total_read: usize,
}

impl<'a> Reader<'a> {
    fn read_byte(&mut self) -> Option<u8> {
        let res = *self.data.get(0)?;
        self.total_read += 1;
        self.data = &self.data[1..];
        Some(res)
    }
    fn read_array<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (start, data) = self.data.split_at(N);
        self.total_read += N;
        self.data = data;
        start.try_into().ok()
    }

    fn read_u16(&mut self) -> Option<u16> {
        self.read_array().map(u16::from_be_bytes)
    }
    fn read_u32(&mut self) -> Option<u32> {
        self.read_array().map(u32::from_be_bytes)
    }
    fn read_u64(&mut self) -> Option<u64> {
        self.read_array().map(u64::from_be_bytes)
    }
}
