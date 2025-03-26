use std::num::NonZero;

const ARBITRARY_DESC: &wgpu::BufferDescriptor = &wgpu::BufferDescriptor {
    label: None,
    size: 100,
    usage: wgpu::BufferUsages::VERTEX,
    mapped_at_creation: false,
};

#[test]
fn reslice_success() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = device.create_buffer(ARBITRARY_DESC);

    assert_eq!(buffer.slice(10..90).slice(10..70), buffer.slice(20..80));
}

#[test]
#[should_panic = "slice offset 10 size 80 is out of range for buffer of size 80"]
fn reslice_out_of_bounds() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = device.create_buffer(ARBITRARY_DESC);

    buffer.slice(10..90).slice(10..90);
}

#[test]
fn getters() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = device.create_buffer(ARBITRARY_DESC);

    let slice_with_size = buffer.slice(10..90);
    assert_eq!(
        (
            slice_with_size.buffer(),
            slice_with_size.offset(),
            slice_with_size.size()
        ),
        (&buffer, 10, NonZero::new(80).unwrap())
    );

    let slice_without_size = buffer.slice(10..);
    assert_eq!(
        (
            slice_without_size.buffer(),
            slice_without_size.offset(),
            slice_without_size.size()
        ),
        (&buffer, 10, NonZero::new(90).unwrap())
    );
}

#[test]
fn into_buffer_binding() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = device.create_buffer(ARBITRARY_DESC);

    // BindingResource doesn’t implement PartialEq, so use matching
    let wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: b,
        offset: 50,
        size: Some(size),
    }) = wgpu::BindingResource::from(buffer.slice(50..80))
    else {
        panic!("didn't match")
    };
    assert_eq!(b, &buffer);
    assert_eq!(size, NonZero::new(30).unwrap());
}
