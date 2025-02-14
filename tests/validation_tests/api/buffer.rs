//! Tests of [`wgpu::Buffer`] and related.

mod buffer_slice {
    #[test]
    fn reslice_success() {
        let (device, _queue) = crate::request_noop_device();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 100,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        assert_eq!(buffer.slice(10..90).slice(10..70), buffer.slice(20..80));
    }

    #[test]
    #[should_panic = "slice offset 10 size 80 is out of range for buffer of size 80"]
    fn reslice_out_of_bounds() {
        let (device, _queue) = crate::request_noop_device();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 100,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        buffer.slice(10..90).slice(10..90);
    }
}
