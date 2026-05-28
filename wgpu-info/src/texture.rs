use exhaust::Exhaust;

#[test]
fn test_compute_render_extent() {
    for format in wgpu::TextureFormat::exhaust() {
        let desc = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1280,
                height: 720,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::empty(),
            view_formats: &[],
        };

        if format.is_multi_planar_format() {
            let _ = desc.compute_render_extent(0, Some(0));
        } else {
            let _ = desc.compute_render_extent(0, None);
        }
    }

    for format in [wgpu::TextureFormat::NV12, wgpu::TextureFormat::P010] {
        let desc = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 8,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 2,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::empty(),
            view_formats: &[],
        };

        assert_eq!(
            desc.compute_render_extent(0, Some(0)),
            wgpu::Extent3d {
                width: 8,
                height: 4,
                depth_or_array_layers: 1,
            }
        );
        assert_eq!(
            desc.compute_render_extent(0, Some(1)),
            wgpu::Extent3d {
                width: 4,
                height: 2,
                depth_or_array_layers: 1,
            }
        );
        assert_eq!(
            desc.compute_render_extent(1, Some(0)),
            wgpu::Extent3d {
                width: 4,
                height: 2,
                depth_or_array_layers: 1,
            }
        );
        assert_eq!(
            desc.compute_render_extent(1, Some(1)),
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            }
        );
    }
}

pub fn max_texture_format_string_size() -> usize {
    wgpu::TextureFormat::exhaust()
        .map(|f| texture_format_name(f).len())
        .max()
        .unwrap()
}

pub fn texture_format_name(format: wgpu::TextureFormat) -> String {
    match format {
        wgpu::TextureFormat::Astc { block, channel } => {
            format!("Astc{block:?}{channel:?}:")
        }
        _ => {
            format!("{format:?}:")
        }
    }
}
