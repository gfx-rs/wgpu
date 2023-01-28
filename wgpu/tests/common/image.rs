use std::{
    ffi::{OsStr, OsString},
    io,
    path::Path,
    str::FromStr,
};

fn read_png(path: impl AsRef<Path>, width: u32, height: u32) -> Option<Vec<u8>> {
    let data = match std::fs::read(&path) {
        Ok(f) => f,
        Err(e) => {
            log::warn!(
                "image comparison invalid: file io error when comparing {}: {}",
                path.as_ref().display(),
                e
            );
            return None;
        }
    };
    let decoder = png::Decoder::new(io::Cursor::new(data));
    let mut reader = decoder.read_info().ok()?;

    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer).ok()?;
    if info.width != width {
        log::warn!("image comparison invalid: size mismatch");
        return None;
    }
    if info.height != height {
        log::warn!("image comparison invalid: size mismatch");
        return None;
    }
    if info.color_type != png::ColorType::Rgba {
        log::warn!("image comparison invalid: color type mismatch");
        return None;
    }
    if info.bit_depth != png::BitDepth::Eight {
        log::warn!("image comparison invalid: bit depth mismatch");
        return None;
    }

    Some(buffer)
}

#[allow(unused_variables)]
fn write_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    data: &[u8],
    compression: png::Compression,
) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let file = io::BufWriter::new(std::fs::File::create(path).unwrap());

        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(compression);
        let mut writer = encoder.write_header().unwrap();

        writer.write_image_data(data).unwrap();
    }
}

pub fn calc_difference(lhs: u8, rhs: u8) -> u8 {
    (lhs as i16 - rhs as i16).unsigned_abs() as u8
}

pub fn compare_image_output(
    path: impl AsRef<Path> + AsRef<OsStr>,
    width: u32,
    height: u32,
    data: &[u8],
    tolerance: u8,
    max_outliers: usize,
) {
    let comparison_data = read_png(&path, width, height);

    if let Some(cmp) = comparison_data {
        assert_eq!(cmp.len(), data.len());

        let difference_data: Vec<_> = cmp
            .chunks_exact(4)
            .zip(data.chunks_exact(4))
            .flat_map(|(cmp_chunk, data_chunk)| {
                [
                    calc_difference(cmp_chunk[0], data_chunk[0]),
                    calc_difference(cmp_chunk[1], data_chunk[1]),
                    calc_difference(cmp_chunk[2], data_chunk[2]),
                    255,
                ]
            })
            .collect();

        let outliers: usize = difference_data
            .chunks_exact(4)
            .map(|colors| {
                (colors[0] > tolerance) as usize
                    + (colors[1] > tolerance) as usize
                    + (colors[2] > tolerance) as usize
            })
            .sum();

        let max_difference = difference_data
            .chunks_exact(4)
            .map(|colors| colors[0].max(colors[1]).max(colors[2]))
            .max()
            .unwrap();

        if outliers > max_outliers {
            // Because the data is mismatched, lets output the difference to a file.
            let old_path = Path::new(&path);
            let actual_path = Path::new(&path).with_file_name(
                OsString::from_str(
                    &(old_path.file_stem().unwrap().to_string_lossy() + "-actual.png"),
                )
                .unwrap(),
            );
            let difference_path = Path::new(&path).with_file_name(
                OsString::from_str(
                    &(old_path.file_stem().unwrap().to_string_lossy() + "-difference.png"),
                )
                .unwrap(),
            );

            write_png(actual_path, width, height, data, png::Compression::Fast);
            write_png(
                difference_path,
                width,
                height,
                &difference_data,
                png::Compression::Fast,
            );

            panic!(
                "Image data mismatch! Outlier count {outliers} over limit {max_outliers}. Max difference {max_difference}"
            )
        } else {
            println!("{outliers} outliers over max difference {max_difference}");
        }
    } else {
        write_png(&path, width, height, data, png::Compression::Best);
    }
}
