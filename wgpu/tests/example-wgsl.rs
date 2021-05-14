use naga::{front::wgsl, valid::Validator};
use std::{fs, path::PathBuf};

#[test]
fn parse_example_wgsl() {
    let read_dir = match PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .read_dir()
    {
        Ok(iter) => iter,
        Err(e) => {
            log::error!("Unable to open the examples folder: {:?}", e);
            return;
        }
    };
    for example_entry in read_dir {
        let read_files = match example_entry {
            Ok(dir_entry) => match dir_entry.path().read_dir() {
                Ok(iter) => iter,
                Err(_) => continue,
            },
            Err(e) => {
                log::warn!("Skipping example: {:?}", e);
                continue;
            }
        };
        for file_entry in read_files {
            let shader = match file_entry {
                Ok(entry) => match entry.path().extension() {
                    Some(ostr) if &*ostr == "wgsl" => {
                        println!("Validating {:?}", entry.path());
                        fs::read_to_string(entry.path()).unwrap_or_default()
                    }
                    _ => continue,
                },
                Err(e) => {
                    log::warn!("Skipping file: {:?}", e);
                    continue;
                }
            };

            let module = wgsl::parse_str(&shader).unwrap();
            //TODO: re-use the validator
            Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
