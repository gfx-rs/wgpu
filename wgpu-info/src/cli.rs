use std::{io, process::exit};

const HELP: &str = "\
Usage: wgpu-info [--output <PATH>] [--json]

Options:
  -h, --help          Print this help
  -o, --output <PATH> Destination to write output to
  -j, --json          Output JSON information instead of text
";

pub fn main() {
    let mut args = pico_args::Arguments::from_env();
    let help = args.contains(["-h", "--help"]);

    if help {
        eprintln!("{HELP}");
        exit(101);
    }

    let output_path: Option<String> = args.opt_value_from_str(["-o", "--output"]).unwrap();
    let json = args.contains(["-j", "--json"]);

    env_logger::init();

    let mut file_handle;
    let mut std_handle;
    let output: &mut dyn io::Write = match output_path {
        Some(path) => {
            file_handle = std::fs::File::create(path).unwrap();
            &mut file_handle
        }
        None => {
            std_handle = io::stdout();
            &mut std_handle
        }
    };

    if !json {
        crate::human::print_adapters(output)
    }
}
