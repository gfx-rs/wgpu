/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[derive(Debug)]
pub struct Trace {
    path: std::path::PathBuf,
    file: std::fs::File,
}

impl Trace {
    pub fn new(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let mut file = std::fs::File::create(path.join("trace.ron"))?;
        std::io::Write::write(&mut file, b"[\n")?;
        Ok(Trace {
            path: path.to_path_buf(),
            file,
        })
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        let _ = std::io::Write::write(&mut self.file, b"]");
    }
}
