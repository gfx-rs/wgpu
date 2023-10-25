use std::{error::Error, fmt::Display, fs, io, path::Path};

use anyhow::{anyhow, ensure};
use nanoserde::{self, DeRon, DeRonErr, SerRon};

#[derive(Debug)]
struct BadRonParse(BadRonParseKind);

impl Display for BadRonParse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed to read RON configuration of HLSL snapshot test")
    }
}

impl Error for BadRonParse {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.0)
    }
}

#[derive(Debug)]
enum BadRonParseKind {
    Read { source: io::Error },
    Parse { source: DeRonErr },
    Empty,
}

impl Display for BadRonParseKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BadRonParseKind::Read { source } => Display::fmt(source, f),
            BadRonParseKind::Parse { source } => Display::fmt(source, f),
            BadRonParseKind::Empty => write!(f, "no configuration was specified"),
        }
    }
}

impl Error for BadRonParseKind {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            BadRonParseKind::Read { source } => source.source(),
            BadRonParseKind::Parse { source } => source.source(),
            BadRonParseKind::Empty => None,
        }
    }
}

#[derive(Debug, DeRon, SerRon)]
pub struct Config {
    pub vertex: Vec<ConfigItem>,
    pub fragment: Vec<ConfigItem>,
    pub compute: Vec<ConfigItem>,
}

impl Config {
    pub fn empty() -> Self {
        Self {
            vertex: Default::default(),
            fragment: Default::default(),
            compute: Default::default(),
        }
    }

    pub fn from_path(path: impl AsRef<Path>) -> anyhow::Result<Config> {
        let path = path.as_ref();
        let raw_config = fs::read_to_string(path)
            .map_err(|source| BadRonParse(BadRonParseKind::Read { source }))?;
        let config = Config::deserialize_ron(&raw_config)
            .map_err(|source| BadRonParse(BadRonParseKind::Parse { source }))?;
        ensure!(!config.is_empty(), BadRonParse(BadRonParseKind::Empty));
        Ok(config)
    }

    pub fn to_file(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        let mut s = self.serialize_ron();
        s.push('\n');
        fs::write(path, &s).map_err(|e| anyhow!("failed to write to {}: {e}", path.display()))
    }

    pub fn is_empty(&self) -> bool {
        let Self {
            vertex,
            fragment,
            compute,
        } = self;
        vertex.is_empty() && fragment.is_empty() && compute.is_empty()
    }
}

#[derive(Debug, DeRon, SerRon)]
pub struct ConfigItem {
    pub entry_point: String,
    /// See also
    /// <https://learn.microsoft.com/en-us/windows/win32/direct3dtools/dx-graphics-tools-fxc-using>.
    pub target_profile: String,
}
