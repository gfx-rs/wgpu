use std::fmt;

#[derive(Debug)]
pub enum Error {
    Unsupported,
    Error(Box<dyn std::error::Error>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported => write!(f, "Battery status is unsupported on this platform"),
            Error::Error(err) => write!(f, "Battery status retrieval failed: {}", err),
        }
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "windows",
    target_os = "dragonfly",
    target_os = "freebsd"
))]
mod platform {
    use super::Error;
    use battery::{self, Manager, State};

    impl From<battery::errors::Error> for Error {
        fn from(err: battery::errors::Error) -> Error {
            // Box the error so that the battery::errors::Error type does
            // not leak out of this module.
            Error::Error(Box::new(err))
        }
    }

    pub fn is_battery_discharging() -> Result<bool, Error> {
        let manager = Manager::new()?;
        for battery in manager.batteries()? {
            if battery?.state() == State::Discharging {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "windows",
    target_os = "dragonfly",
    target_os = "freebsd"
)))]
mod platform {
    use super::Error;

    pub fn is_battery_discharging() -> Result<bool, Error> {
        Err(Error::Unsupported)
    }
}

pub use platform::is_battery_discharging;
