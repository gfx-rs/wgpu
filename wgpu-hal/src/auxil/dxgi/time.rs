#![allow(dead_code)] // IPresentationManager is unused currently

use std::mem;

use winapi::um::{
    profileapi::{QueryPerformanceCounter, QueryPerformanceFrequency},
    winnt::LARGE_INTEGER,
};

pub enum PresentationTimer {
    /// DXGI uses QueryPerformanceCounter
    Dxgi {
        /// How many ticks of QPC per second
        frequency: u64,
    },
    /// IPresentationManager uses QueryInterruptTimePrecise
    #[allow(non_snake_case)]
    IPresentationManager {
        fnQueryInterruptTimePrecise: unsafe extern "system" fn(*mut winapi::ctypes::c_ulonglong),
    },
}

impl std::fmt::Debug for PresentationTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Dxgi { frequency } => f
                .debug_struct("DXGI")
                .field("frequency", &frequency)
                .finish(),
            Self::IPresentationManager {
                fnQueryInterruptTimePrecise,
            } => f
                .debug_struct("IPresentationManager")
                .field(
                    "QueryInterruptTimePrecise",
                    &(fnQueryInterruptTimePrecise as usize),
                )
                .finish(),
        }
    }
}

impl PresentationTimer {
    /// Create a presentation timer using QueryPerformanceFrequency (what DXGI uses for presentation times)
    pub fn new_dxgi() -> Self {
        let mut frequency: LARGE_INTEGER = unsafe { mem::zeroed() };
        let success = unsafe { QueryPerformanceFrequency(&mut frequency) };
        assert_ne!(success, 0);

        Self::Dxgi {
            frequency: unsafe { *frequency.QuadPart() } as u64,
        }
    }

    /// Create a presentation timer using QueryInterruptTimePrecise (what IPresentationManager uses for presentation times)
    ///
    /// Panics if QueryInterruptTimePrecise isn't found (below Win10)
    pub fn new_ipresentation_manager() -> Self {
        // We need to load this explicitly, as QueryInterruptTimePrecise is only available on Windows 10+
        //
        // Docs say it's in kernel32.dll, but it's actually in kernelbase.dll.
        let kernelbase =
            libloading::os::windows::Library::open_already_loaded("kernelbase.dll").unwrap();
        // No concerns about lifetimes here as kernelbase is always there.
        let ptr = unsafe { kernelbase.get(b"QueryInterruptTimePrecise").unwrap() };
        Self::IPresentationManager {
            fnQueryInterruptTimePrecise: *ptr,
        }
    }

    /// Gets the current time in nanoseconds.
    pub fn get_timestamp_ns(&self) -> u128 {
        // Always do u128 math _after_ hitting the timing function.
        match *self {
            PresentationTimer::Dxgi { frequency } => {
                let mut counter: LARGE_INTEGER = unsafe { mem::zeroed() };
                let success = unsafe { QueryPerformanceCounter(&mut counter) };
                assert_ne!(success, 0);

                // counter * (1_000_000_000 / freq) but re-ordered to make more precise
                (unsafe { *counter.QuadPart() } as u128 * 1_000_000_000) / frequency as u128
            }
            PresentationTimer::IPresentationManager {
                fnQueryInterruptTimePrecise,
            } => {
                let mut counter = 0;
                unsafe { fnQueryInterruptTimePrecise(&mut counter) };

                // QueryInterruptTimePrecise uses units of 100ns for its tick.
                counter as u128 * 100
            }
        }
    }
}
