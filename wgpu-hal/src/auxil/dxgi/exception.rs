use std::{borrow::Cow, slice};

use parking_lot::{lock_api::RawMutex, Mutex};
use winapi::{
    um::{errhandlingapi, winnt},
    vc::excpt,
};

// This is a mutex as opposed to an atomic as we need to completely
// lock everyone out until we have registered or unregistered the
// exception handler, otherwise really nasty races could happen.
//
// By routing all the registration through these functions we can guarentee
// there is either 1 or 0 exception handlers registered, not multiple.
static EXCEPTION_HANLDER_COUNT: Mutex<usize> = Mutex::const_new(parking_lot::RawMutex::INIT, 0);

pub fn register_exception_handler() {
    let mut count_guard = EXCEPTION_HANLDER_COUNT.lock();
    if *count_guard == 0 {
        unsafe {
            errhandlingapi::AddVectoredExceptionHandler(0, Some(output_debug_string_handler))
        };
    }
    *count_guard += 1;
}

pub fn unregister_exception_handler() {
    let mut count_guard = EXCEPTION_HANLDER_COUNT.lock();
    if *count_guard == 1 {
        unsafe {
            errhandlingapi::RemoveVectoredExceptionHandler(output_debug_string_handler as *mut _)
        };
    }
    *count_guard -= 1;
}

const MESSAGE_PREFIXES: &[(&str, log::Level)] = &[
    ("CORRUPTION", log::Level::Error),
    ("ERROR", log::Level::Error),
    ("WARNING", log::Level::Warn),
    ("INFO", log::Level::Info),
    ("MESSAGE", log::Level::Debug),
];

unsafe extern "system" fn output_debug_string_handler(
    exception_info: *mut winnt::EXCEPTION_POINTERS,
) -> i32 {
    // See https://stackoverflow.com/a/41480827
    let record = unsafe { &*(*exception_info).ExceptionRecord };
    if record.NumberParameters != 2 {
        return excpt::EXCEPTION_CONTINUE_SEARCH;
    }
    let message = match record.ExceptionCode {
        winnt::DBG_PRINTEXCEPTION_C => String::from_utf8_lossy(unsafe {
            slice::from_raw_parts(
                record.ExceptionInformation[1] as *const u8,
                record.ExceptionInformation[0],
            )
        }),
        winnt::DBG_PRINTEXCEPTION_WIDE_C => Cow::Owned(String::from_utf16_lossy(unsafe {
            slice::from_raw_parts(
                record.ExceptionInformation[1] as *const u16,
                record.ExceptionInformation[0],
            )
        })),
        _ => return excpt::EXCEPTION_CONTINUE_SEARCH,
    };

    let message = match message.strip_prefix("D3D12 ") {
        Some(msg) => msg
            .trim_end_matches("\n\0")
            .trim_end_matches("[ STATE_CREATION WARNING #0: UNKNOWN]"),
        None => return excpt::EXCEPTION_CONTINUE_SEARCH,
    };

    let (message, level) = match MESSAGE_PREFIXES
        .iter()
        .find(|&&(prefix, _)| message.starts_with(prefix))
    {
        Some(&(prefix, level)) => (&message[prefix.len() + 2..], level),
        None => (message, log::Level::Debug),
    };

    if level == log::Level::Warn && message.contains("#82") {
        // This is are useless spammy warnings (#820, #821):
        // "The application did not pass any clear value to resource creation"
        return excpt::EXCEPTION_CONTINUE_SEARCH;
    }

    if level == log::Level::Warn && message.contains("DRAW_EMPTY_SCISSOR_RECTANGLE") {
        // This is normal, WebGPU allows passing empty scissor rectangles.
        return excpt::EXCEPTION_CONTINUE_SEARCH;
    }

    let _ = std::panic::catch_unwind(|| {
        log::log!(level, "{}", message);
    });

    if cfg!(debug_assertions) && level == log::Level::Error {
        // Set canary and continue
        crate::VALIDATION_CANARY.set();
    }

    excpt::EXCEPTION_CONTINUE_EXECUTION
}
