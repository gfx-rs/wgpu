/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WGPU_h
#define WGPU_h

// Prelude of types necessary before including wgpu_ffi_generated.h
namespace mozilla {
namespace webgpu {
namespace ffi {

#define WGPU_INLINE
#define WGPU_FUNC
#define WGPU_DESTRUCTOR_SAFE_FUNC

extern "C" {
#include "wgpu_ffi_generated.h"
}

#undef WGPU_INLINE
#undef WGPU_FUNC
#undef WGPU_DESTRUCTOR_SAFE_FUNC

}  // namespace ffi
}  // namespace webgpu
}  // namespace mozilla

#endif  // WGPU_h
