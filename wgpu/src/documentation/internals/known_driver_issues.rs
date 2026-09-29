/*!
# Known Driver Issues

This page lists known issues with drivers that are not, or cannot, be worked
around by `wgpu`.

Issues are categorized by GPU vendor and backend. Where details such as
"driver version" are recorded, they are not intended to be exhaustive: these
are just the values where the issue has been observed. Just because your
combination is not explicitly listed does not mean you aren't susceptible to
the same issue.

## WGSL / All

- **WGSL draft spec compliance**

  As outlined in the README, section
  [Tracking the WebGPU and WGSL draft specifications](https://github.com/gfx-rs/wgpu#tracking-the-webgpu-and-wgsl-draft-specifications),
  the version of WGSL implemented by `wgpu` will likely differ from what is
  in the up-to-date spec.

## Intel / Vulkan

- **Integrated GPU not visible to programs named `cube`**

  | Property         | Values observed   |
  | ---------------- | ----------------- |
  | Operating System | Windows 10 64-bit |
  | Driver OEM       | Intel, Microsoft  |
  | Driver Version   | 27.20.100.8935    |

  In a dual-GPU system, the driver for the Intel GPU will hide itself from
  programs named `cube`. The issue can be worked around by disabling the
  discrete GPU or by renaming the program to something other than `cube`.

- **Hangs when the same semaphore is signaled and waited upon**

  | Property         | Values observed |
  | ---------------- | --------------- |
  | Operating System | Linux 5.14      |
  | Driver OEM       | Intel           |
  | Driver Version   | Mesa 21.2.3     |

  Using a "relay" semaphore confuses the Intel driver in Mesa. Reported in
  <https://gitlab.freedesktop.org/mesa/mesa/-/issues/5508>. Can be worked
  around by only doing one submission per frame.

- **`cmd_begin_debug_utils_label_ext` pointer is NULL**

  | Property         | Values observed     |
  | ---------------- | ------------------- |
  | Operating System | Windows 10 Pro 21H1 |
  | Driver OEM       | UHD Graphics 630    |
  | Driver Version   | 24.20.100.6287      |

  The extension is supported, but the pointer to the function is NULL.
  Escalated in <https://github.com/MaikKlein/ash/issues/484>. Still needs an
  upstream report.

## Intel / DirectX 12

- **Offset after resizing window** —
  [wgpu-rs#647](https://github.com/gfx-rs/wgpu-rs/issues/647)

  | Property         | Values observed   |
  | ---------------- | ----------------- |
  | Operating System | Windows 10 64-bit |
  | Driver OEM       | Microsoft         |
  | Driver Version   | 26.20.100.7639    |

  This issue affects programs running on any GPU if the Intel GPU is being
  used by the compositor. This would typically be the case in dual-GPU
  systems. It can be worked around by replacing the Microsoft OEM driver with
  the generic Intel driver.

- **`CreatePlacedResource` TDRs** —
  [wgpu#1319](https://github.com/gfx-rs/wgpu/issues/1319)

  | Property         | Values observed                |
  | ---------------- | ------------------------------ |
  | Operating System | Windows 10 64-bit, build 19042 |
  | GPU              | HD Graphics 4600               |
  | Driver Version   | 20.19.15.5171                  |

  Reported in
  <https://github.com/IGCIT/Intel-GPU-Community-Issue-Tracker-IGCIT/issues/44>.

## Intel / DX11

- **Buffer write not synchronized with draw** —
  [wgpu#1060](https://github.com/gfx-rs/wgpu/issues/1060)

  | Property         | Values observed |
  | ---------------- | --------------- |
  | Operating System | Windows         |

  Buffer upload isn't being properly synchronized with the draw call that is
  using it. This is entirely on the driver to get right; there is no way to
  do manual sync.

## Intel / All

- **Cannot combine MSAA with sRGB resolve** —
  [wgpu#725](https://github.com/gfx-rs/wgpu/issues/725)

  | Property         | Values observed |
  | ---------------- | --------------- |
  | Operating System | Linux           |

  Attempting to do MSAA resolve at the same time as sRGB resolve will cause
  MSAA to not work and the clear color to not be properly converted into
  sRGB.

## Nvidia / Vulkan

- **Hang/panic/device loss after specific render sequence** —
  [wgpu#983](https://github.com/gfx-rs/wgpu/issues/983)

  | Property         | Values observed           |
  | ---------------- | ------------------------- |
  | Operating System | Windows 10 64-bit         |
  | GPU              | GeForce RTX 2080 Ti       |
  | Driver Name      | GeForce Game Ready Driver |
  | Driver Version   | 456.38, 457.09, 457.30    |

  Does not occur on driver version 452.06. After submitting a frame to
  `wgpu`, Vulkan seems to become unstable and this manifests itself in a few
  ways.

- **`write_buffer` only works when aligned to 16** —
  [wgpu#1323](https://github.com/gfx-rs/wgpu/issues/1323)

  | Property         | Values observed              |
  | ---------------- | ---------------------------- |
  | Operating System | Windows 10 64-bit            |
  | GPU              | GeForce GTX 1050, 1070       |
  | Driver Version   | 27.21.14.5256, 27.21.14.6589 |

  We issue `vkCmdFillBuffer` for the uninitialized area of a buffer. If only
  the first 40 bytes were filled, the fill buffer is called with an offset of
  40, but the driver ends up erasing all the data starting with offset 32.
  Issue reported directly to the NVIDIA level-2 support team, ticket
  210412-000661.

- **Flat-interpolated integers come out as zeroes** —
  [wgpu#1775](https://github.com/gfx-rs/wgpu/issues/1775)

  | Property         | Values observed  |
  | ---------------- | ---------------- |
  | Operating System | Linux            |
  | GPU              | GeForce GTX 1050 |
  | Driver Version   | 390              |

  Fragment shader sees interpolated values as 0. Issue reported to NVIDIA
  Vulkan support.

## WARP / D3D12

- **CPU descriptors are used after `CopyDescriptors`** —
  [wgpu#1002](https://github.com/gfx-rs/wgpu/issues/1002)

  | Property         | Values observed   |
  | ---------------- | ----------------- |
  | Operating System | Windows 10 64-bit |

## AMD / D3D12

- **Updating partial texture data doesn't show up** —
  [wgpu#1306](https://github.com/gfx-rs/wgpu/issues/1306)

  | Property         | Values observed   |
  | ---------------- | ----------------- |
  | Operating System | Windows 10 64-bit |
  | GPU              | Ryzen 3500U       |
  | Driver Name      | ?                 |
  | Driver Version   | ?                 |

  When a texture only gets partially updated, sampling from it returns
  zeroes. Issue has been reported directly to AMD.

## Mesa / Vulkan

- **Integer overflow with very large buffers** —
  [wgpu#2796](https://github.com/gfx-rs/wgpu/pull/2796)

  | Property         | Values observed   |
  | ---------------- | ----------------- |
  | Operating System | Linux             |
  | Driver Name      | Mesa/lavapipe     |

  When a buffer size or range parameter does not fit a 32-bit signed integer,
  Mesa runs into integer overflows.

*/
