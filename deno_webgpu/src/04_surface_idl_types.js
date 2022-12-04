// Copyright 2018-2022 the Deno authors. All rights reserved. MIT license.

// @ts-check
/// <reference path="../../core/lib.deno_core.d.ts" />
/// <reference path="../web/internal.d.ts" />
/// <reference path="../web/lib.deno_web.d.ts" />
/// <reference path="./lib.deno_webgpu.d.ts" />

"use strict";

((window) => {
  const webidl = window.__bootstrap.webidl;
  const { GPUTextureUsage } = window.__bootstrap.webgpu;

  // ENUM: PredefinedColorSpace
  webidl.converters["PredefinedColorSpace"] = webidl.createEnumConverter(
    "PredefinedColorSpace",
    [
      "srgb",
      "display-p3",
    ],
  );

  // ENUM: GPUCanvasAlphaMode
  webidl.converters["GPUCanvasAlphaMode"] = webidl.createEnumConverter(
    "GPUCanvasAlphaMode",
    [
      "opaque",
      "premultiplied",
    ],
  );

  // DICT: GPUCanvasConfiguration
  const dictMembersGPUCanvasConfiguration = [
    { key: "device", converter: webidl.converters.GPUDevice, required: true },
    {
      key: "format",
      converter: webidl.converters.GPUTextureFormat,
      required: true,
    },
    {
      key: "usage",
      converter: webidl.converters["GPUTextureUsageFlags"],
      defaultValue: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    {
      key: "alphaMode",
      converter: webidl.converters["GPUCanvasAlphaMode"],
      defaultValue: "opaque",
    },
  ];
  webidl.converters["GPUCanvasConfiguration"] = webidl
    .createDictionaryConverter(
      "GPUCanvasConfiguration",
      dictMembersGPUCanvasConfiguration,
    );
})(this);
