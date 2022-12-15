// Copyright 2018-2022 the Deno authors. All rights reserved. MIT license.

// @ts-check
/// <reference path="../../core/lib.deno_core.d.ts" />
/// <reference path="../web/internal.d.ts" />
/// <reference path="../web/lib.deno_web.d.ts" />
/// <reference path="./lib.deno_webgpu.d.ts" />

"use strict";

((window) => {
  const core = window.Deno.core;
  const ops = core.ops;
  const webidl = window.__bootstrap.webidl;
  const { Symbol } = window.__bootstrap.primordials;
  const { _device, assertDevice, createGPUTexture } = window.__bootstrap.webgpu;

  const _surfaceRid = Symbol("[[surfaceRid]]");
  const _config = Symbol("[[config]]");
  const _width = Symbol("[[width]]");
  const _height = Symbol("[[height]]");
  class GPUCanvasContext {
    [_surfaceRid];
    /** @type {number} */
    [_width];
    /** @type {number} */
    [_height];
    /** @type {InnerGPUDevice} */
    [_device];
    [_config];

    constructor() {
      webidl.illegalConstructor();
    }

    configure(configuration) {
      webidl.assertBranded(this, GPUCanvasContextPrototype);
      const prefix = "Failed to execute 'configure' on 'GPUCanvasContext'";
      webidl.requiredArguments(arguments.length, 1, { prefix });
      configuration = webidl.converters.GPUCanvasConfiguration(configuration, {
        prefix,
        context: "Argument 1",
      });

      this[_device] = configuration.device[_device];
      this[_config] = configuration;
      const device = assertDevice(this, { prefix, context: "configuration.device" });

      const { err } = ops.op_webgpu_surface_configure({
        surfaceRid: this[_surfaceRid],
        deviceRid: device.rid,
        format: configuration.format,
        usage: configuration.usage,
        width: this[_width],
        height: this[_height],
        alphaMode: configuration.alphaMode,
      });

      device.pushError(err);
    }

    unconfigure() {
      webidl.assertBranded(this, GPUCanvasContextPrototype);

      // TODO
    }

    getCurrentTexture() {
      webidl.assertBranded(this, GPUCanvasContextPrototype);
      const prefix = "Failed to execute 'getCurrentTexture' on 'GPUCanvasContext'";
      const device = assertDevice(this, { prefix, context: "this" });

      const { rid } = ops.op_webgpu_surface_get_current_texture(device.rid, this[_surfaceRid]);

      const texture = createGPUTexture(
        {
          size: {
            width: this[_width],
            height: this[_height],
            depthOrArrayLayers: 1,
          },
          mipLevelCount: 1,
          sampleCount: 1,
          dimension: "2d",
          format: this[_config].format,
          usage: this[_config].usage,
        },
        device,
        rid,
      );
      device.trackResource(texture);
      return texture;
    }

    present() { // TODO: not part of spec
      webidl.assertBranded(this, GPUCanvasContextPrototype);
      const prefix = "Failed to execute 'present' on 'GPUCanvasContext'";
      const device = assertDevice(this, { prefix, context: "this" });
      ops.op_webgpu_surface_present(device.rid, this[_surfaceRid]);
    }
  }
  const GPUCanvasContextPrototype = GPUCanvasContext.prototype;

  function createCanvasContext(surfaceRid, width, height) {
    const canvasContext = webidl.createBranded(GPUCanvasContext);
    canvasContext[_surfaceRid] = surfaceRid;
    canvasContext[_width] = width;
    canvasContext[_height] = height;
    return canvasContext;
  }

  window.__bootstrap.webgpu = {
    ...window.__bootstrap.webgpu,
    _width,
    _height,
    GPUCanvasContext,
    createCanvasContext,
  };
})(this);
