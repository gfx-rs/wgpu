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
  const { _device, assertDevice } = window.__bootstrap.webgpu;

  const _surfaceRid = Symbol("[[surfaceRid]]");
  const _width = Symbol("[[width]]");
  const _height = Symbol("[[height]]");
  class GPUCanvasContext {
    [_surfaceRid];
    /** @type {number} */
    [_width];
    /** @type {number} */
    [_height];

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

      const device = assertDevice({ [_device]: configuration.device }, { prefix, context: "configuration.device" });

      const { err } = ops.op_webgpu_surface_configure({
        surface_rid: this[_surfaceRid],
        deviceRid: device.rid,
        format: configuration.format,
        usage: configuration.usage,
        width: this[_width],
        height: this[_height],
        alpha_mode: configuration.alphaMode,
      });

      device.pushError(err);
    }

    unconfigure() {
      webidl.assertBranded(this, GPUCanvasContextPrototype);

      // TODO
    }

    getCurrentTexture() {
      webidl.assertBranded(this, GPUCanvasContextPrototype);

    }
  }
  const GPUCanvasContextPrototype = GPUCanvasContext.prototype;

  function createCanvasContext(rawHandleRid) {
    const canvasContext = webidl.createBranded(GPUCanvasContext);
    const { rid } = ops.op_webgpu_create_surface(rawHandleRid);
    canvasContext[_surfaceRid] = rid;
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
