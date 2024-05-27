// Adapted from https://github.com/denoland/deno/blob/6abf126c2a7a451cded8c6b5e6ddf1b69c84055d/runtime/js/99_main.js

// Removes the `__proto__` for security reasons.  This intentionally makes
// Deno non compliant with ECMA-262 Annex B.2.2.1
//
delete Object.prototype.__proto__;

import { core, primordials } from "ext:core/mod.js";
const {
  Error,
  ObjectDefineProperty,
  ObjectDefineProperties,
  ObjectSetPrototypeOf,
  Symbol,
  DateNow,
} = primordials;

import { pathFromURL } from "ext:deno_web/00_infra.js";
import * as webidl from "ext:deno_webidl/00_webidl.js";
import * as globalInterfaces from "ext:deno_web/04_global_interfaces.js";
import * as event from "ext:deno_web/02_event.js";
import * as timers from "ext:deno_web/02_timers.js";
import * as base64 from "ext:deno_web/05_base64.js";
import * as encoding from "ext:deno_web/08_text_encoding.js";
import { Console } from "ext:deno_console/01_console.js";
import * as url from "ext:deno_url/00_url.js";
import { DOMException } from "ext:deno_web/01_dom_exception.js";
import * as performance from "ext:deno_web/15_performance.js";
import { loadWebGPU } from "ext:deno_webgpu/00_init.js";
import * as imageData from "ext:deno_web/16_image_data.js";
const webgpu = loadWebGPU();

// imports needed to pass module evaluation
import "ext:deno_url/01_urlpattern.js";
import "ext:deno_web/01_mimesniff.js";
import "ext:deno_web/03_abort_signal.js";
import "ext:deno_web/06_streams.js";
import "ext:deno_web/09_file.js";
import "ext:deno_web/10_filereader.js";
import "ext:deno_web/12_location.js";
import "ext:deno_web/13_message_port.js";
import "ext:deno_web/14_compression.js";
import "ext:deno_webgpu/02_surface.js";

let globalThis_;

class NotFound extends Error {
  constructor(msg) {
    super(msg);
    this.name = "NotFound";
  }
}

class BrokenPipe extends Error {
  constructor(msg) {
    super(msg);
    this.name = "BrokenPipe";
  }
}

class AlreadyExists extends Error {
  constructor(msg) {
    super(msg);
    this.name = "AlreadyExists";
  }
}

class InvalidData extends Error {
  constructor(msg) {
    super(msg);
    this.name = "InvalidData";
  }
}

class TimedOut extends Error {
  constructor(msg) {
    super(msg);
    this.name = "TimedOut";
  }
}

class WriteZero extends Error {
  constructor(msg) {
    super(msg);
    this.name = "WriteZero";
  }
}

class UnexpectedEof extends Error {
  constructor(msg) {
    super(msg);
    this.name = "UnexpectedEof";
  }
}

class NotSupported extends Error {
  constructor(msg) {
    super(msg);
    this.name = "NotSupported";
  }
}

const util = {
  writable(value) {
    return {
      value,
      writable: true,
      enumerable: true,
      configurable: true,
    };
  },
  nonEnumerable(value) {
    return {
      value,
      writable: true,
      configurable: true,
    };
  },
  readOnly(value) {
    return {
      value,
      enumerable: true,
    };
  },
  getterOnly(getter) {
    return {
      get: getter,
      set() {
      },
      enumerable: true,
      configurable: true,
    };
  },
};

class Navigator {
  constructor() {
    webidl.illegalConstructor();
  }

  [Symbol.for("Deno.customInspect")](inspect) {
    return `${this.constructor.name} ${inspect({})}`;
  }
}
const NavigatorPrototype = Navigator.prototype;

const navigator = webidl.createBranded(Navigator);

ObjectDefineProperties(NavigatorPrototype, {
  gpu: {
    configurable: true,
    enumerable: true,
    get() {
      webidl.assertBranded(this, NavigatorPrototype);
      return webgpu.gpu;
    },
  },
});

const windowOrWorkerGlobalScope = {
  CloseEvent: util.nonEnumerable(event.CloseEvent),
  CustomEvent: util.nonEnumerable(event.CustomEvent),
  DOMException: util.nonEnumerable(DOMException),
  ErrorEvent: util.nonEnumerable(event.ErrorEvent),
  Event: util.nonEnumerable(event.Event),
  EventTarget: util.nonEnumerable(event.EventTarget),
  Navigator: util.nonEnumerable(Navigator),
  navigator: util.getterOnly(() => navigator),
  MessageEvent: util.nonEnumerable(event.MessageEvent),
  Performance: util.nonEnumerable(performance.Performance),
  PerformanceEntry: util.nonEnumerable(performance.PerformanceEntry),
  PerformanceMark: util.nonEnumerable(performance.PerformanceMark),
  PerformanceMeasure: util.nonEnumerable(performance.PerformanceMeasure),
  TextDecoder: util.nonEnumerable(encoding.TextDecoder),
  TextEncoder: util.nonEnumerable(encoding.TextEncoder),
  URL: util.nonEnumerable(url.URL),
  URLSearchParams: util.nonEnumerable(url.URLSearchParams),
  atob: util.writable(base64.atob),
  btoa: util.writable(base64.btoa),
  console: util.writable(new Console(core.print)),
  setInterval: util.writable(timers.setInterval),
  setTimeout: util.writable(timers.setTimeout),
  clearInterval: util.writable(timers.clearInterval),
  clearTimeout: util.writable(timers.clearTimeout),
  performance: util.writable(performance.performance),
  ImageData: core.propNonEnumerable(imageData.ImageData),

  GPU: util.nonEnumerable(webgpu.GPU),
  GPUAdapter: util.nonEnumerable(webgpu.GPUAdapter),
  GPUAdapterInfo: util.nonEnumerable(webgpu.GPUAdapterInfo),
  GPUSupportedLimits: util.nonEnumerable(webgpu.GPUSupportedLimits),
  GPUSupportedFeatures: util.nonEnumerable(webgpu.GPUSupportedFeatures),
  GPUDeviceLostInfo: util.nonEnumerable(webgpu.GPUDeviceLostInfo),
  GPUDevice: util.nonEnumerable(webgpu.GPUDevice),
  GPUQueue: util.nonEnumerable(webgpu.GPUQueue),
  GPUBuffer: util.nonEnumerable(webgpu.GPUBuffer),
  GPUBufferUsage: util.nonEnumerable(webgpu.GPUBufferUsage),
  GPUMapMode: util.nonEnumerable(webgpu.GPUMapMode),
  GPUTextureUsage: util.nonEnumerable(webgpu.GPUTextureUsage),
  GPUTexture: util.nonEnumerable(webgpu.GPUTexture),
  GPUTextureView: util.nonEnumerable(webgpu.GPUTextureView),
  GPUSampler: util.nonEnumerable(webgpu.GPUSampler),
  GPUBindGroupLayout: util.nonEnumerable(webgpu.GPUBindGroupLayout),
  GPUPipelineLayout: util.nonEnumerable(webgpu.GPUPipelineLayout),
  GPUBindGroup: util.nonEnumerable(webgpu.GPUBindGroup),
  GPUShaderModule: util.nonEnumerable(webgpu.GPUShaderModule),
  GPUShaderStage: util.nonEnumerable(webgpu.GPUShaderStage),
  GPUComputePipeline: util.nonEnumerable(webgpu.GPUComputePipeline),
  GPURenderPipeline: util.nonEnumerable(webgpu.GPURenderPipeline),
  GPUColorWrite: util.nonEnumerable(webgpu.GPUColorWrite),
  GPUCommandEncoder: util.nonEnumerable(webgpu.GPUCommandEncoder),
  GPURenderPassEncoder: util.nonEnumerable(webgpu.GPURenderPassEncoder),
  GPUComputePassEncoder: util.nonEnumerable(webgpu.GPUComputePassEncoder),
  GPUCommandBuffer: util.nonEnumerable(webgpu.GPUCommandBuffer),
  GPURenderBundleEncoder: util.nonEnumerable(webgpu.GPURenderBundleEncoder),
  GPURenderBundle: util.nonEnumerable(webgpu.GPURenderBundle),
  GPUQuerySet: util.nonEnumerable(webgpu.GPUQuerySet),
  GPUError: util.nonEnumerable(webgpu.GPUError),
  GPUValidationError: util.nonEnumerable(webgpu.GPUValidationError),
  GPUOutOfMemoryError: util.nonEnumerable(webgpu.GPUOutOfMemoryError),
};

windowOrWorkerGlobalScope.console.enumerable = false;

const mainRuntimeGlobalProperties = {
  Window: globalInterfaces.windowConstructorDescriptor,
  window: util.readOnly(globalThis),
  self: util.readOnly(globalThis),
};

const denoNs = {
  exit(code) {
    core.ops.op_exit(code);
  },
  readFileSync(path) {
    return core.ops.op_read_file_sync(pathFromURL(path));
  },
  readTextFileSync(path) {
    const buf = core.ops.op_read_file_sync(pathFromURL(path));
    const decoder = new TextDecoder();
    return decoder.decode(buf);
  },
  writeFileSync(path, buf) {
    return core.ops.op_write_file_sync(pathFromURL(path), buf);
  },
};

core.registerErrorClass("NotFound", NotFound);
core.registerErrorClass("AlreadyExists", AlreadyExists);
core.registerErrorClass("InvalidData", InvalidData);
core.registerErrorClass("TimedOut", TimedOut);
core.registerErrorClass("WriteZero", WriteZero);
core.registerErrorClass("UnexpectedEof", UnexpectedEof);
core.registerErrorClass("NotSupported", NotSupported);
core.registerErrorBuilder(
  "DOMExceptionOperationError",
  function DOMExceptionOperationError(msg) {
    return new DOMException(msg, "OperationError");
  },
);
core.registerErrorBuilder(
    "DOMExceptionAbortError",
    function DOMExceptionAbortError(msg) {
      return new domException.DOMException(msg, "AbortError");
    },
);
core.registerErrorBuilder(
    "DOMExceptionInvalidCharacterError",
    function DOMExceptionInvalidCharacterError(msg) {
      return new domException.DOMException(msg, "InvalidCharacterError");
    },
);
core.registerErrorBuilder(
    "DOMExceptionDataError",
    function DOMExceptionDataError(msg) {
      return new domException.DOMException(msg, "DataError");
    },
);

let hasBootstrapped = false;

function bootstrapRuntime({ args, cwd }) {
  if (hasBootstrapped) {
    throw new Error("Runtime has already been bootstrapped.");
  }
  performance.setTimeOrigin(DateNow());
  globalThis_ = globalThis;

  // Remove bootstrapping data from the global scope
  delete globalThis.__bootstrap;
  delete globalThis.bootstrap;
  hasBootstrapped = true;

  event.setEventTargetData(globalThis);
  event.saveGlobalThisReference(globalThis);

  Error.prepareStackTrace = core.prepareStackTrace;

  ObjectDefineProperties(globalThis, windowOrWorkerGlobalScope);
  ObjectDefineProperties(globalThis, mainRuntimeGlobalProperties);
  ObjectSetPrototypeOf(globalThis, Window.prototype);
  event.setEventTargetData(globalThis);

  denoNs.args = args;
  denoNs.cwd = () => cwd;

  ObjectDefineProperty(globalThis, "Deno", util.readOnly(denoNs));

  Error.prepareStackTrace = core.prepareStackTrace;
}

globalThis.bootstrap = bootstrapRuntime;
