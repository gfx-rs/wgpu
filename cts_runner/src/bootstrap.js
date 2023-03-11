// Adapted from https://github.com/denoland/deno/blob/6abf126c2a7a451cded8c6b5e6ddf1b69c84055d/runtime/js/99_main.js

// Removes the `__proto__` for security reasons.  This intentionally makes
// Deno non compliant with ECMA-262 Annex B.2.2.1
//
delete Object.prototype.__proto__;

const core = Deno.core;
const primordials = globalThis.__bootstrap.primordials;
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
import { Console } from "ext:deno_console/02_console.js";
import * as url from "ext:deno_url/00_url.js";
import DOMException from "ext:deno_web/01_dom_exception.js";
import * as performance from "ext:deno_web/15_performance.js";
import * as webgpu from "ext:deno_webgpu/01_webgpu.js";

let globalThis_;

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

ObjectDefineProperties(Navigator.prototype, {
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

  GPU: util.nonEnumerable(webgpu.GPU),
  GPUAdapter: util.nonEnumerable(webgpu.GPUAdapter),
  GPUAdapterLimits: util.nonEnumerable(webgpu.GPUAdapterLimits),
  GPUSupportedFeatures: util.nonEnumerable(webgpu.GPUSupportedFeatures),
  GPUDevice: util.nonEnumerable(webgpu.GPUDevice),
  GPUQueue: util.nonEnumerable(webgpu.GPUQueue),
  GPUBuffer: util.nonEnumerable(webgpu.GPUBuffer),
  GPUBufferUsage: util.nonEnumerable(webgpu.GPUBufferUsage),
  GPUMapMode: util.nonEnumerable(webgpu.GPUMapMode),
  GPUTexture: util.nonEnumerable(webgpu.GPUTexture),
  GPUTextureUsage: util.nonEnumerable(webgpu.GPUTextureUsage),
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
  GPUOutOfMemoryError: util.nonEnumerable(webgpu.GPUOutOfMemoryError),
  GPUValidationError: util.nonEnumerable(webgpu.GPUValidationError),
};

windowOrWorkerGlobalScope.console.enumerable = false;

const mainRuntimeGlobalProperties = {
  Window: globalInterfaces.windowConstructorDescriptor,
  window: util.readOnly(globalThis),
  self: util.readOnly(globalThis),
};

const denoNs = {
  exit(code) {
    core.opSync("op_exit", code);
  },
  readFileSync(path) {
    return core.opSync("op_read_file_sync", pathFromURL(path));
  },
  readTextFileSync(path) {
    const buf = core.opSync("op_read_file_sync", pathFromURL(path));
    const decoder = new TextDecoder();
    return decoder.decode(buf);
  },
  writeFileSync(path, buf) {
    return core.opSync("op_write_file_sync", pathFromURL(path), buf);
  },
};

core.registerErrorBuilder(
  "DOMExceptionOperationError",
  function DOMExceptionOperationError(msg) {
    return new DOMException(msg, "OperationError");
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
