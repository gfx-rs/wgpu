const adapter = await navigator.gpu.requestAdapter();

const numbers = [1, 4, 3, 295];

const device = await adapter.requestDevice();

const shaderCode = `
@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience
// The Collatz Conjecture states that for any integer n:
// If n is even, n = n/2
// If n is odd, n = 3n+1
// And repeat this process for each new n, you will always eventually reach 1.
// Though the conjecture has not been proven, no counterexample has ever been found.
// This function returns how many times this recurrence needs to be applied to reach 1.
fn collatz_iterations(n_base: u32) -> u32{
    var n: u32 = n_base;
    var i: u32 = 0u;
    loop {
        if (n <= 1u) {
            break;
        }
        if (n % 2u == 0u) {
            n = n / 2u;
        }
        else {
            // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
            if (n >= 1431655765u) {   // 0x55555555u
                return 4294967295u;   // 0xffffffffu
            }
            n = 3u * n + 1u;
        }
        i = i + 1u;
    }
    return i;
}
@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
}`;

const shaderModule = device.createShaderModule({
  code: shaderCode,
});

const size = new Uint32Array(numbers).byteLength;

const stagingBuffer = device.createBuffer({
  size: size,
  usage: 1 | 8,
});

const storageBuffer = device.createBuffer({
  label: "Storage Buffer",
  size: size,
  usage: 0x80 | 8 | 4,
  mappedAtCreation: true,
});

const buf = new Uint32Array(storageBuffer.getMappedRange());

buf.set(numbers);

storageBuffer.unmap();

const computePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: shaderModule,
    entryPoint: "main",
  },
});
const bindGroupLayout = computePipeline.getBindGroupLayout(0);

const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: {
        buffer: storageBuffer,
      },
    },
  ],
});

const encoder = device.createCommandEncoder();

const computePass = encoder.beginComputePass();
computePass.setPipeline(computePipeline);
computePass.setBindGroup(0, bindGroup);
computePass.insertDebugMarker("compute collatz iterations");
computePass.dispatchWorkgroups(numbers.length);
computePass.end();

encoder.copyBufferToBuffer(storageBuffer, 0, stagingBuffer, 0, size);

device.queue.submit([encoder.finish()]);

await stagingBuffer.mapAsync(1);

const data = stagingBuffer.getMappedRange();

function isTypedArrayEqual(a, b) {
  if (a.byteLength !== b.byteLength) return false;
  return a.every((val, i) => val === b[i]);
}

const actual = new Uint32Array(data);
const expected = new Uint32Array([0, 2, 7, 55]);

console.error("actual", actual);
console.error("expected", expected);

if (!isTypedArrayEqual(actual, expected)) {
  throw new TypeError("Actual does not equal expected!");
}

stagingBuffer.unmap();

device.destroy();
