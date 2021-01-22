[[builtin(global_invocation_id)]]
var global_id: vec3<u32>;

struct PrimeIndices {
    data: array<u32>;
}; // this is used as both input and output for convenience

[[group(0), binding(0)]]
var<storage> v_indices: [[access(read_write)]] PrimeIndices;

// The Collatz Conjecture states that for any integer n:
// If n is even, n = n/2
// If n is odd, n = 3n+1
// And repeat this process for each new n, you will always eventually reach 1.
// Though the conjecture has not been proven, no counterexample has ever been found.
// This function returns how many times this recurrence needs to be applied to reach 1.
fn collatz_iterations(n: u32) -> u32{
    var i: u32 = u32(0);
    loop {
        if (n <= u32(1)) {
            break;
        }
        if (n % u32(2) == u32(0)) {
            n = n / u32(2);
        }
        else {
            n = u32(3) * n + i32(1);
        }
        i = i + u32(1);
    }
    return i;
}

[[stage(compute), workgroup_size(1)]]
fn main() {
    v_indices.data[global_id.x] = collatz_iterations(v_indices.data[global_id.x]);
}
