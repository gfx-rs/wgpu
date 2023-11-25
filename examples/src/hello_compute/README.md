# hello-compute

Runs a compute shader to determine the number of iterations of the rules from
Collatz Conjecture

- If n is even, n = n/2
- If n is odd, n = 3n+1

that it will take to finish and reach the number `1`.

## To Run

```
# Pass in any 4 numbers as arguments
RUST_LOG=hello_compute cargo run --bin wgpu-examples hello_compute 1 4 3 295
```

## Example Output

```
[2020-04-25T11:15:33Z INFO  hello_compute] Steps: [0, 2, 7, 55]
```
