// The total size of a struct is checked during lowering. For a struct,
// we can associate the error with the struct itself.

struct TwoArrays {
    arr1: array<u32, 1 << 28>,
    arr2: array<u32, 1 << 28>,
}
