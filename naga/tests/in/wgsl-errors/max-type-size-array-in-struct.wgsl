// If a struct member is oversize, the error is raised during lowering.
// For struct members we can associate the error with the member.

struct ContainsLargeArray {
    arr: array<u32, 1 << 29>,
}
