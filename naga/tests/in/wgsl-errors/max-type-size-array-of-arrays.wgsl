// If the size of the base type of an array is oversize, the error is raised
// during lowering. Anonymous types don't get spans so this error isn't very
// helpful.

alias ArrayOfArrays = array<array<u32, 1 << 29>, 22>;
