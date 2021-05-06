[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	//TODO: execution-only barrier?
	storageBarrier();
	workgroupBarrier();
}
