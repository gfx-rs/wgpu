[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	//TODO: execution-only barrier?
	storageBarrier();
	workgroupBarrier();

	var pos: i32;
	// switch without cases
    switch (1) {
        default: {
			pos = 1;
		}
    }

	switch (pos) {
		case 1: {
			pos = 0;
			break;
		}
		case 2: {
			pos = 1;
		}
		case 3: {
			pos = 2;
			fallthrough;
		}
		case 4: {}
        default: {
			pos = 3;
		}
    }
}
