const g_false: bool = false;

@compute @workgroup_size(1, 1, 1) 
fn foo() {
    return;
}
