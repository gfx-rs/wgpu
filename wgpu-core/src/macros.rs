macro_rules! backends_map {
    // one let statement per backend with mapped data
    (
        let map = |$backend:pat| $map:block;
        $(
            #[cfg($backend_cfg:meta)] let $pat:pat = map($expr:expr);
        )*
    ) => {
        $(
            #[cfg($backend_cfg)]
            let $pat = {
                let $backend = $expr;
                $map
            };
        )*
    };

    // one block statement per backend with mapped data
    (
        let map = |$backend:pat| $map:block;
        $(
            #[cfg($backend_cfg:meta)] map($expr:expr),
        )*
    ) => {
        $(
            #[cfg($backend_cfg)]
            {
                let $backend = $expr;
                $map
            }
        )*
    };
}

#[test]
fn test_backend_macro() {
    struct Foo {
        #[cfg(any(windows, all(unix, not(target_os = "ios"), not(target_os = "macos")),))]
        vulkan: u32,

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        metal: u32,

        #[cfg(dx12)]
        dx12: u32,

        #[cfg(dx11)]
        dx11: u32,
    }

    // test struct construction
    let test_foo: Foo = {
        Foo {
            #[cfg(vulkan)]
            vulkan: 101,
            #[cfg(metal)]
            metal: 102,
            #[cfg(dx12)]
            dx12: 103,
            #[cfg(dx11)]
            dx11: 104,
        }
    };

    let mut vec = Vec::new();

    // test basic statement-per-backend
    backends_map! {
        let map = |(id, chr)| {
            vec.push((id, chr));
        };

        #[cfg(vulkan)]
        map((test_foo.vulkan, 'a')),

        #[cfg(metal)]
        map((test_foo.metal, 'b')),

        #[cfg(dx12)]
        map((test_foo.dx12, 'c')),

        #[cfg(dx11)]
        map((test_foo.dx11, 'd')),
    }

    #[cfg(any(windows, all(unix, not(target_os = "ios"), not(target_os = "macos")),))]
    assert!(vec.contains(&(101, 'a')));

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    assert!(vec.contains(&(102, 'b')));

    #[cfg(dx12)]
    assert!(vec.contains(&(103, 'c')));

    #[cfg(dx11)]
    assert!(vec.contains(&(104, 'd')));

    // test complex statement-per-backend
    backends_map! {
        let map = |(id, pred, code)| {
            if pred(id) {
                code();
            }
        };

        #[cfg(vulkan)]
        map((test_foo.vulkan, |v| v == 101, || println!("vulkan"))),

        #[cfg(metal)]
        map((test_foo.metal, |v| v == 102, || println!("metal"))),

        #[cfg(dx12)]
        map((test_foo.dx12, |v| v == 103, || println!("dx12"))),

        #[cfg(dx11)]
        map((test_foo.dx11, |v| v == 104, || println!("dx11"))),
    }

    // test struct construction 2
    let test_foo_2: Foo = Foo {
        #[cfg(vulkan)]
        vulkan: 1,

        #[cfg(metal)]
        metal: 2,

        #[cfg(dx12)]
        dx12: 3,

        #[cfg(dx11)]
        dx11: 4,
    };

    #[cfg(vulkan)]
    let var_vulkan = test_foo_2.vulkan;

    #[cfg(metal)]
    let var_metal = test_foo_2.metal;

    #[cfg(dx12)]
    let var_dx12 = test_foo_2.dx12;

    #[cfg(dx11)]
    let var_dx11 = test_foo_2.dx11;

    backends_map! {
        let map = |(id, chr, var)| { (chr, id, var) };

        #[cfg(vulkan)]
        let var_vulkan = map((test_foo_2.vulkan, 'a', var_vulkan));

        #[cfg(metal)]
        let var_metal = map((test_foo_2.metal, 'b', var_metal));

        #[cfg(dx12)]
        let var_dx12 = map((test_foo_2.dx12, 'c', var_dx12));

        #[cfg(dx11)]
        let var_dx11 = map((test_foo_2.dx11, 'd', var_dx11));
    }

    #[cfg(vulkan)]
    {
        println!("backend int: {:?}", var_vulkan);
    }

    #[cfg(metal)]
    {
        println!("backend int: {:?}", var_metal);
    }

    #[cfg(dx12)]
    {
        println!("backend int: {:?}", var_dx12);
    }

    #[cfg(dx11)]
    {
        println!("backend int: {:?}", var_dx11);
    }

    #[cfg(any(windows, all(unix, not(target_os = "ios"), not(target_os = "macos")),))]
    let _ = var_vulkan;

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    let _ = var_metal;

    #[cfg(dx12)]
    let _ = var_dx12;

    #[cfg(dx11)]
    let _ = var_dx11;
}
