/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

macro_rules! backends {
    // one let statement per backend
    (
        #[vulkan] let $vk_field:pat = $vk_expr:expr;
        #[metal] let $mtl_field:pat = $mtl_expr:expr;
        #[dx12] let $dx12_field:pat = $dx12_expr:expr;
        #[dx11] let $dx11_field:pat = $dx11_expr:expr;
    ) => {
        #[cfg(any(
            windows,
            all(unix, not(any(target_os = "ios", target_os = "macos"))),
            feature = "gfx-backend-vulkan",
        ))]
        let $vk_field = $vk_expr;

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        let $mtl_field = $mtl_expr;

        #[cfg(windows)]
        let $dx12_field = $dx12_expr;

        #[cfg(windows)]
        let $dx11_field = $dx11_expr;
    };

    // one block statement per backend
    (
        #[vulkan] $vk_block:block
        #[metal] $mtl_block:block
        #[dx12] $dx12_block:block
        #[dx11] $dx11_block:block
    ) => {
        #[cfg(any(
            windows,
            all(unix, not(any(target_os = "ios", target_os = "macos"))),
            feature = "gfx-backend-vulkan",
        ))]
        $vk_block

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        $mtl_block

        #[cfg(windows)]
        $dx12_block

        #[cfg(windows)]
        $dx11_block
    };

    // a struct constructor with one field per backend
    (
        $Struct:path {
            #[vulkan] $vk_field:ident: $vk_expr:expr,
            #[metal] $mtl_field:ident: $mtl_expr:expr,
            #[dx12] $dx12_field:ident: $dx12_expr:expr,
            #[dx11] $dx11_field:ident: $dx11_expr:expr,
        }
    ) => {{
        $Struct {
            #[cfg(any(
                windows,
                all(unix, not(any(target_os = "ios", target_os = "macos"))),
                feature = "gfx-backend-vulkan",
            ))]
            $vk_field: $vk_expr,

            #[cfg(any(target_os = "ios", target_os = "macos"))]
            $mtl_field: $mtl_expr,

            #[cfg(windows)]
            $dx12_field: $dx12_expr,

            #[cfg(windows)]
            $dx11_field: $dx11_expr,
        }
    }};
}

macro_rules! backends_map {
    // one let statement per backend with mapped data
    (
        let map = |$backend:pat| $map:block;
        $(
            #[$backend_attr:ident] let $pat:pat = map($expr:expr);
        )*
    ) => {
        backends! {
            $(
                #[$backend_attr]
                let $pat = {
                    let $backend = $expr;
                    $map
                };
            )*
        }
    };

    // one block statement per backend with mapped data
    (
        let map = |$backend:pat| $map:block;
        $(
            #[$backend_attr:ident] map($expr:expr),
        )*
    ) => {
        backends! {
            $(
                #[$backend_attr]
                {
                    let $backend = $expr;
                    $map
                }
            )*
        }
    };

    // a struct constructor with one field per backend with mapped data
    (
        let map = |$backend:pat| $map:block;
        $Struct:path {
            $(
                #[$backend_attr:ident] $ident:ident : map($expr:expr),
            )*
        }
    ) => {
        backends! {
            $Struct {
                $(
                    #[$backend_attr]
                    $ident: {
                        let $backend = $expr;
                        $map
                    },
                )*
            }
        }
    };
}

#[test]
fn test_backend_macro() {
    struct Foo {
        #[cfg(any(
            windows,
            all(unix, not(any(target_os = "ios", target_os = "macos"))),
            feature = "gfx-backend-vulkan",
        ))]
        vulkan: u32,

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        metal: u32,

        #[cfg(windows)]
        dx12: u32,

        #[cfg(windows)]
        dx11: u32,
    }

    // test struct construction
    let test_foo: Foo = backends_map! {
        let map = |init| { init - 100 };
        Foo {
            #[vulkan] vulkan: map(101),
            #[metal] metal: map(102),
            #[dx12] dx12: map(103),
            #[dx11] dx11: map(104),
        }
    };

    let mut vec = Vec::new();

    // test basic statement-per-backend
    backends_map! {
        let map = |(id, chr)| {
            vec.push((id, chr));
        };

        #[vulkan]
        map((test_foo.vulkan, 'a')),

        #[metal]
        map((test_foo.metal, 'b')),

        #[dx12]
        map((test_foo.dx12, 'c')),

        #[dx11]
        map((test_foo.dx11, 'd')),
    }

    #[cfg(any(
        windows,
        all(unix, not(any(target_os = "ios", target_os = "macos"))),
        feature = "gfx-backend-vulkan",
    ))]
    assert!(vec.contains(&(1, 'a')));

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    assert!(vec.contains(&(2, 'b')));

    #[cfg(windows)]
    assert!(vec.contains(&(3, 'c')));

    #[cfg(windows)]
    assert!(vec.contains(&(4, 'd')));

    // test complex statement-per-backend
    backends_map! {
        let map = |(id, pred, code)| {
            if pred(id) {
                code();
            }
        };

        #[vulkan]
        map((test_foo.vulkan, |v| v == 1, || println!("vulkan"))),

        #[metal]
        map((test_foo.metal, |v| v == 2, || println!("metal"))),

        #[dx12]
        map((test_foo.dx12, |v| v == 3, || println!("dx12"))),

        #[dx11]
        map((test_foo.dx11, |v| v == 4, || println!("dx11"))),
    }

    // test struct construction 2
    let test_foo_2: Foo = backends! {
        Foo {
            #[vulkan]
            vulkan: 1,

            #[metal]
            metal: 2,

            #[dx12]
            dx12: 3,

            #[dx11]
            dx11: 4,
        }
    };

    backends! {
        #[vulkan]
        let var_vulkan = test_foo_2.vulkan;

        #[metal]
        let var_metal = test_foo_2.metal;

        #[dx12]
        let var_dx12 = test_foo_2.dx12;

        #[dx11]
        let var_dx11 = test_foo_2.dx11;
    }

    backends_map! {
        let map = |(id, chr, var)| { (chr, id, var) };

        #[vulkan]
        let var_vulkan = map((test_foo_2.vulkan, 'a', var_vulkan));

        #[metal]
        let var_metal = map((test_foo_2.metal, 'b', var_metal));

        #[dx12]
        let var_dx12 = map((test_foo_2.dx12, 'c', var_dx12));

        #[dx11]
        let var_dx11 = map((test_foo_2.dx11, 'd', var_dx11));
    }

    backends! {
        #[vulkan]
        {
            println!("backend int: {:?}", var_vulkan);
        }

        #[metal]
        {
            println!("backend int: {:?}", var_metal);
        }

        #[dx12]
        {
            println!("backend int: {:?}", var_dx12);
        }

        #[dx11]
        {
            println!("backend int: {:?}", var_dx11);
        }
    }

    #[cfg(any(
        windows,
        all(unix, not(any(target_os = "ios", target_os = "macos"))),
        feature = "gfx-backend-vulkan",
    ))]
    let _ = var_vulkan;

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    let _ = var_metal;

    #[cfg(windows)]
    let _ = var_dx12;

    #[cfg(windows)]
    let _ = var_dx11;
}
