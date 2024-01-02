struct ExampleDesc {
    name: &'static str,
    function: fn(),
    #[allow(dead_code)] // isn't used on native
    webgl: bool,
    #[allow(dead_code)] // isn't used on native
    webgpu: bool,
}

const EXAMPLES: &[ExampleDesc] = &[
    ExampleDesc {
        name: "boids",
        function: wgpu_examples::boids::main,
        webgl: false, // No compute
        webgpu: true,
    },
    ExampleDesc {
        name: "bunnymark",
        function: wgpu_examples::bunnymark::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "conservative_raster",
        function: wgpu_examples::conservative_raster::main,
        webgl: false,  // No conservative raster
        webgpu: false, // No conservative raster
    },
    ExampleDesc {
        name: "cube",
        function: wgpu_examples::cube::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "hello",
        function: wgpu_examples::hello::main,
        webgl: false, // No canvas for WebGL
        webgpu: true,
    },
    ExampleDesc {
        name: "hello_compute",
        function: wgpu_examples::hello_compute::main,
        webgl: false, // No compute
        webgpu: true,
    },
    ExampleDesc {
        name: "hello_synchronization",
        function: wgpu_examples::hello_synchronization::main,
        webgl: false, // No canvas for WebGL
        webgpu: true,
    },
    ExampleDesc {
        name: "hello_triangle",
        function: wgpu_examples::hello_triangle::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "hello_windows",
        function: wgpu_examples::hello_windows::main,
        webgl: false,  // Native only example
        webgpu: false, // Native only example
    },
    ExampleDesc {
        name: "hello_workgroups",
        function: wgpu_examples::hello_workgroups::main,
        webgl: false,
        webgpu: true,
    },
    ExampleDesc {
        name: "mipmap",
        function: wgpu_examples::mipmap::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "msaa_line",
        function: wgpu_examples::msaa_line::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "render_to_texture",
        function: wgpu_examples::render_to_texture::main,
        webgl: false, // No canvas for WebGL
        webgpu: true,
    },
    ExampleDesc {
        name: "repeated_compute",
        function: wgpu_examples::repeated_compute::main,
        webgl: false, // No compute
        webgpu: true,
    },
    ExampleDesc {
        name: "shadow",
        function: wgpu_examples::shadow::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "skybox",
        function: wgpu_examples::skybox::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "srgb_blend",
        function: wgpu_examples::srgb_blend::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "stencil_triangles",
        function: wgpu_examples::stencil_triangles::main,
        webgl: true,
        webgpu: true,
    },
    ExampleDesc {
        name: "storage_texture",
        function: wgpu_examples::storage_texture::main,
        webgl: false, // No storage textures
        webgpu: true,
    },
    ExampleDesc {
        name: "texture_arrays",
        function: wgpu_examples::texture_arrays::main,
        webgl: false,  // No texture arrays
        webgpu: false, // No texture arrays
    },
    ExampleDesc {
        name: "timestamp_queries",
        function: wgpu_examples::timestamp_queries::main,
        webgl: false,  // No canvas for WebGL
        webgpu: false, // No timestamp queries
    },
    ExampleDesc {
        name: "uniform_values",
        function: wgpu_examples::uniform_values::main,
        webgl: false, // No compute
        webgpu: true,
    },
    ExampleDesc {
        name: "water",
        function: wgpu_examples::water::main,
        webgl: false, // No RODS
        webgpu: true,
    },
];

fn get_example_name() -> Option<String> {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let query_string = web_sys::window()?.location().search().ok()?;

            wgpu_examples::framework::parse_url_query_string(&query_string, "example").map(String::from)
        } else {
            std::env::args().nth(1)
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn print_examples() {
    // Get the document, header, and body elements.
    let document = web_sys::window().unwrap().document().unwrap();

    for backend in ["webgl2", "webgpu"] {
        let ul = document
            .get_element_by_id(&format!("{backend}-list"))
            .unwrap();

        for example in EXAMPLES {
            if backend == "webgl2" && !example.webgl {
                continue;
            }
            if backend == "webgpu" && !example.webgpu {
                continue;
            }

            let link = document.create_element("a").unwrap();
            link.set_text_content(Some(example.name));
            link.set_attribute(
                "href",
                &format!("?backend={backend}&example={}", example.name),
            )
            .unwrap();
            link.set_class_name("example-link");

            let item = document.create_element("div").unwrap();
            item.append_child(&link).unwrap();
            item.set_class_name("example-item");
            ul.append_child(&item).unwrap();
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn print_unknown_example(_result: Option<String>) {}

#[cfg(not(target_arch = "wasm32"))]
fn print_unknown_example(result: Option<String>) {
    if let Some(example) = result {
        println!("Unknown example: {}", example);
    } else {
        println!("Please specify an example as the first argument!");
    }

    println!("\nAvailable Examples:");
    for examples in EXAMPLES {
        println!("\t{}", examples.name);
    }
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    print_examples();

    let Some(example) = get_example_name() else {
        print_unknown_example(None);
        return;
    };

    let Some(found) = EXAMPLES.iter().find(|e| e.name == example) else {
        print_unknown_example(Some(example));
        return;
    };

    (found.function)();
}
