const EXAMPLES: &[(&str, fn())] = &[
    ("boids", wgpu_examples::boids::main),
    ("bunnymark", wgpu_examples::bunnymark::main),
    (
        "conservative_raster",
        wgpu_examples::conservative_raster::main,
    ),
    ("cube", wgpu_examples::cube::main),
    ("hello", wgpu_examples::hello::main),
    ("hello_compute", wgpu_examples::hello_compute::main),
    (
        "hello_synchronization",
        wgpu_examples::hello_synchronization::main,
    ),
    ("hello_triangle", wgpu_examples::hello_triangle::main),
    ("hello_windows", wgpu_examples::hello_windows::main),
    ("hello_workgroups", wgpu_examples::hello_workgroups::main),
    ("mipmap", wgpu_examples::mipmap::main),
    ("msaa_line", wgpu_examples::msaa_line::main),
    ("render_to_texture", wgpu_examples::render_to_texture::main),
    ("repeated_compute", wgpu_examples::repeated_compute::main),
    ("shadow", wgpu_examples::shadow::main),
    ("skybox", wgpu_examples::skybox::main),
    ("srgb_blend", wgpu_examples::srgb_blend::main),
    ("stencil_triangles", wgpu_examples::stencil_triangles::main),
    ("storage_texture", wgpu_examples::storage_texture::main),
    ("texture_arrays", wgpu_examples::texture_arrays::main),
    ("timestamp_queries", wgpu_examples::timestamp_queries::main),
    ("uniform_values", wgpu_examples::uniform_values::main),
    ("water", wgpu_examples::water::main),
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

fn print_unknown_example(result: Option<String>) {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use wasm_bindgen::JsCast;
            use web_sys::HtmlStyleElement;

            // Get the document, header, and body elements.
            let document = web_sys::window().unwrap().document().unwrap();
            let head = document.head().unwrap();
            let body = document.body().unwrap();

            // Add a basic style sheet to center the text and remove some margin.
            let style_sheet: HtmlStyleElement = document.create_element("style").unwrap().dyn_into().unwrap();
            style_sheet.set_inner_text("div { text-align: center; } p { margin: 4px }");
            head.append_child(&style_sheet).unwrap();

            // A div to provide a container and some padding.
            let div = document.create_element("div").unwrap();
            body.append_child(&div).unwrap();

            let user_message = if let Some(example) = result {
                format!("Unknown example: {example}. Please choose an example!")
            } else {
                String::from("Please choose an example!")
            };

            // A header to display the message to the user.
            let header = document.create_element("h1").unwrap();
            header.set_text_content(Some(&user_message));
            div.append_child(&header).unwrap();

            // Write a link for each example, wrapped in a paragraph.
            for (name, _) in EXAMPLES {
                let paragraph = document.create_element("p").unwrap();
                let link = document.create_element("a").unwrap();
                link.set_text_content(Some(name));
                link.set_attribute("href", &format!("?example={name}")).unwrap();
                paragraph.append_child(&link).unwrap();
                div.append_child(&paragraph).unwrap();
            }
        } else {
            if let Some(example) = result {
                println!("Unknown example: {}", example);
            } else {
                println!("Please specify an example as the first argument!");
            }

            println!("\nAvailable Examples:");
            for (name, _) in EXAMPLES {
                println!("\t{name}");
            }
        }
    }
}

fn main() {
    let Some(example) = get_example_name() else {
        print_unknown_example(None);
        return;
    };

    let Some((_, function)) = EXAMPLES.iter().find(|(name, _)| *name == example) else {
        print_unknown_example(Some(example));
        return;
    };

    function();
}
