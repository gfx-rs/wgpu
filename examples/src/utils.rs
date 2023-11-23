#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Replaces the site body with a message telling the user to open the console and use that.
#[cfg(target_arch = "wasm32")]
pub fn add_web_nothing_to_see_msg() {
    web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.body())
        .expect("Could not get document / body.")
        .set_inner_html("<h1>Nothing to see here! Open the console!</h1>");
}

/// Outputs a vector of RGBA bytes as a png image with the given dimensions on the given path.
#[cfg(not(target_arch = "wasm32"))]
pub fn output_image_native(image_data: Vec<u8>, texture_dims: (usize, usize), path: String) {
    let mut png_data = Vec::<u8>::with_capacity(image_data.len());
    let mut encoder = png::Encoder::new(
        std::io::Cursor::new(&mut png_data),
        texture_dims.0 as u32,
        texture_dims.1 as u32,
    );
    encoder.set_color(png::ColorType::Rgba);
    let mut png_writer = encoder.write_header().unwrap();
    png_writer.write_image_data(&image_data[..]).unwrap();
    png_writer.finish().unwrap();
    log::info!("PNG file encoded in memory.");

    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&png_data[..]).unwrap();
    log::info!("PNG file written to disc as \"{}\".", path);
}

/// Effectively a version of `output_image_native` but meant for web browser contexts.
///
/// This is achieved via in `img` element on the page. If the target image element does
/// not exist, this function creates one. If it does, the image data is overridden.
///
/// This function makes use of a hidden staging canvas which the data is copied to in
/// order to create a data URL.
#[cfg(target_arch = "wasm32")]
pub fn output_image_wasm(image_data: Vec<u8>, texture_dims: (usize, usize)) {
    let document = web_sys::window().unwrap().document().unwrap();
    let body = document.body().unwrap();

    let canvas = if let Some(found_canvas) = document.get_element_by_id("staging-canvas") {
        match found_canvas.dyn_into::<web_sys::HtmlCanvasElement>() {
            Ok(canvas_as_canvas) => canvas_as_canvas,
            Err(e) => {
                log::error!(
                    "In searching for a staging canvas for outputting an image \
                    (element with id \"staging-canvas\"), found non-canvas element: {:?}.
                    Replacing with standard staging canvas.",
                    e
                );
                e.remove();
                create_staging_canvas(&document)
            }
        }
    } else {
        log::info!("Output image staging canvas element not found; creating.");
        create_staging_canvas(&document)
    };
    // Having the size attributes the right size is so important, we should always do it
    // just to be safe. Also, what if we might want the image size to be able to change?
    let image_dimension_strings = (texture_dims.0.to_string(), texture_dims.1.to_string());
    canvas
        .set_attribute("width", image_dimension_strings.0.as_str())
        .unwrap();
    canvas
        .set_attribute("height", image_dimension_strings.1.as_str())
        .unwrap();

    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();
    let image_data = web_sys::ImageData::new_with_u8_clamped_array(
        wasm_bindgen::Clamped(&image_data),
        texture_dims.0 as u32,
    )
    .unwrap();
    context.put_image_data(&image_data, 0.0, 0.0).unwrap();

    // Get the img element that will act as our target for rendering from the canvas.
    let image_element = if let Some(found_image_element) =
        document.get_element_by_id("output-image-target")
    {
        match found_image_element.dyn_into::<web_sys::HtmlImageElement>() {
            Ok(e) => e,
            Err(e) => {
                log::error!(
                    "Found an element with the id \"output-image-target\" but it was not an image: {:?}.
                    Replacing with default image output element.",
                    e
                );
                e.remove();
                create_output_image_element(&document)
            }
        }
    } else {
        log::info!("Output image element not found; creating.");
        create_output_image_element(&document)
    };
    // The canvas is currently the image we ultimately want. We can create a data url from it now.
    let data_url = canvas.to_data_url().unwrap();
    image_element.set_src(&data_url);
    log::info!("Copied image from staging canvas to image element.");

    if document.get_element_by_id("image-for-you-text").is_none() {
        log::info!("\"Image for you\" text not found; creating.");
        let p = document
            .create_element("p")
            .expect("Failed to create p element for \"image for you text\".");
        p.set_text_content(Some(
            "The above image is for you!
        You can drag it to your desktop to download.",
        ));
        p.set_id("image-for-you-text");
        body.append_child(&p)
            .expect("Failed to append \"image for you text\" to document body.");
    }
}

#[cfg(target_arch = "wasm32")]
fn create_staging_canvas(document: &web_sys::Document) -> web_sys::HtmlCanvasElement {
    let body = document.body().expect("Failed to get document body.");
    let new_canvas = document
        .create_element("canvas")
        .expect("Failed to create staging canvas.")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    // We don't want to show the canvas, we just want it to exist in the background.
    new_canvas.set_attribute("hidden", "true").unwrap();
    new_canvas.set_attribute("background-color", "red").unwrap();
    body.append_child(&new_canvas).unwrap();
    log::info!("Created new staging canvas: {:?}", &new_canvas);
    new_canvas
}

#[cfg(target_arch = "wasm32")]
fn create_output_image_element(document: &web_sys::Document) -> web_sys::HtmlImageElement {
    let body = document.body().expect("Failed to get document body.");
    let new_image = document
        .create_element("img")
        .expect("Failed to create output image element.")
        .dyn_into::<web_sys::HtmlImageElement>()
        .unwrap();
    new_image.set_id("output-image-target");
    body.append_child(&new_image)
        .expect("Failed to append output image target to document body.");
    log::info!("Created new output target image: {:?}", &new_image);
    new_image
}
