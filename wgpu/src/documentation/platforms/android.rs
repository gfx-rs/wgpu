/*!
# Running on Android

## Surface creation

On Android, surface creation should happen on the resume event of the
window, and surface cleanup on the suspend event. A valid surface cannot be
created beforehand. This means that, when using `winit`, `sdl2`, or anything
similar, surface creation should happen in the resume handler, roughly like
this:

```ignore
// winit's `ApplicationHandler::resumed`:
fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    let window = event_loop.create_window(Window::default_attributes()).unwrap();
    let surface = instance.create_surface(window).unwrap();
    // ...
}
```

Correspondingly, drop the [`Surface`] when the window is suspended.
*/

use crate::Surface;
