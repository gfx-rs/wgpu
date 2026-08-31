/*!
# Integrating wgpu into Existing iOS and Android Apps

`wgpu` does not depend on any windowing library, so it does not provide
window creation and management functions. Only when creating a window-based
[`Surface`] may an argument that implements the
[`raw-window-handle`](https://github.com/rust-windowing/raw-window-handle)
abstract interface be required. The
[`winit`](https://github.com/rust-windowing/winit) crate used by the `wgpu`
examples is a cross-platform window creation and management crate that
implements that interface; it takes over the window management and event
loop of the entire app.

There is no doubt that, for game apps, the combination of `wgpu` + `winit`
is very suitable. However, a large number of non-game apps also often need
to use graphics APIs (for charts, image filters, etc.). These apps need to
use a lot of system UI components and interactions, and `winit`'s way of
taking over the entire app window is not suitable. So, it would be very
useful to integrate `wgpu` into existing iOS and Android apps without using
third-party window-management libraries.

[wgpu-in-app](https://github.com/jinleili/wgpu-in-app) is a complete doc and
example of how to do it. The doc section also has two Chinese versions:
[与 iOS App 集成](https://jinleili.github.io/learn-wgpu-zh/integration-and-debugging/ios/)
and
[与 Android App 集成](https://jinleili.github.io/learn-wgpu-zh/integration-and-debugging/android/).
*/

use crate::Surface;
