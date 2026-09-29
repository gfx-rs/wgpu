#![doc = crate::macros::doc_image!("xcode-new-project.webp")]
#![doc = crate::macros::doc_image!("xcode-project-type.webp")]
#![doc = crate::macros::doc_image!("xcode-build-tool.webp")]
#![doc = crate::macros::doc_image!("xcode-edit-scheme.webp")]
#![doc = crate::macros::doc_image!("xcode-scheme-options.webp")]
#![doc = crate::macros::doc_image!("xcode-scheme-arguments.webp")]
#![doc = crate::macros::doc_image!("xcode-select-binary.webp")]
#![doc = crate::macros::doc_image!("xcode-play-button.webp")]
#![doc = crate::macros::doc_image!("xcode-attached.webp")]
#![doc = crate::macros::doc_image!("xcode-start-capture.webp")]
/*!
# Debugging with Xcode (Metal GPU Capture)

Xcode provides powerful Metal GPU capture and shader debugging tools. This
page walks through attaching Xcode to a `wgpu` application so you can enable
Metal validation and capture frames.

A Chinese version is also available:
[使用 Xcode 调试 wgpu 程序](https://jinleili.github.io/learn-wgpu-zh/integration-and-debugging/ios/#xcode-%E4%B8%8E-metal).

First we'll start by opening Xcode and creating a new project using the menu
or "Create a new Xcode project" in the startup dialog.

![Xcode startup dialog][xcode-new-project.webp]

Select "External Build System" as the project type.

![Xcode project type][xcode-project-type.webp]

Choose which build tool to use in the "Build Tool" field. This will be
called whenever we build in Xcode.

We can write our `cargo` build command here if we'd like Xcode to run it,
but we could also skip the build step to avoid Xcode building the project
for us.

It's possible to skip the build step by specifying our build command as
`ls`, `echo`, or anything else that exits successfully (note: `:` doesn't
appear to work here).

The rest of the fields don't actually matter to us, so we can put any values
there.

![Xcode build tool][xcode-build-tool.webp]

Click on the project name and then click "Edit Scheme".

![Xcode edit scheme][xcode-edit-scheme.webp]

Under the "Options" tab we can choose the level of validation for Metal and
whether to enable GPU frame capture.

![Xcode scheme options][xcode-scheme-options.webp]

Under the "Arguments" tab we'll choose which executable to run. We want this
to be the binary created by `cargo`. We'll select "Other" and then find the
binary in the target directory.

![Xcode scheme arguments][xcode-scheme-arguments.webp]

![Xcode select binary][xcode-select-binary.webp]

Next we'll click the play button to run our binary and attach Xcode to it.

![Xcode play button][xcode-play-button.webp]

Now we should see our application running and some output telling us that
Metal validation has been enabled.

![Xcode attached][xcode-attached.webp]

To start a GPU capture, click the camera button while Xcode is attached to
our running application.

![Xcode start capture][xcode-start-capture.webp]

After a frame has been captured, we will be able to use all the regular
Metal tools (e.g. shader debugging, GPU statistics, etc.).

*/
