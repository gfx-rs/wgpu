/*!
# Running on ANGLE

[ANGLE](https://angleproject.org) is a translation layer from GLES to other
backends, developed by Google. We support running our GLES3 backend over it.

How you select ANGLE depends on the platform:

- On Apple platforms, enable the `angle` feature. This is the only way to enable
  the GLES backend there.
- On Windows, build with `--cfg windows_angle`. Without it, the GLES backend uses
  WGL instead of ANGLE's EGL.
- On Linux, the GLES backend already uses EGL, so it loads ANGLE's `libEGL` with
  no extra configuration.

You must also place the ANGLE libraries in a location visible to the
application. These binaries can be downloaded from
[gfbuild-angle](https://github.com/google/gfbuild-angle) artifacts;
[manual compilation](https://github.com/google/angle/blob/main/doc/DevSetup.md)
may be required on Macs with Apple silicon.

On Windows, you generally need to copy the libraries into the working
directory, into the same directory as the executable, or somewhere in your
`PATH`. On Linux, you can point to them using the `LD_LIBRARY_PATH`
environment variable.
*/
