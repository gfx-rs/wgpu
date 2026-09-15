/*!
# Running on ANGLE

[ANGLE](https://angleproject.org) is a translation layer from GLES to other
backends, developed by Google. We support running our GLES3 backend over it
in order to reach platforms with D3D11 support, which aren't accessible
otherwise.

In order to run with ANGLE, the `angle` feature has to be enabled, and ANGLE
libraries placed in a location visible to the application. These binaries
can be downloaded from
[gfbuild-angle](https://github.com/DileSoft/gfbuild-angle) artifacts;
[manual compilation](https://github.com/google/angle/blob/main/doc/DevSetup.md)
may be required on Macs with Apple silicon.

On Windows, you generally need to copy the libraries into the working
directory, into the same directory as the executable, or somewhere in your
`PATH`. On Linux, you can point to them using the `LD_LIBRARY_PATH`
environment variable.
*/
