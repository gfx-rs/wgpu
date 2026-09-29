# Running the tests on Android and iOS

An approach of cross-compiling the binaries on a host, then using various tools such as `adb` and `ssh` to transfer the test binaries over to run them, can be used.

Below are example configuration and runner scripts that can be used to perform this yourself.

## Android

<details>
<summary>.cargo/config.toml</summary>

```toml
[target.aarch64-linux-android]
# Runner script is written in powershell
runner = ["pwsh", "-File", "<runner location>run-on-android.ps1"]
rustflags = [
    "-C", "linker=clang",
    "-C", "link-arg=-fuse-ld=lld",
    "-C", "link-arg=--target=aarch64-linux-android",
    "-C", "link-arg=--sysroot=<ndk/sysroot location>",
    "-C", "link-arg=-B<ndk/sysroot location>/usr/lib/aarch64-linux-android/<version>",
    "-C", "link-arg=-L<ndk/sysroot location>/usr/lib/aarch64-linux-android/<version>",
    "-C", "link-arg=-L<ndk/sysroot location>/usr/lib/aarch64-linux-android",
    "-C", "link-arg=-L<ndk location>/toolchains/llvm/prebuilt/<host>/lib/clang/20/lib/linux/aarch64"
]
```

</details>

<details>
<summary>run-on-android.ps1</summary>

```powershell
param(
    [string]$BinaryPath
)

$ErrorActionPreference = "Stop"

# /data/local/tmp is the most common directory where arbitrary binaries can be run
$RemoteDir  = "/data/local/tmp/runner"
$BinaryName = Split-Path -Leaf $BinaryPath
$RemotePath = "$RemoteDir/$BinaryName"

adb push --sync "$BinaryPath" "$RemotePath" *>$null

adb shell "chmod 755 $RemotePath"

$EscapedArgs = $args | ForEach-Object {
    if ($_ -match " ") {
        "`"$_`""
    } else {
        $_
    }
}

$LinuxArgs = $EscapedArgs -join " "
# Set LD_LIBRARY_PATH to the location of libc++_shared.so
$DeviceCmd = "cd $RemoteDir && LD_LIBRARY_PATH=$RemoteDir ./$BinaryName $LinuxArgs"

adb shell "$DeviceCmd"
exit $LASTEXITCODE
```

</details>

All Android binaries link to `libc++_shared.so`. Remember to put a copy alongside the binaries.

## iOS

Apple does not normally provide shell access and `sshd` on their operating system, so jailbreaking the system is required.

Only a copy of the iOS sysroot is required to cross-compile, so Linux and Windows can also try this.

<details>
<summary>.cargo/config.toml</summary>

```toml
[target.aarch64-apple-ios]
# Runner script is written in powershell
runner = ["pwsh", "-File", "<runner location>run-on-ios.ps1"]
rustflags = [
    "-C", "linker=clang",
    "-C", "link-arg=-fuse-ld=lld",
    "-C", "link-arg=--target=aarch64-apple-ios",
    "-C", "link-arg=--sysroot=<sysroot location>",
    "-C", "link-arg=-miphoneos-version-min=<minversion>",
    "-C", "link-arg=-rpath",
    "-C", "link-arg=@executable_path/Frameworks",
    "-Lnative=<sysroot location>/usr/lib",
    "-Lframework=<sysroot location>/System/Library/Frameworks",
]
```

</details>

If you're on a platform other than macOS, set an environment variable of `SDKROOT` to your sysroot location.

This avoids `cc-rs` trying to run `xcrun`.

<details>
<summary>run-on-ios.ps1</summary>

```powershell
param(
    [string]$BinaryPath
)

$SshTarget = "<target user>@<target host>"
$BinaryName = Split-Path -Leaf $BinaryPath
# Select one of the following paths depending on rootless or rootful jailbreak
$RemoteDir  = "/var/jb/var/mobile/runner"
#$RemoteDir  = "/var/root/runner"
$RemotePath = "$RemoteDir/$BinaryName"

$EscapedArgs = $args | ForEach-Object {
    if ($_ -match " ") {
        "`"$_`""
    } else {
        $_
    }
}

$LinuxArgs = $EscapedArgs -join " "
$DeviceCmd = "./$BinaryName $LinuxArgs"

ssh $SshTarget "test -f $RemotePath"

if ($LASTEXITCODE -ne 0) {
    scp -q "$BinaryPath" "$SshTarget`:$RemotePath"
    $DeviceCmd = "chmod 755 $BinaryName && ldid -Sent.xml $BinaryName && $DeviceCmd"
}

ssh $SshTarget "cd $RemoteDir && $DeviceCmd"
exit $LASTEXITCODE
```

</details>

Due to iOS security measures, a binary needs to contain specific entitlements to run without a container and access IOKit, so it is required to have the `ldid` tool installed on your target device to set the entitlements with `ent.xml`. It should be placed in your runner directory.

<details>
<summary>ent.xml</summary>

```xml
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>platform-application</key>
    <true/>
    <key>com.apple.private.security.container-required</key>
    <false/>
    <key>com.apple.security.iokit-user-client-class</key>
    <array>
        <string>AGXCommandQueue</string>
        <string>AGXDevice</string>
        <string>AGXDeviceUserClient</string>
        <string>AGXSharedUserClient</string>
        <string>IOSurfaceRootUserClient</string>
    </array>
</dict>
</plist>
```

</details>
