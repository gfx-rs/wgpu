RUST_BACKTRACE:=1
EXCLUDES:=
FEATURE_RUST:=
FEATURE_NATIVE:=

ifeq (,$(TARGET))
	CHECK_TARGET_FLAG=
else
	CHECK_TARGET_FLAG=--target $(TARGET)
endif

ifeq ($(OS),Windows_NT)
	ifeq ($(TARGET),x86_64-pc-windows-gnu)
		FEATURE_RUST=vulkan
		FEATURE_NATIVE=gfx-backend-vulkan
	else
		FEATURE_RUST=dx12
		FEATURE_NATIVE=gfx-backend-dx12
	endif
else
	UNAME_S:=$(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		FEATURE_RUST=vulkan
		FEATURE_NATIVE=gfx-backend-vulkan
	endif
	ifeq ($(UNAME_S),Darwin)
		FEATURE_RUST=metal
		FEATURE_NATIVE=gfx-backend-metal
	endif
endif


.PHONY: all check test doc clear lib-native lib-remote examples-native examples-remote

all: examples-native examples-remote

check:
	cargo check --all

test:
	cargo test --all --features "$(FEATURE_NATIVE) $(FEATURE_RUST)"

doc:
	cargo doc --all

clear:
	cargo clean
	rm wgpu-bindings/wgpu.h wgpu-bindings/wgpu-remote.h

lib-native: Cargo.lock wgpu-native/Cargo.toml $(wildcard wgpu-native/**/*.rs)
	cargo build --manifest-path wgpu-native/Cargo.toml --features "local,$(FEATURE_NATIVE)"

lib-remote: Cargo.lock wgpu-remote/Cargo.toml $(wildcard wgpu-native/**/*.rs wgpu-remote/**/*.rs)
	cargo build --manifest-path wgpu-remote/Cargo.toml --features $(FEATURE_RUST)

ffi/wgpu.h: wgpu-native/cbindgen.toml $(wildcard wgpu-native/**/*.rs)
	rustup run nightly cbindgen wgpu-native >ffi/wgpu.h

ffi/wgpu-remote.h:  wgpu-remote/cbindgen.toml $(wildcard wgpu-native/**/*.rs wgpu-remote/**/*.rs)
	rustup run nightly cbindgen wgpu-remote >ffi/wgpu-remote.h

examples-native: lib-native ffi/wgpu.h examples/hello_triangle_c/main.c
	cd examples/hello_triangle_c && mkdir -p build && cd build && cmake .. && make

examples-remote: lib-remote ffi/wgpu-remote.h examples/hello_remote_c/main.c
	cd examples/hello_remote_c && mkdir -p build && cd build && cmake .. && make
