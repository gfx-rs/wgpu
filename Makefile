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


.PHONY: all check test doc clear lib-native lib-remote lib-rust examples-rust examples-gfx gfx ffi-examples

all: examples-rust examples-gfx ffi-examples

check:
	cargo check --all

test:
	cargo test --all --features "$(FEATURE_NATIVE) $(FEATURE_RUST)"

doc:
	cargo doc --all

clear:
	cargo clean
	rm wgpu-bindings/wgpu.h

lib-native: Cargo.lock wgpu-native/Cargo.toml $(wildcard wgpu-native/**/*.rs)
	cargo build --manifest-path wgpu-native/Cargo.toml --features "local,$(FEATURE_NATIVE)"

lib-remote: Cargo.lock wgpu-remote/Cargo.toml $(wildcard wgpu-native/**/*.rs wgpu-remote/**/*.rs)
	cargo build --manifest-path wgpu-remote/Cargo.toml --features $(FEATURE_RUST)

lib-rust: Cargo.lock wgpu-rs/Cargo.toml $(wildcard wgpu-rs/**/*.rs)
	cargo build --manifest-path wgpu-rs/Cargo.toml --features $(FEATURE_RUST)

ffi/wgpu.h: wgpu-native/cbindgen.toml $(wildcard wgpu-native/**/*.rs)
	cbindgen wgpu-native >ffi/wgpu.h

ffi/wgpu-remote.h:  wgpu-remote/cbindgen.toml $(wildcard wgpu-native/**/*.rs wgpu-remote/**/*.rs)
	cbindgen wgpu-remote >ffi/wgpu-remote.h

wgpu-bindings/*.h: Cargo.lock $(wildcard wgpu-bindings/src/*.rs) lib-native lib-remote
	cargo +nightly run --manifest-path wgpu-bindings/Cargo.toml

ffi-examples: lib-native lib-remote ffi/wgpu.h ffi/wgpu-remote.h
	cd examples/hello_triangle_c && mkdir -p build && cd build && cmake .. && make
	cd examples/hello_remote_c && mkdir -p build && cd build && cmake .. && make

examples-rust: examples/Cargo.toml $(wildcard wgpu-native/**/*.rs wgpu-rs/**/*.rs)
	cargo build --manifest-path examples/Cargo.toml --features $(FEATURE_RUST)

examples-gfx: examples-rust gfx-examples/Cargo.toml $(wildcard gfx-examples/*.rs)
	cargo build --manifest-path gfx-examples/Cargo.toml --features $(FEATURE_RUST)

gfx:
	cargo run --manifest-path gfx-examples/Cargo.toml --bin $(name) --features $(FEATURE_RUST)
