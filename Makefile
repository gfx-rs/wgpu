RUST_BACKTRACE:=1
EXCLUDES:=
FEATURE_RUST:=
FEATURE_NATIVE:=

GENERATOR_PLATFORM:=

FFI_DIR:=ffi
BUILD_DIR:=build
CLEAN_FFI_DIR:=
CREATE_BUILD_DIR:=

WILDCARD_WGPU_NATIVE:=$(wildcard wgpu-native/**/*.rs)
WILDCARD_WGPU_NATIVE_AND_REMOTE:=$(wildcard wgpu-native/**/*.rs wgpu-remote/**/*.rs)

ifeq (,$(TARGET))
	CHECK_TARGET_FLAG=
else
	CHECK_TARGET_FLAG=--target $(TARGET)
endif

ifeq ($(OS),Windows_NT)
	CLEAN_FFI_DIR=del $(FFI_DIR)\*.* /Q /S
	CREATE_BUILD_DIR=mkdir $(BUILD_DIR)
	GENERATOR_PLATFORM=-DCMAKE_GENERATOR_PLATFORM=x64

	ifeq ($(TARGET),x86_64-pc-windows-gnu)
		FEATURE_RUST=vulkan
		FEATURE_NATIVE=gfx-backend-vulkan
	else
		FEATURE_RUST=dx12
		FEATURE_NATIVE=gfx-backend-dx12
	endif
else
	CLEAN_FFI_DIR=rm $(FFI_DIR)/**
	CREATE_BUILD_DIR=mkdir -p $(BUILD_DIR)

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

.PHONY: all check test doc clear lib-native lib-remote examples-compute example-triangle example-remote

all: examples-compute example-triangle example-remote

check:
	cargo check --all

test:
	cargo test --all --features "$(FEATURE_NATIVE) $(FEATURE_RUST)"

doc:
	cargo doc --all

clear:
	cargo clean
	$(CLEAN_FFI_DIR)

lib-native: Cargo.lock wgpu-native/Cargo.toml $(WILDCARD_WGPU_NATIVE)
	cargo build --manifest-path wgpu-native/Cargo.toml --features "local,$(FEATURE_NATIVE)"

lib-remote: Cargo.lock wgpu-remote/Cargo.toml $(WILDCARD_WGPU_NATIVE_AND_REMOTE)
	cargo build --manifest-path wgpu-remote/Cargo.toml --features $(FEATURE_RUST)

$(FFI_DIR)/wgpu.h: wgpu-native/cbindgen.toml $(WILDCARD_WGPU_NATIVE)
	rustup run nightly cbindgen wgpu-native > $(FFI_DIR)/wgpu.h

$(FFI_DIR)/wgpu-remote.h: wgpu-remote/cbindgen.toml $(WILDCARD_WGPU_NATIVE_AND_REMOTE)
	rustup run nightly cbindgen wgpu-remote > $(FFI_DIR)/wgpu-remote.h

example-compute: lib-native $(FFI_DIR)/wgpu.h examples/compute/main.c
	cd examples/compute && $(CREATE_BUILD_DIR) && cd build && cmake .. -DBACKEND=$(FEATURE_RUST) $(GENERATOR_PLATFORM) && cmake --build .

example-triangle: lib-native $(FFI_DIR)/wgpu.h examples/triangle/main.c
	cd examples/triangle && $(CREATE_BUILD_DIR) && cd build && cmake .. -DBACKEND=$(FEATURE_RUST) $(GENERATOR_PLATFORM) && cmake --build .

example-remote: lib-remote $(FFI_DIR)/wgpu-remote.h examples/remote/main.c
	cd examples/remote && $(CREATE_BUILD_DIR) && cd build && cmake .. && cmake --build .
