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


.PHONY: all check test doc clear lib-native lib-rust examples-native examples-rust gfx

all: examples-native examples-rust examples-gfx

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

lib-rust: Cargo.lock wgpu-rs/Cargo.toml $(wildcard wgpu-rs/**/*.rs)
	cargo build --manifest-path wgpu-rs/Cargo.toml --features $(FEATURE_RUST)

wgpu-bindings/*.h: Cargo.lock wgpu-bindings/src/*.rs lib-native
	cargo +nightly-2018-12-27 run --manifest-path wgpu-bindings/Cargo.toml

examples-native: lib-native wgpu-bindings/wgpu.h $(wildcard wgpu-native/**/*.c)
	#$(MAKE) -C examples

examples-rust: lib-rust examples/Cargo.toml $(wildcard wgpu-native/**/*.rs)
	cargo build --manifest-path examples/Cargo.toml --features $(FEATURE_RUST)

examples-gfx: lib-rust gfx-examples/Cargo.toml $(wildcard gfx-examples/*.rs)
	cargo build --manifest-path gfx-examples/Cargo.toml --features $(FEATURE_RUST)

gfx:
	cargo run --manifest-path gfx-examples/Cargo.toml --bin $(name) --features $(FEATURE_RUST)
