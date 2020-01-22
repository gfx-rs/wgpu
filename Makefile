RUST_BACKTRACE:=1
EXCLUDES:=

GENERATOR_PLATFORM:=

FFI_DIR:=ffi
BUILD_DIR:=build
CLEAN_FFI_DIR:=
CREATE_BUILD_DIR:=

WILDCARD_WGPU_NATIVE:=$(wildcard wgpu-native/**/*.rs wgpu-core/**/*.rs)
WILDCARD_WGPU_REMOTE:=$(wildcard wgpu-remote/**/*.rs wgpu-core/**/*.rs)

GIT_TAG=$(shell git describe --abbrev=0 --tags)
GIT_TAG_FULL=$(shell git describe --tags)
OS_NAME=

ifeq (,$(TARGET))
	CHECK_TARGET_FLAG=
else
	CHECK_TARGET_FLAG=--target $(TARGET)
endif

ifeq ($(OS),Windows_NT)
	CLEAN_FFI_DIR=del $(FFI_DIR)\*.* /Q /S
	CREATE_BUILD_DIR=mkdir $(BUILD_DIR)
	GENERATOR_PLATFORM=-DCMAKE_GENERATOR_PLATFORM=x64
else
	CLEAN_FFI_DIR=rm $(FFI_DIR)/**
	CREATE_BUILD_DIR=mkdir -p $(BUILD_DIR)
endif

ifeq ($(OS),Windows_NT)
	LIB_EXTENSION=dll
	OS_NAME=windows
	ZIP_TOOL=7z
else
	UNAME_S:=$(shell uname -s)
	ZIP_TOOL=zip
	ifeq ($(UNAME_S),Linux)
		LIB_EXTENSION=so
		OS_NAME=linux
	endif
	ifeq ($(UNAME_S),Darwin)
		LIB_EXTENSION=dylib
		OS_NAME=macos
	endif
endif


.PHONY: all check test doc clear \
	example-compute example-triangle example-remote \
	run-example-compute run-example-triangle run-example-remote \
	lib-native lib-native-release \
	lib-remote

#TODO: example-remote
all: example-compute example-triangle lib-remote

package: lib-native lib-native-release
	mkdir -p dist
	echo "$(GIT_TAG_FULL)" > dist/commit-sha
	for RELEASE in debug release; do \
		ARCHIVE=wgpu-$$RELEASE-$(OS_NAME)-$(GIT_TAG).zip; \
		rm -f dist/$$ARCHIVE; \
		if [ $(ZIP_TOOL) = zip ]; then \
			zip -j dist/$$ARCHIVE target/$$RELEASE/libwgpu_*.$(LIB_EXTENSION) ffi/*.h dist/commit-sha; \
		else \
			7z a -tzip dist/$$ARCHIVE ./target/$$RELEASE/wgpu_*.$(LIB_EXTENSION) ./ffi/*.h ./dist/commit-sha; \
		fi; \
	done

check:
	cargo check --all

test:
	cargo test --all

doc:
	cargo doc --all

clear:
	cargo clean
	$(CLEAN_FFI_DIR)

lib-native: Cargo.lock wgpu-native/Cargo.toml $(WILDCARD_WGPU_NATIVE)
	cargo build --manifest-path wgpu-native/Cargo.toml

lib-native-release: Cargo.lock wgpu-native/Cargo.toml $(WILDCARD_WGPU_NATIVE)
	cargo build --manifest-path wgpu-native/Cargo.toml --release

lib-remote: Cargo.lock wgpu-remote/Cargo.toml $(WILDCARD_WGPU_REMOTE)
	cargo build --manifest-path wgpu-remote/Cargo.toml

$(FFI_DIR)/wgpu.h: wgpu-native/cbindgen.toml $(WILDCARD_WGPU_NATIVE)
	rustup run nightly cbindgen -o $(FFI_DIR)/wgpu.h wgpu-native

$(FFI_DIR)/wgpu-remote.h: wgpu-remote/cbindgen.toml $(WILDCARD_WGPU_REMOTE)
	rustup run nightly cbindgen -o $(FFI_DIR)/wgpu-remote.h wgpu-remote

example-compute: lib-native $(FFI_DIR)/wgpu.h examples/compute/main.c
	cd examples/compute && $(CREATE_BUILD_DIR) && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. $(GENERATOR_PLATFORM) && cmake --build .

run-example-compute: example-compute
	cd examples/compute/build && ./compute 1 2 3 4

example-triangle: lib-native $(FFI_DIR)/wgpu.h examples/triangle/main.c
	cd examples/triangle && $(CREATE_BUILD_DIR) && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. $(GENERATOR_PLATFORM) && cmake --build .

run-example-triangle: example-triangle
	cd examples/triangle/build && ./triangle

example-remote: lib-remote $(FFI_DIR)/wgpu-remote.h examples/remote/main.c
	cd examples/remote && $(CREATE_BUILD_DIR) && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. $(GENERATOR_PLATFORM) && cmake --build .

run-example-remote: example-remote
	cd examples/remote/build && ./remote
