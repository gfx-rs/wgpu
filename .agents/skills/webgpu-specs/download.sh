#!/bin/sh

set -e

TARGET_DIR="$(cargo metadata --format-version 1 | jq -r ".target_directory")"

WEBGPU="$TARGET_DIR/claude/webgpu-spec"
WGSL="$TARGET_DIR/claude/wgsl-spec"
mkdir -p "$TARGET_DIR/claude"

if [ -f "$WEBGPU.etag" ]; then
    curl --etag-save "$WEBGPU.etag.new" --etag-compare "$WEBGPU.etag" -fsSL https://raw.githubusercontent.com/gpuweb/gpuweb/main/spec/index.bs -o "$WEBGPU.bs"
    [ -s "$WEBGPU.etag.new" ] && mv "$WEBGPU.etag.new" "$WEBGPU.etag" || rm "$WEBGPU.etag.new"
else
    curl --etag-save "$WEBGPU.etag" https://raw.githubusercontent.com/gpuweb/gpuweb/main/spec/index.bs -o "$WEBGPU.bs"
fi

if [ -f "$WGSL.etag" ]; then
    curl --etag-save "$WGSL.etag.new" --etag-compare "$WGSL.etag" -fsSL https://raw.githubusercontent.com/gpuweb/gpuweb/main/wgsl/index.bs -o "$WGSL.bs"
    [ -s "$WGSL.etag.new" ] && mv "$WGSL.etag.new" "$WGSL.etag" || rm "$WGSL.etag.new"
else
    curl --etag-save "$WGSL.etag" https://raw.githubusercontent.com/gpuweb/gpuweb/main/wgsl/index.bs -o "$WGSL.bs"
fi
