CC=gcc
CFLAGS=-I.
DEPS=./wgpu-bindings/wgpu.h
OUTDIR=./build
LINK_ARGS=-L ../target/debug -lwgpu_native

%.o: %.c $(DEPS)
	$(CC) $(LINK_ARGS) -c -o $(OUTDIR)/$@ $< $(CFLAGS)

hello_triangle_c: hello_triangle_c/main.c
	mkdir -p $(OUTDIR)
	$(CC) $(LINK_ARGS) -o $(OUTDIR)/$@ $^ $(CFLAGS)

.PHONY: all

all: hello_triangle_c
