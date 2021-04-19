.PHONY: all clean validate-spv validate-msl validate-glsl validate-dot validate-wgsl
.SECONDARY: boids.metal quad.metal
SNAPSHOTS_IN=tests/in
SNAPSHOTS_OUT=tests/out

all:
	cargo fmt
	cargo test --all-features
	cargo clippy --all-features

clean:
	rm *.metal *.air *.metallib *.vert *.frag *.comp *.spv

%.metal: $(SNAPSHOTS_IN)/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --features wgsl-in,msl-out -- $< $@

%.air: %.metal
	xcrun -sdk macosx metal -c $< -mmacosx-version-min=10.11

%.metallib: %.air
	xcrun -sdk macosx metallib $< -o $@

%.dot: $(SNAPSHOTS_IN)/%.wgsl $(wildcard src/*.rs src/front/wgsl/*.rs src/back/dot/*.rs bin/convert.rs)
	cargo run --features wgsl-in,dot-out -- $< $@

%.png: %.dot
	dot -Tpng $< -o $@

validate-spv: $(SNAPSHOTS_OUT)/*.spvasm
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/"};	\
		cat $${file} | spirv-as --target-env vulkan1.0 -o - | spirv-val; \
	done

validate-msl: $(SNAPSHOTS_OUT)/*.msl
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/"};	\
		cat $${file} | xcrun -sdk macosx metal -mmacosx-version-min=10.11 -x metal - -o /dev/null; \
	done

validate-glsl: $(SNAPSHOTS_OUT)/*.glsl
	@set -e && for file in $(SNAPSHOTS_OUT)/*.Vertex.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S vert; \
	done
	@set -e && for file in $(SNAPSHOTS_OUT)/*.Fragment.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S frag; \
	done
	@set -e && for file in $(SNAPSHOTS_OUT)/*.Compute.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S comp; \
	done

validate-dot: $(SNAPSHOTS_OUT)/*.dot
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/"};	\
		cat $${file} | dot -o /dev/null; \
	done

validate-wgsl: $(SNAPSHOTS_OUT)/*.wgsl
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/"};	\
		cargo run --bin convert --features wgsl-in $${file} >/dev/null; \
	done
