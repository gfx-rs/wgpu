.PHONY: all clean validate-spv validate-msl validate-glsl validate-dot validate-wgsl validate-hlsl-dxc validate-hlsl-fxc
.SECONDARY: boids.metal quad.metal
SNAPSHOTS_BASE_IN=tests/in
SNAPSHOTS_BASE_OUT=tests/out

all:
	cargo fmt
	cargo test --all-features --workspace
	cargo clippy --all-features --workspace -- -D warnings

clean:
	rm *.metal *.air *.metallib *.vert *.frag *.comp *.spv

bench:
	#rm -Rf target/criterion
	cargo bench

%.metal: $(SNAPSHOTS_BASE_IN)/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --features wgsl-in,msl-out -- $< $@

%.air: %.metal
	xcrun -sdk macosx metal -c $< -mmacosx-version-min=10.11

%.metallib: %.air
	xcrun -sdk macosx metallib $< -o $@

%.dot: $(SNAPSHOTS_BASE_IN)/%.wgsl $(wildcard src/*.rs src/front/wgsl/*.rs src/back/dot/*.rs bin/naga.rs)
	cargo run --features wgsl-in,dot-out -- $< $@

%.png: %.dot
	dot -Tpng $< -o $@

validate-spv: $(SNAPSHOTS_BASE_OUT)/spv/*.spvasm
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};	\
		cat $${file} | spirv-as --target-env vulkan1.0 -o - | spirv-val; \
	done

validate-msl: $(SNAPSHOTS_BASE_OUT)/msl/*.msl
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};	\
		header=$$(head -n1 $${file});	\
		cat $${file} | xcrun -sdk macosx metal -mmacosx-version-min=10.11 -std=macos-$${header:13:8} -x metal - -o /dev/null; \
	done

validate-glsl: $(SNAPSHOTS_BASE_OUT)/glsl/*.glsl
	@set -e && for file in $(SNAPSHOTS_BASE_OUT)/glsl/*.Vertex.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S vert; \
	done
	@set -e && for file in $(SNAPSHOTS_BASE_OUT)/glsl/*.Fragment.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S frag; \
	done
	@set -e && for file in $(SNAPSHOTS_BASE_OUT)/glsl/*.Compute.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S comp; \
	done

validate-dot: $(SNAPSHOTS_BASE_OUT)/dot/*.dot
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};	\
		cat $${file} | dot -o /dev/null; \
	done

validate-wgsl: $(SNAPSHOTS_BASE_OUT)/wgsl/*.wgsl
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};	\
		cargo run $${file}; \
	done

validate-hlsl-dxc: SHELL:=/bin/bash # required because config files uses arrays
validate-hlsl-dxc: $(SNAPSHOTS_BASE_OUT)/hlsl/*.hlsl
	@set -e && for file in $^ ; do \
		DXC_PARAMS="-Wno-parentheses-equality -Zi -Qembed_debug -Od"; \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"}; \
		config="$$(dirname $${file})/$$(basename $${file}).config"; \
		. $${config}; \
		for (( i=0; i<$${#vertex[@]}; i++ )); do \
			name=`echo $${vertex[i]} | cut -d \: -f 1`; \
			profile=`echo $${vertex[i]} | cut -d \: -f 2`; \
			(set -x; dxc $${file} -T $${profile} -E $${name} $${DXC_PARAMS} > /dev/null); \
		done; \
		for (( i=0; i<$${#fragment[@]}; i++ )); do \
			name=`echo $${fragment[i]} | cut -d \: -f 1`; \
			profile=`echo $${fragment[i]} | cut -d \: -f 2`; \
			(set -x; dxc $${file} -T $${profile} -E $${name} $${DXC_PARAMS} > /dev/null); \
		done; \
		for (( i=0; i<$${#compute[@]}; i++ )); do \
			name=`echo $${compute[i]} | cut -d \: -f 1`; \
			profile=`echo $${compute[i]} | cut -d \: -f 2`; \
			(set -x; dxc $${file} -T $${profile} -E $${name} $${DXC_PARAMS} > /dev/null); \
		done; \
		echo "======================"; \
	done

validate-hlsl-fxc: SHELL:=/bin/bash # required because config files uses arrays
validate-hlsl-fxc: $(SNAPSHOTS_BASE_OUT)/hlsl/*.hlsl
	@set -e && for file in $^ ; do \
		FXC_PARAMS="-Zi -Od"; \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"}; \
		config="$$(dirname $${file})/$$(basename $${file}).config"; \
		. $${config}; \
		for (( i=0; i<$${#vertex[@]}; i++ )); do \
			name=`echo $${vertex[i]} | cut -d \: -f 1`; \
			profile=`echo $${vertex[i]} | cut -d \: -f 2`; \
			sm=`echo $${profile} | cut -d \_ -f 2`; \
			if (( sm < 6 )); then \
				(set -x; fxc $${file} -T $${profile} -E $${name} $${FXC_PARAMS} > /dev/null); \
			fi \
		done; \
		for (( i=0; i<$${#fragment[@]}; i++ )); do \
			name=`echo $${fragment[i]} | cut -d \: -f 1`; \
			profile=`echo $${fragment[i]} | cut -d \: -f 2`; \
			sm=`echo $${profile} | cut -d \_ -f 2`; \
			if (( sm < 6 )); then \
				(set -x; fxc $${file} -T $${profile} -E $${name} $${FXC_PARAMS} > /dev/null); \
			fi \
		done; \
		for (( i=0; i<$${#compute[@]}; i++ )); do \
			name=`echo $${compute[i]} | cut -d \: -f 1`; \
			profile=`echo $${compute[i]} | cut -d \: -f 2`; \
			sm=`echo $${profile} | cut -d \_ -f 2`; \
			if (( sm < 6 )); then \
				(set -x; fxc $${file} -T $${profile} -E $${name} $${FXC_PARAMS} > /dev/null); \
			fi \
		done; \
		echo "======================"; \
	done
