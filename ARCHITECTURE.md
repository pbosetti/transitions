# Architecture

## Purpose

`transitions` builds a single composite image from an ordered sequence of source images of identical size. The composite is formed by taking one evenly sized slice from each input frame and concatenating those slices into the final output. This makes the tool suitable for visualizing gradual change across time, motion, or camera movement.

The software has two operating modes:

- A direct slicing mode, where each source image contributes a slice at its original coordinates.
- An alignment-aware mode, where each source image is first registered against its predecessor so that slices are extracted from a stabilized coordinate system.

The architecture is therefore organized around one central goal: convert an ordered image sequence into a deterministic slice-based composite while optionally compensating for camera translation and small rotation between frames.

## System overview

The executable is implemented primarily in [`src/main.cpp`](/Users/p4010/Develop/transitions/src/main.cpp), while the FFT-based phase-correlation primitive used for alignment lives in [`src/phase_corr.cpp`](/Users/p4010/Develop/transitions/src/phase_corr.cpp) and [`src/phase_corr.hpp`](/Users/p4010/Develop/transitions/src/phase_corr.hpp).

At a high level, the pipeline is:

1. Parse options and discover input files.
2. Read lightweight metadata from each file and establish a stable ordering.
3. Validate that all selected inputs share the same dimensions.
4. Optionally reduce the sequence with preview sampling.
5. If alignment is enabled, estimate a transform for each image relative to the previous image and accumulate those transforms to a common coordinate system.
6. Render one slice per image, either directly or through the estimated transform.
7. Merge the rendered slices into the final JPEG or TIFF output.

The implementation keeps the data model intentionally small:

- `InputEntry` stores file path, filename, capture time, and dimensions.
- `Image` stores decoded 8-bit RGB pixels for rendering.
- `Transform` stores translation in pixels and rotation in degrees.
- `AlignmentData` stores both pairwise transforms and accumulated global transforms.
- `CropRect` stores the common visible area after alignment when padding is disabled.

## Input ordering and selection

The first stage converts a directory of supported files into a deterministic ordered sequence. JPEG and DNG inputs are supported for ingestion, with JPEG and TIFF supported for output.

Ordering is based on one of two strategies:

- Natural filename ordering.
- Capture-time ordering derived from EXIF metadata for JPEG or LibRaw timestamp data for DNG.

When capture time is missing, the code falls back to natural filename order as a tie-breaker. This matters because every later stage assumes that adjacent images in the sequence represent adjacent temporal states.

Preview mode is a sequence reduction step, not a rendering optimization. It selects a uniformly spaced subset of the ordered input sequence while always preserving the first and last frame. The rest of the pipeline then operates on that reduced sequence exactly as if it were the full input set.

## Decoding and internal image representations

The code uses two distinct image representations because rendering and alignment have different requirements.

- Rendering uses full RGB images decoded into an `Image` structure with 8-bit interleaved pixels.
- Alignment uses a grayscale `phasecorr::Image` with single-precision floating-point samples.

For JPEG and DNG, the alignment path intentionally decodes reduced-resolution images:

- JPEG alignment decoding uses libjpeg scaling with denominator `2`.
- DNG alignment decoding uses LibRaw `half_size`.

This is an architectural choice to reduce the cost of transform estimation while preserving enough structure for reliable registration. After grayscale conversion, those reduced images are cached to disk in a compact binary form under `temp/alignment_cache/`, so the expensive decode step is separated from later transform estimation.

## Core algorithm

The core algorithm combines sequential registration with slice extraction.

### 1. Pairwise transform estimation

When auto-alignment is enabled, image `i` is aligned against image `i - 1`, not directly against the first image. This produces a sequence of relative transforms:

- `T_0 = identity`
- `T_i = transform(image_i-1 -> image_i)` for `i > 0`

This pairwise design is pragmatic:

- Adjacent frames are usually more similar than distant frames.
- The search space for alignment remains small.
- The algorithm only needs local consistency to build a global trajectory.

### 2. Grayscale conversion

Alignment works on luminance, not color. Each RGB image is converted to grayscale using the standard weighted sum:

`gray = 0.299 * R + 0.587 * G + 0.114 * B`

This reduces the registration problem to a single channel and avoids per-channel correlation complexity.

### 3. Phase correlation for translation

The low-level translation estimator in [`src/phase_corr.cpp`](/Users/p4010/Develop/transitions/src/phase_corr.cpp) implements standard phase correlation:

1. Compute the FFT of the reference image and the candidate image.
2. Multiply one spectrum by the complex conjugate of the other.
3. Normalize each frequency bin by its magnitude to keep only phase information.
4. Run the inverse FFT.
5. Find the peak in the correlation surface.
6. Convert the peak location into a signed `(dx, dy)` translation, accounting for wrap-around.

The result is a translation estimate plus the peak value, which acts as a confidence score for candidate rotations.

### 4. Rotation search around phase correlation

Phase correlation itself estimates translation, not rotation. The software handles small rotational differences by brute-force angle search around the candidate image:

- A coarse pass tests angles from `-5` to `+5` degrees in `1` degree increments on half-resolution grayscale images.
- A refinement pass tests offsets from `-0.4` to `+0.4` degrees around the best coarse angle on the higher-resolution grayscale images.

For each tested angle:

1. Rotate the candidate grayscale image around its center using bilinear interpolation.
2. Run phase correlation against the reference grayscale image.
3. Keep the angle whose correlation peak is strongest.

This creates a simple but effective estimator for small camera rotations without introducing a more complex feature-based registration system.

### 5. Scale restoration

Because alignment images may be decoded at reduced resolution, the estimated translation is scaled back to the full-resolution image domain before rendering. Rotation is preserved directly in degrees.

### 6. Transform accumulation

Relative transforms are converted into global transforms by composition. The accumulated transform for image `i` maps that image into the coordinate system of the first frame. Composition is not simple component-wise addition because translations must be rotated by the previously accumulated rotation before being added.

This produces a camera trajectory across the sequence:

- `G_0 = identity`
- `G_i = G_i-1 composed with T_i`

Rendering uses `G_i`, not the raw pairwise transforms.

## Rendering architecture

Rendering is split into two stages: slice generation and slice merge.

### Slice geometry

If there are `N` selected input images, the output is partitioned into `N` equal-width vertical slices or `N` equal-height horizontal slices. Integer boundaries are computed from proportional positions, so the full output extent is covered without gaps.

Each input contributes exactly one slice:

- In non-aligned mode, the slice is a direct crop.
- In aligned mode, the slice is sampled from the transformed image domain.

### Transformed sampling

Aligned rendering uses inverse mapping. For each output pixel in a slice:

1. Convert the output coordinate into the aligned global coordinate system.
2. Apply the inverse of the image transform implicitly to find the source coordinate in the original image.
3. Sample the source image with bilinear interpolation.
4. Write black when the mapped coordinate lies outside the source image.

There is one fast path: if rotation is effectively zero and translation is effectively integer-valued, the renderer uses row copies instead of per-pixel interpolation. On ARM platforms, bulk RGB copies can use NEON intrinsics.

### Cropping and padding

Alignment can expose empty borders. The software supports two policies:

- `pad_result = true`: keep the original image extent and fill uncovered regions with black.
- `pad_result = false`: compute the intersection of all transformed image footprints and crop the composite to that common visible rectangle.

The crop rectangle is computed geometrically by transforming the four corners of every source image and intersecting their axis-aligned bounds. This guarantees that all rendered slices come from valid content across the full sequence.

### Final merge

Slice generation produces an in-memory vector of slice images. The final output writer then streams rows to disk:

- For vertical slicing, each output row is built by concatenating the corresponding row from every slice.
- For horizontal slicing, slice rows are written sequentially.

This stage is format-agnostic apart from the scanline writer implementation, which currently supports JPEG and TIFF.

## Concurrency and staging

The architecture uses concurrency selectively rather than uniformly.

- Metadata scanning is currently implemented sequentially even though the API exposes a `parallelism` parameter.
- Slice rendering can overlap decode and render work using asynchronous tasks, while bounding the number of pending tasks.
- Pixel work inside transformed rendering can be split into row blocks.
- Alignment computation can be split into chunks, but each chunk still processes a consecutive subsequence because pairwise registration depends on the previous frame.

The important design principle is pipeline overlap: decode one image, render or cache it asynchronously, then continue decoding the next image. This reduces idle time without requiring the whole sequence to be resident in memory at once during every stage.

## Persistence and checkpoints

Alignment is treated as reusable intermediate state.

- Grayscale alignment inputs are cached as binary files in `temp/alignment_cache/`.
- Estimated transforms are serialized as JSON through `nlohmann::json`.

If the user supplies `--alignment-json` and the file already exists, the program skips recomputation and validates that the stored entries match the current ordered input sequence. This makes the alignment stage restartable and repeatable.

## Module responsibilities

- [`src/main.cpp`](/Users/p4010/Develop/transitions/src/main.cpp): command-line parsing, metadata extraction, ordering, image decoding, grayscale cache generation, transform estimation, slice rendering, crop computation, output writing, and overall orchestration.
- [`src/phase_corr.cpp`](/Users/p4010/Develop/transitions/src/phase_corr.cpp): FFT-based phase correlation kernel used to estimate translation.
- [`src/phase_corr.hpp`](/Users/p4010/Develop/transitions/src/phase_corr.hpp): public interface for the phase-correlation module.
- [`src/image.hpp`](/Users/p4010/Develop/transitions/src/image.hpp): lightweight grayscale image container used by the alignment code.

This separation is narrow but coherent: the main executable owns the full pipeline, while the mathematically specialized FFT primitive is isolated.

## Limitations and computational efficiency

The current design is effective for its intended problem, but its assumptions are narrow.

- All input images must have identical dimensions.
- Registration assumes only translation plus small rotation.
- Rotation search is bounded to a small angular window around zero.
- There is no scale estimation, perspective correction, or local warping.
- Phase correlation can become unstable when adjacent frames have weak overlap, strong exposure changes, motion blur, or repeated textures.
- Pairwise accumulation can drift over long sequences because small relative errors compound into the global transform.
- Slice rendering currently keeps all rendered slices in memory before the final merge, which increases peak memory usage for large image sets.

From a computational perspective, the dominant costs are:

- Image decode and color conversion.
- Repeated candidate-image rotations during alignment.
- FFTs for every tested rotation.
- Bilinear resampling during transformed slice rendering.

Several implementation choices reduce those costs:

- Alignment runs on half-resolution decoded images.
- A coarse-to-fine angle search avoids evaluating the full-resolution image at every coarse angle.
- Integer-translation rendering uses direct row copies instead of interpolation.
- Output is written as scanlines rather than constructing a second full-size output buffer.

Even with these optimizations, the alignment path remains substantially more expensive than direct slicing because it performs multiple FFT-based registrations per adjacent image pair. In asymptotic terms, if `N` is the number of images, `A` is the number of tested angles, and `P` is the number of pixels in an alignment image, the alignment stage is roughly proportional to `O(N * A * P log P)`, while the final rendering stage is roughly linear in the number of output pixels sampled.
