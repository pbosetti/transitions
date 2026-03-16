# transitions

`transitions` is a C++20 command line tool that composes a sequence of same-sized
input images into a single output image made of evenly sized slices.

## Supported formats

- Input: JPEG (`.jpg`, `.jpeg`) and DNG (`.dng`)
- Output: JPEG (`.jpg`, `.jpeg`) and TIFF (`.tif`, `.tiff`)

DNG output was intentionally not implemented. While writing DNG is technically
possible with Adobe's DNG SDK, this tool generates a rendered composite image,
so JPEG and TIFF are the practical output formats.

## Build

```sh
cmake -B build -G Ninja
cmake --build build
```

The project expects `libjpeg` and `libraw` to be available through `pkg-config`.

## Usage

```sh
./build/transitions [options] <input_dir> <output.{jpg,tif}>
```

Options:

- `--sort <name|time>`: sort by natural filename order or capture time, defaulting to `time`
- `--reverse`: reverse the selected order
- `--horizontal`: use horizontal slices instead of vertical slices
- `--auto-align`: align each image to the previous one using translation only and pad missing areas with black
- `--global-auto-align`: align the full sequence and crop to the common visible area
- `--dump-intermediate`: save slices in `temp/` and alignment data in `temp/alignment.json` before the final merge
- `--preview <n>`: keep `n` regularly sampled images, always including the first and the last
- `--parallel <n>`: load input images using `n` threads
- `--quality <1-100>`: set output JPEG quality for JPEG output

Notes:

- All input images must have the same dimensions.
- Natural sorting means `2.jpg` comes before `10.jpg`.
- When sorting by time, the tool uses EXIF `DateTimeOriginal` for JPEG images and LibRaw capture timestamp data for DNG images. Files without capture time fall back to filename order.
- `--dump-intermediate` creates a `temp/` folder next to the output file and recreates it for each run.
- Alignment JSON is written as a single object keyed by filename, with both relative and total offsets, for example: `{"file_01.jpg":{"relative":{"x":0,"y":0},"total":{"x":10,"y":12}}}`.
