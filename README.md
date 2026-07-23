# MLFilter

A DirectShow filter for processing video frames with machine-learning models on the GPU.

Supports FP16 and FP32 ONNX models.

The filter is mainly intended to be used to upscale video with models like the
[Anime JaNai HD V3 models](https://github.com/the-database/mpv-AnimeJaNai/releases/tag/3.0.0)
(join the [JaNai Discord server](https://discord.gg/EeFfZUBvxj) to get their latest models).

## How To Use

- Download the latest [release](https://github.com/nwoltman/MLFilter/releases) and unzip the contents to your desired location
- Run the `install.bat` script as an administrator (right-click it to find this option)
  - Do not move the folder after installing the filter. If you want to move it, uninstall the filter
    with `uninstall.bat` first, then move the folder, then re-install it.
- Open your video player software and add `MLFilter` as an external filter
- Set the filter to `Prefer` (or use `Merit` if you have other, similar filters and need them to run in a specific order)
- Double-click the filter to open the filter configuration. Select an HD and/or an SD ONNX model.
  The SD model is used when either source dimension is below 720p (1280x720); otherwise
  the HD model is used. With neither model configured, the filter removes itself from the graph.
- Press `Ctrl+Alt+J` to show the MLFilter Debug overlay when playing a video
- To update MLFilter later, close any programs using MLFilter, right-click `update.bat`, and select
  **Run as administrator**. The updater only downloads files when a newer release is available.

### Running the benchmark

After installing MLFilter and configuring an HD model, run `run_benchmark.bat` from the release
folder. It benchmarks 1920x1080 and 1280x720 resolutions. Lower resolutions will always perform
better, so if the timings for those HD resolutions are good, SD resolutions will be even better.

## System requirements

You need:

- **An NVIDIA graphics card, GeForce RTX 20-series / GTX 16-series or newer**
  - AMD GPUs are planned to be supported in the future
- **An up-to-date NVIDIA driver** (version 610.62 or newer). If you have any trouble, the first
  thing to try is updating your driver from NVIDIA's website or the NVIDIA App.
- **A 64-bit media player that supports external DirectShow filters**, such as MPC-BE/HC

### For NVIDIA Users

The first time you play a video at a new resolution, the filter spends a few moments building the
inference engine for your model and card before playback starts. This happens once per resolution;
after that, videos of that size start right away. This process will repeat each time you upgrade
your GPU driver or when this filter is updated with newer versions of NVIDIA's TensorRT software.

Built engines are written to `%LOCALAPPDATA%\MLFilter\Engines\`. When a new engine is built, the
previous one automatically gets deleted. You can also manually delete all built engines use the
"Delete all engine files" button in the filter configuration GUI.

## Development setup

This repository uses git submodules. Clone with submodules, or initialize them in an existing clone.

```batch
git clone --recurse-submodules https://github.com/nwoltman/MLFilter.git
```

The engine-building backend uses NVIDIA TensorRT and the CUDA Toolkit. Install them with the
zip-package approach below before building.

1. **Install a recent NVIDIA driver** compatible with the CUDA/TensorRT version you choose.

2. **Download the TensorRT Windows zip.** Get the latest Windows x86_64 zip package for CUDA 13
   from NVIDIA's TensorRT download page and extract it. Move the folder somewhere stable, e.g.:

   ```
   C:\SDKs\TensorRT-11.x.x.x\
   ```

3. **Set `TENSORRT_ROOT`.** Add a user or system environment variable pointing at that folder:

   ```batch
   setx TENSORRT_ROOT "C:\SDKs\TensorRT-11.x.x.x"
   ```

   Open a new terminal afterward. Also add the TensorRT `bin` folder to `PATH` so Windows finds
   the TensorRT DLLs at runtime:

   ```
   %TENSORRT_ROOT%\bin
   ```

   (NVIDIA documents adding the TensorRT `bin` folder to `PATH` for the zip install, because that
   is how Windows locates the TensorRT DLLs while running samples and tools.)

4. **Install the CUDA Toolkit** version required by the TensorRT release (CUDA 13). Do a *Custom*
   install and uncheck everything except the first "CUDA" checkbox and its children. The installer
   sets the `CUDA_PATH` environment variable.

### Visual Studio Project

The build reads `$(TENSORRT_ROOT)` and `$(CUDA_PATH)` (see `config/tensorrt.props`):

- Include directories: `$(TENSORRT_ROOT)\include`, `$(CUDA_PATH)\include`
- Library directories: `$(TENSORRT_ROOT)\lib`, `$(CUDA_PATH)\lib\x64`
- Link libraries: `nvinfer_11.lib`, `nvonnxparser_11.lib`, `cudart.lib`, `nvml.lib`

If you install a different TensorRT major version, update the `TensorRtLibSuffix` property in
`config/tensorrt.props` (e.g. `_11`) to match the import-lib suffix in `$(TENSORRT_ROOT)\lib`.

## Building

Open `MLFilter.sln` in Visual Studio and build the `Release|x64` configuration, or use the
helper script (it locates MSBuild via vswhere, so no Developer prompt is needed):

```batch
.\build.ps1                       # builds Release|x64
.\build.ps1 -Configuration Debug  # Debug build
.\build.ps1 -Rebuild              # clean rebuild
```

The output is `x64\Release\MLFilter_x64.ax`.

## Registering the filter

For local testing, register the built `.ax` with `install_dev.bat` (it self-elevates and
defaults to the Release build):

```batch
install_dev.bat            # registers x64\Release\MLFilter_x64.ax
install_dev.bat Debug      # registers the Debug build
```

This registers the `.ax` in place; it relies on `%TENSORRT_ROOT%\bin` being on `PATH` (from the
development setup) so the player can load the TensorRT DLLs. Unregister with
`regsvr32 /u x64\Release\MLFilter_x64.ax`. In MPC-BE, add MLFilter as an external filter and
open its properties to configure the model and file patterns.

## Creating a redistributable release

`make_release.ps1` packages a release for machines that do **not** have the TensorRT/CUDA SDKs
installed. With no argument it performs a dry run and only rebuilds the `release\` folder:

```batch
.\make_release.ps1          # dry run: builds release\
.\make_release.ps1 1.2.3    # publishes GitHub release v1.2.3
```

The release script builds `Release|x64` before packaging.
A real release requires the [GitHub CLI](https://cli.github.com/) to be installed and authenticated.
It creates the tag and GitHub release, then uploads `MLFilter-vX.Y.Z.zip` and one TensorRT
builder-resource DLL per supported GPU architecture.

The zip contains the architecture-independent TensorRT DLLs. `install.bat` uses `nvidia-smi` to detect the installed
GPU architecture and downloads the matching builder-resource DLL from the same GitHub release
before registering the filter. Supported architectures are `sm75` (Turing, RTX 20xx/GTX 16xx),
`sm86` (Ampere, RTX 30xx), `sm89` (Ada, RTX 40xx), and `sm120` (Blackwell, RTX 50xx).

```
release\
  MLFilter_x64.ax          the filter
  install.bat              registers the filter (run as administrator)
  install_dependency.ps1   downloads the correct GPU architecture DLL
  run_benchmark.bat        runs the standard benchmark matrix
  uninstall.bat            unregisters the filter
  update.bat               launches the updater
  update.ps1               checks for and installs a newer GitHub release
  bin\
    benchmark_x64.exe      standalone benchmark
    *.dll                  architecture-independent TensorRT DLLs
```

The TensorRT DLLs are delay-loaded and `DllMain` prepends this `bin\` folder to the process
search path, so the filter finds its bundled dependencies even though they live in a
subfolder.
