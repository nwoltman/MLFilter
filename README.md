# MLFilter

A DirectShow filter for processing video frames with machine-learning models on the GPU.

Supports FP16 and FP32 ONNX models.

The filter is mainly intended to be used to upscale video with models like the
[latest Anime JaNai models](https://github.com/the-database/mpv-AnimeJaNai/tree/main/BuildMpvUpscale2xAnimeJaNai/mpv-upscale-2x_animejanai/animejanai/onnx)
(or the [JaNai V3 models](https://github.com/the-database/mpv-AnimeJaNai/releases/tag/3.0.0)).

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
- Press `Ctrl+Alt+J` to show/hide the MLFilter Debug overlay when playing a video
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

### Temporary downstream format diagnostic

This working-tree diagnostic is enabled automatically in both Release and Debug builds. It logs
the immediate downstream pin's advertised media types and checks 22 candidate pixel formats with
`IPin::QueryAccept` before MLFilter attempts its normal RGB48 connection. It does not change
MLFilter's output format or deliver test frames.

1. Close PotPlayer, build with `.\build.ps1`, and register the build with `install_dev.bat`.
2. Select PotPlayer's **Built-in Direct3D 11 Video Renderer**, enable MLFilter as an external
   filter, and configure an ONNX model as usual. Use a video that meets MLFilter's resolution
   and file-pattern settings so the filter stays in the graph.
3. Open that video once. A report is written even if output negotiation fails and playback
   cannot start, provided PotPlayer attempts to connect MLFilter's output.
4. Open `%LOCALAPPDATA%\MLFilter\Logs\` and find the newest `renderer-formats-<PID>.log`.
   Reports append for each connection attempt in the player process; restarting PotPlayer
   normally creates a new file. Look at `Downstream filter` to confirm which filter was queried,
   then search for `SUMMARY` to see which candidates passed at least one header variant.

If `Downstream filter` names a subtitle or video processing filter, these are that filter's
results. Disable the intervening filter and repeat to query the renderer directly. There may
be several connection attempts to different filters in the same log, so read each section's
identity rather than combining their results.

Advertised types are preferences, not necessarily a complete capability list. `ACCEPTED` means
only that `QueryAccept` returned `S_OK` for the logged dimensions, header, orientation and color
metadata. It does not prove successful connection/playback, GPU texture delivery, HDR handling,
or preservation of all channel bits. Candidate probes include RGB48, BGR48, Y416, Y410, P016,
P010 and common 4:2:2/8-bit formats. They test `VideoInfo2` with and without color tags and
`VideoInfo` without tags; standard RGB DIBs also test both height signs. Tagged YUV probes use
limited-range BT.709. Skipped variants and HRESULT errors are recorded separately.

To remove the diagnostic after collecting the report, remove the `Connect` override and
diagnostic include from `src/output_pin.cpp` and its declaration from `src/output_pin.h`,
delete `src/renderer_diagnostics.h` and `src/renderer_diagnostics.cpp`, and remove their entries
from `MLFilter.vcxproj` and this README section.

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
  install.bat              registers the filter (run as administrator)
  run_benchmark.bat        runs the standard benchmark matrix
  uninstall.bat            unregisters the filter
  update.bat               launches the updater
  bin\
    MLFilter_x64.ax        the filter
    benchmark_x64.exe      standalone benchmark
    install_dependency.ps1 downloads the correct GPU architecture DLL
    update.ps1             checks for and installs a newer GitHub release
    *.dll                  architecture-independent TensorRT DLLs
```

The TensorRT DLLs are delay-loaded and `DllMain` prepends the filter's `bin\` folder to
the process search path so the filter finds its bundled dependencies.
