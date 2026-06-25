# MLFilter

A DirectShow filter for processing video frames with machine-learning models on the GPU.

MLFilter is a DirectShow transform filter for media players such as MPC-BE. It is being
built to upscale video frames with an ONNX model on the GPU. The current phase implements:

- A pass-through filter (`MLFilter_x64.ax`) that delivers frames downstream unmodified.
- A configuration property page where you select an ONNX model and, optionally, a list of
  filename wildcard patterns (one per line, e.g. `*.mkv`). When patterns are set, the filter
  only processes files whose path matches one of them; for non-matching files it bypasses the
  connection and removes itself from the graph. Leave the list blank to process every file.
- On-demand engine building: when a file is played, the filter reads the video's actual
  resolution as the graph connects and builds the matching static, fp16 TensorRT engine (via
  the native TensorRT C++ API) before playback starts, showing a progress window while it
  works. A 1920×1080 engine is pre-built when you select a model so the most common case never
  waits. fp32 models are automatically converted to fp16 first (TensorRT 11 engines are
  strongly typed, so precision follows the model).

GPU inference inside the filter (running the engine, producing RGB48 output) and a DirectML
backend for AMD GPUs are planned for later phases. The engine-builder backend is abstracted
behind an `IEngineBuilder` interface so additional backends can be added.

Built engines are written to `%LOCALAPPDATA%\MLFilter\Engines\`. Each engine filename encodes
the model, resolution, GPU name, driver version, and TensorRT version. When you rebuild after a
driver, GPU, or TensorRT change, the engines for the old configuration are deleted automatically.

## Development setup

The engine-building backend uses NVIDIA TensorRT and the CUDA Toolkit. Install them with the
zip-package approach below before building.

1. **Install a recent NVIDIA driver** compatible with the CUDA/TensorRT version you choose.

2. **Download the TensorRT Windows zip.** Get the latest Windows x86_64 zip package for CUDA 13
   from NVIDIA's TensorRT download page and extract it. Move the folder somewhere stable, e.g.:

   ```
   C:\SDKs\TensorRT-11.x.x.x\
   ```

3. **Set `TENSORRT_ROOT`.** Add a user or system environment variable pointing at that folder:

   ```
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

### Visual Studio project setup

The build reads `$(TENSORRT_ROOT)` and `$(CUDA_PATH)` (see `tensorrt.props`):

- Include directories: `$(TENSORRT_ROOT)\include`, `$(CUDA_PATH)\include`
- Library directories: `$(TENSORRT_ROOT)\lib`, `$(CUDA_PATH)\lib\x64`
- Link libraries: `nvinfer_11.lib`, `nvonnxparser_11.lib`, `cudart.lib`, `nvml.lib`

If you install a different TensorRT major version, update the `TensorRtLibSuffix` property in
`tensorrt.props` (e.g. `_11`) to match the import-lib suffix in `$(TENSORRT_ROOT)\lib`.

## Building

Open `MLFilter.sln` in Visual Studio and build the `Release|x64` configuration, or use the
helper script (it locates MSBuild via vswhere, so no Developer prompt is needed):

```
.\make_dev.ps1                       # builds Release|x64
.\make_dev.ps1 -Configuration Debug  # Debug build
.\make_dev.ps1 -Rebuild              # clean rebuild
```

The output is `x64\Release\MLFilter_x64.ax`. (You can also run
`msbuild MLFilter.sln /p:Configuration=Release /p:Platform=x64` directly.)

## Registering the filter

For local testing, register the built `.ax` with `install_dev.bat` (it self-elevates and
defaults to the Release build):

```
install_dev.bat            # registers x64\Release\MLFilter_x64.ax
install_dev.bat Debug      # registers the Debug build
```

This registers the `.ax` in place; it relies on `%TENSORRT_ROOT%\bin` being on `PATH` (from the
development setup) so the player can load the TensorRT DLLs. Unregister with
`regsvr32 /u x64\Release\MLFilter_x64.ax`. In MPC-BE, add MLFilter as an external filter and
open its properties to configure the model and file patterns.

## Creating a redistributable release

`make_release.ps1` packages a self-contained release for machines that do **not** have the
TensorRT/CUDA SDKs installed:

```
.\make_release.ps1     # builds release\ (~1.5 GB)
```

It takes no parameters — the build is always the same. Build the `Release|x64` configuration
first (e.g. with `.\make_dev.ps1`). It produces one release that works for most people, in
`release\`: it copies the built `MLFilter_x64.ax`, generates `install.bat`/`uninstall.bat`, and
gathers the runtime DLLs the filter needs into a `bin\` subfolder — the TensorRT DLLs (runtime,
ONNX parser, plugins) plus the CUDA runtime DLLs TensorRT loads dynamically (cudart, nvrtc,
nvJitLink, …).

To keep the size reasonable, it bundles TensorRT builder-resource DLLs only for consumer GPU
architectures — `sm75` (Turing, RTX 20xx/GTX 16xx), `sm86` (Ampere, RTX 30xx), `sm89` (Ada, RTX
40xx), `sm120` (Blackwell, RTX 50xx), plus `ptx` (a JIT fallback that lets other architectures
still build). Datacenter architectures (sm80/sm90/sm100), cuBLAS (TensorRT 10+ keeps its cuBLAS
tactics off by default), and the unused `lean`/`dispatch`/`vc_plugin` runtimes are excluded.

```
release\
  MLFilter_x64.ax     the filter
  install.bat         registers the filter (run as administrator)
  uninstall.bat       unregisters the filter
  bin\                bundled TensorRT + CUDA runtime DLLs
```

The TensorRT DLLs are delay-loaded and `DllMain` prepends this `bin\` folder to the process
search path, so the filter finds its bundled dependencies even though they live in a
subfolder. Registration records the `.ax`'s location, so keep the folder in place (or re-run
`install.bat` after moving it). `release\` is git-ignored.
