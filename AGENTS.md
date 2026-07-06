## General Instructions

When writing code, use blank lines to separate logical sections of functions.

After completing changes to code, use `build.ps1` to ensure everything builds correctly.

## Quality

One of this project's goals is to maintain the highest possible quality for the filter output video.
Use industry-standard algorithms (e.g. IEEE standards) and video-related quality judgements when
making changes.

To maintain high quality, use existing, well-known libraries when possible or write code following
the best practices defined by existing libraries if using a library directly is not an option.

## Preparedness for Future Changes

This project currently only supports NVIDIA GPUs. In the future, we will want it to also support AMD
GPUs which will use DirectML for upscaling inference.
Ensure that changes to the project structure code so that there is a simple pathway for adding AMD
GPU support in the future.
