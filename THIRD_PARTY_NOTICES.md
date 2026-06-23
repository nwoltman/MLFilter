# Third Party Notices

This file lists third-party software, SDKs, runtime libraries, tools, and
documentation references that may be used by MLFilter.

MLFilter's own source code is intended to be licensed under the Apache License,
Version 2.0. Third-party components remain governed by their own licenses.

This notice file is a starting point. Update it before each public release to
match the exact third-party files, binaries, source code, and tools included in
that release package.

## NVIDIA TensorRT

Project: NVIDIA TensorRT

Website: https://developer.nvidia.com/tensorrt

Documentation: https://docs.nvidia.com/deeplearning/tensorrt/

Open source repository: https://github.com/NVIDIA/TensorRT

License:

- TensorRT open source components are licensed under the Apache License,
  Version 2.0, as stated in the NVIDIA TensorRT GitHub repository.
- TensorRT SDK binaries and redistributable runtime files are governed by the
  NVIDIA TensorRT SDK license agreement.

Use in MLFilter:

- MLFilter may use TensorRT to build and run GPU-specific inference engines for
  NVIDIA GPUs.
- MLFilter may include a TensorRT-based engine builder utility that builds a
  serialized TensorRT engine from a user-provided ONNX model.
- Release packages may include TensorRT runtime DLLs only when redistribution is
  permitted by the applicable NVIDIA license terms.

Notes:

- TensorRT engines are hardware-specific and may depend on the installed NVIDIA
  driver, GPU architecture, TensorRT version, and build settings.
- NVIDIA developer tools, including SDK-provided command-line tools, should not
  be redistributed unless the applicable NVIDIA license terms identify them as
  distributable.
- If MLFilter reuses source code from NVIDIA TensorRT samples, including
  `samples/trtexec`, retain the required NVIDIA copyright notices and Apache
  License notices.

## NVIDIA CUDA Runtime

Project: NVIDIA CUDA Toolkit / CUDA Runtime

Website: https://developer.nvidia.com/cuda-toolkit

Documentation: https://docs.nvidia.com/cuda/

License:

- CUDA Toolkit and CUDA runtime components are governed by NVIDIA's applicable
  CUDA software license terms.

Use in MLFilter:

- MLFilter's NVIDIA backend may require CUDA runtime libraries used by TensorRT
  and related GPU execution paths.
- Release packages may include CUDA runtime DLLs only when redistribution is
  permitted by the applicable NVIDIA license terms.

Notes:

- The installed NVIDIA display driver is provided by NVIDIA and is not part of
  MLFilter.
- Users must have a compatible NVIDIA GPU and driver for the TensorRT backend.

## NVIDIA ONNX Parser for TensorRT

Project: TensorRT ONNX Parser

Repository: https://github.com/NVIDIA/TensorRT

License:

- The ONNX parser source distributed in the NVIDIA TensorRT open source
  repository is licensed under the Apache License, Version 2.0.
- Binary parser DLLs distributed as part of the TensorRT SDK are governed by
  the applicable NVIDIA TensorRT SDK license terms.

Use in MLFilter:

- MLFilter's TensorRT engine builder may use the TensorRT ONNX parser to import
  user-provided ONNX models and compile TensorRT engines.

## ONNX

Project: Open Neural Network Exchange

Website: https://onnx.ai/

Repository: https://github.com/onnx/onnx

License:

- ONNX is licensed under the Apache License, Version 2.0.

Use in MLFilter:

- MLFilter may accept ONNX model files supplied by users.
- MLFilter does not grant any rights to user-provided models. Model files remain
  governed by their own licenses.

Notes:

- Users are responsible for ensuring they have the right to use each model they
  load with MLFilter.

## ONNX Runtime

Project: Microsoft ONNX Runtime

Website: https://onnxruntime.ai/

Repository: https://github.com/microsoft/onnxruntime

License:

- ONNX Runtime is licensed under the MIT License.

Use in MLFilter:

- ONNX Runtime may be used by future MLFilter backends, for example DirectML,
  CPU, CUDA, or other execution providers.

Notes:

- Remove this section if ONNX Runtime is not included in a release package and
  is not used by the shipped application.

## Microsoft DirectML

Project: Microsoft DirectML

Documentation: https://learn.microsoft.com/windows/ai/directml/dml

Repository: https://github.com/microsoft/DirectML

License:

- DirectML headers, samples, and related open source materials are governed by
  the license terms in Microsoft's DirectML repository.
- DirectML runtime components provided by Windows are governed by Microsoft's
  applicable Windows license terms.

Use in MLFilter:

- DirectML may be used by a future MLFilter backend for vendor-neutral GPU
  inference on Windows.

Notes:

- Remove this section if DirectML is not included in a release package and is
  not used by the shipped application.

## DirectShow Base Classes

Project: Microsoft DirectShow Base Classes

Documentation: https://learn.microsoft.com/windows/win32/directshow/directshow

License:

- DirectShow SDK headers, libraries, samples, and base classes are governed by
  the applicable Microsoft Windows SDK license terms.

Use in MLFilter:

- MLFilter may use DirectShow APIs and may derive from or adapt DirectShow base
  class sample code when implementing the filter, pins, allocator negotiation,
  and registration logic.

Notes:

- If Microsoft sample source code is copied or modified, retain all required
  Microsoft copyright and license notices.

## Microsoft Windows SDK

Project: Microsoft Windows SDK

Website: https://developer.microsoft.com/windows/downloads/windows-sdk/

License:

- The Windows SDK is governed by Microsoft's applicable Windows SDK license
  terms.

Use in MLFilter:

- MLFilter may use Windows SDK headers, import libraries, tools, and COM /
  DirectShow interfaces.

## User-Provided Models

MLFilter may allow users to load third-party ONNX models or other model formats.
Those model files are not part of MLFilter unless explicitly distributed in an
MLFilter release package.

Users are responsible for ensuring they have the rights to use any model file
they load. Model licenses may restrict commercial use, redistribution,
modification, generated output, or use with specific runtimes.

## Release Checklist

Before publishing a release package:

1. List every third-party DLL, EXE, library, header, source file, script, model,
   and documentation file included in the package.
2. Confirm that each included component is redistributable under its applicable
   license.
3. Add or update notices for each included component.
4. Include required license text files for Apache-2.0, MIT, BSD, Microsoft, and
   NVIDIA components as applicable.
5. Verify whether any NVIDIA developer tools are included. If so, confirm that
   the applicable NVIDIA license terms permit redistribution.
6. Verify that MLFilter's installer and documentation do not imply that NVIDIA,
   Microsoft, AMD, or any other third party sponsors or endorses MLFilter unless
   written permission has been obtained.

