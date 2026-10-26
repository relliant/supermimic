# Dependencies and reference provenance

The new deployment implementation references, but does not vendor or modify,
`~/Project/GR00T-WholeBodyControl/gear_sonic_deploy`. Its C++ SDK2 transport,
message conventions and thread design informed this independent adapter. SONIC
model weights, planners, token encoders and its controller executable are not
redistributed here.

The available reference directory has no accessible Git metadata, so no commit
is invented. SHA256 identities recorded during implementation:

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp`:
  `073f8d04f292473851b6f0b923e6af7b2dc0686658eba19a8deab08c839e4fb1`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`:
  `0a0c3375fbeac21959a3d1ffd7931177853382f3b60a503f1c6466f2af7a524f`
- `gear_sonic_deploy/thirdparty/unitree_sdk2/include/unitree/idl/hg/LowState_.hpp`:
  `e01be29667100a6f2cd829b3ea8a90f846271b20a0cde96331ceecfb4d46e042`
- `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`:
  `9873d67aa6b9dcbe2ceba2c9d0a8810ba0e303fbd3fc3a6bcfde24fb111ab3a3`

SDK2 is linked externally through `SDK2_ROOT`. ONNX Runtime, Eigen, OpenSSL,
nlohmann-json, CycloneDDS, Python SDK2, MuJoCo, NumPy, PyTorch and ONNX remain
separately obtained dependencies. Preserve their license notices when packaging
a binary and associated shared libraries. CMake does not download them. The
Unitree SDK2 license below must accompany distributions linking its libraries.
LAFAN1 reference clips retain their separate CC BY-NC-ND 4.0 restrictions; the
repository's license does not relicense those data.

## Unitree SDK2: BSD 3-Clause License

Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
