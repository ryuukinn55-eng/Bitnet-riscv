# BitNet RISC-V Inference

## Installation

### Requirements
- python>=3.9
- cmake>=3.22
- clang>=18
    - For Windows users, install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/). In the installer, toggle on at least the following options(this also automatically installs the required additional tools like CMake):
        -  Desktop-development with C++
        -  C++-CMake Tools for Windows
        -  Git for Windows
        -  C++-Clang Compiler for Windows
        -  MS-Build Support for LLVM-Toolset (clang)
    - For Debian/Ubuntu users, you can download with [Automatic installation script](https://apt.llvm.org/)

        `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
- conda (highly recommend)

### Build from source
**1. Clone the repo**

```bash
git clone --recursive https://github.com/ryuukinn55-eng/Bitnet-riscv.git
cd Bitnet-riscv
```

**2. Install dependencies**

```bash
# (Recommended) Create a new conda environment
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp

pip install -r requirements.txt
```

**3. Download the model**

```bash
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
```

**4. Build the project**

```bash
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

> [!NOTE]
> RISC-V cross-compilation flags (toolchain path, `-march` string, compiler options) are hardcoded in the `compile()` function inside `setup_env.py`. If you need to adjust them for your environment, edit that function directly:
>
> ```python
> # setup_env.py — riscv64 branch inside compile()
> cmd += [
>  f"-DCMAKE_TOOLCHAIN_FILE={your_toolchain_path}",  # change toolchain path
>  f"-march={your_march_string}",                     # change march string
>  ...
> ]
> ```

<details>
<summary><code>setup_env.py</code> usage</summary>


```
usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,
                    tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,
                    tiiuae/Falcon3-10B-Instruct-1.58bit}]
                    [--model-dir MODEL_DIR] [--log-dir LOG_DIR]
                    [--quant-type {i2_s,tl1}] [--quant-embd] [--use-pretuned]

Setup the environment for running inference

optional arguments:
  -h, --help            show this help message and exit
  --hf-repo, -hr        Model used for inference
  --model-dir, -md      Directory to save/load the model
  --log-dir, -ld        Directory to save the logging info
  --quant-type, -q      Quantization type: {i2_s, tl1}
  --quant-embd          Quantize the embeddings to f16
  --use-pretuned, -p    Use the pretuned kernel parameters
```

</details>

---

## Usage

### Run with QEMU

```bash
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "You are a helpful assistant" \
  -cnv
```

### Run on BPI-F3 

```bash
./build/bin/llama-cli \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -n 8 \
  -t 8 \
  -p "You are a helpful assistant" \
  -ngl 0 \
  -c 2048 \
  --temp 0.8 \
  -b 1
```

<details>
<summary><code>run_inference.py</code> usage</summary>


```
usage: run_inference.py [-h] [-m MODEL] [-n N_PREDICT] -p PROMPT
                        [-t THREADS] [-c CTX_SIZE] [-temp TEMPERATURE] [-cnv]

Run inference

optional arguments:
  -h, --help            show this help message and exit
  -m, --model           Path to model file
  -n, --n-predict       Number of tokens to predict
  -p, --prompt          Prompt to generate text from
  -t, --threads         Number of threads to use
  -c, --ctx-size        Size of the prompt context
  -temp, --temperature  Sampling temperature
  -cnv, --conversation  Enable chat mode (system prompt via -p)
```

</details>