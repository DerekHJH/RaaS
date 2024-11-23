# RaaS: Reasoning-Aware Attention Sparsity for Efficient Long-Context LLM Inference


## TL;DR
RaaS is an efficient long-context LLM inference framework that leverages **query-aware sparsity** in KV cache to reduce memory and computation requirements during attention and thus boost throughput. 


## Installation
1. Clone this repo (also clone submodules)
```
git clone --recurse-submodules https://github.com/DerekHJH/RaaS
cd RaaS
```
2. Install dependency libraries
```
conda create -yn RaaS python=3.10
conda activate RaaS

# RaaS
pip install -e .

# Flash-Attention
pip install ninja packaging
pip install flash-attn==2.6.3 --no-build-isolation

# Install CMake (with version >= 3.26.4)
conda install cmake

# build libraft
cd kernels/3rdparty/raft
./build.sh libraft
```
3. Compile kernel benchmarks (Optional). Remember to configure env variables for CUDA (Check the [tutorial](https://faculty.cc.gatech.edu/~hyesoon/spr09/installcuda.html)).
```
cd kernels
mkdir build && cd build
cmake ..
make -j
```
4. Build end-to-end operators with PyBind
```
# This will automatically build and link the operators
cd quest/ops
bash setup.sh
```
## Accuracy Evaluation
Our evaluations are based on [LongChat-7B-v1.5-32K](https://huggingface.co/lmsys/longchat-7b-v1.5-32k?clone=true) and [Yarn-Llama2-7B-128K](https://huggingface.co/NousResearch/Yarn-Llama-2-7b-128k) models, which are capable of handling long-context text generations. We evaluate both passkey retrieval and LongBench benchmarks. We provide several scripts to reproduce our results in the paper:

To get the Passkey Retrieval results, please modify and execute:
```
bash scripts/passkey.sh
```

To reproduce the LongBench results, please modify and execute:
```
bash scripts/longbench.sh
```

To evaluate the perplexity result of PG-19, please execute:
```
bash scripts/ppl_eval.sh
```

## Efficiency Evaluation
Kernels and end-to-end effiency are evaluated on NVIDIA Ada6000 and RTX4090 GPUs with CUDA version of 12.4. We provide several scripts to reproduce our results in the paper:

### Kernel-level Efficiency
We also release the unit tests and benchmarks used for kernel implementations. Correctness of kernel is verified by unit tests in `kernels/src/test`, while performance is evaluated by NVBench in `kernels/src/bench`. We also test the correctness of PyBind operators in `quest/tests` with PyTorch results via PyTest.

To test the correctness of kernels, please execute:
```
cd kernels/build
./test_batch_decode # or any other operator
```
Or utilize PyTest:
```
cd quest/tests
PYTHONPATH=$PYTHONPATH:../../ pytest
```
To reproduce the kernel performance shown in paper, please execute:
```
cd kernels/build
./bench_batch_decode -a seqlen=4096 -a page_budget=[64,512]
# or any other operator
```
With sample output:
![](./assets/figures/fig-kernel-bench.png)

### End-to-end Efficiency

Quest can achieve up to 2.23× end-to-end speedup while performing well on tasks with long dependencies with negligible accuracy loss:

![](./assets/figures/fig_e2e.png)

We incorporate all implemented operators into a full pipeline to evaluate the end-to-end efficiency in text generations. Based on the [Huggingface Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py), we enable a KV-Cache manager which supports query-aware sparsity as shown in `quest/models/QuestAttention.py`.

To reproduce the end-to-end efficiency results in Figure.10, please execute:
```
bash scripts/bench_efficiency_e2e.sh
```

For the qualitative analysis of baselines, we use FlashInfer kernel to estimate the performance of H2O and TOVA. To reproduce the results in Figure.11, please execute:
```
bash scripts/bench_kernels.sh
```

## Examples
We provide several examples to demonstrate the usage of Quest. These examples are implemented with the end-to-end integration of Quest operators, and can be executed with the following commands (please make sure you have setup all the operators):
```
python3 scripts/example_textgen.py
```
With example output of long-context summarization under LongChat-7B-v1.5-32K model:
![](./assets/figures/fig-examples.png)

You can also try `scripts/example_demo.py` to test the performance of Quest on your own text generation tasks. We provide a simple interface to load the model and generate text with Quest operators. The above demo is an example with 32K input on FP16 LongChat-7B-v1.5-32K. Quest with 2048 token budget achieves 1.7x speedup compared to full cache FlashInfer version.

