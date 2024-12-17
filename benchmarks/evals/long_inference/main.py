import logging

import pandas as pd
from tqdm.contrib import tenumerate
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
tot_length = 2**12  # 32K

prefill_lengths = list(range(2**10, tot_length, 2**10))
prefill_time = []
decode_time = []

for i, prefill_length in tenumerate(prefill_lengths):
    decode_length = tot_length - prefill_length
    sampling_params = SamplingParams(min_tokens=decode_length, max_tokens=decode_length)

    output = llm.generate(
        sampling_params=sampling_params,
        prompt_token_ids=[i]
        * prefill_length,  # Dummy tokens, different in each iteration to avoid KV reuse
        # use_tqdm=False, # Prevent excessive output
    )
    prefill_time.append(output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time)
    decode_time.append(output[0].metrics.finished_time - output[0].metrics.first_token_time)

pd.DataFrame(
    {
        "prefill_length": prefill_lengths,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
    }
).to_json("results/long_inference.json")
