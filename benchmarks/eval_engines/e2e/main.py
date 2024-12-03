import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from tqdm.contrib import tenumerate
from transformers import Pipeline

from benchmarks.data_sets.data_set import Data_set
from benchmarks.eval_engines.eval_engine import Configs, EvalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class E2EConfigs(Configs):
    # Overriding the default values of the parent class.
    tot_num_data: int = int(1e6)
    all_datasets: List[str] = field(default_factory=lambda: ["math500"])  # Fixed mutable default
    all_models: List[str] = field(default_factory=lambda: ["peiyi9979/mistral-7b-sft"])
    all_approaches: List[str] = field(default_factory=lambda: ["full", "quest", "streamingllm"])


class E2EEvalEngine(EvalEngine):

    def run_inference(self, pipe: Pipeline, dataset: Data_set) -> Data_set:

        logger.info("Run the inference. This might take a long time... Good luck")
        results = defaultdict(list)
        for i, (prompt, answer) in tenumerate(dataset, desc="dataset", leave=False):
            model_output, JCT, TPOT, num_decode = self.test_model(pipe, prompt, answer)
            results[f"output_{self.configs.approach}"].append(model_output)
            results[f"JCT_{self.configs.approach}"].append(JCT)
            results[f"TPOT_{self.configs.approach}"].append(TPOT)
            results[f"num_decode_{self.configs.approach}"].append(num_decode)
        dataset.update(results)
        dataset.calc_accuracy(self.configs.approach)
        dataset.save_dataset(self.configs.result_path)

        # Print some aggregate information
        accuracy_avg = np.mean(self.dataset.data[f"accuracy_{self.configs.approach}"])
        JCT_avg = np.mean(self.dataset.data[f"JCT_{self.configs.approach}"])
        TPOT_avg = np.mean(self.dataset.data[f"TPOT_{self.configs.approach}"])
        num_decode_avg = np.mean(self.dataset.data[f"num_decode_{self.configs.approach}"])
        logger.info(f"Average accuracy of {self.configs.approach}: {accuracy_avg:.3f}")
        logger.info(f"Average JCT of {self.configs.approach}: {JCT_avg:.2f} s")
        logger.info(f"Average TPOT of {self.configs.approach}: {TPOT_avg:.2f} s")
        logger.info(f"Average num_decode of {self.configs.approach}: {num_decode_avg:.2f}")

        return dataset

    def test_model(self, pipe, prompt, answer) -> Tuple[str, float, float, float, int]:

        torch.cuda.empty_cache()
        pipe.model.past_key_values.clear()

        input_ids = pipe.tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        start_time = time.perf_counter()
        model_output = pipe.model.generate(
            input_ids,
            max_length=pipe.model.config.max_position_embeddings,
            num_return_sequences=1,
            return_dict_in_generate=True,
            past_key_values=pipe.model.past_key_values,
        )

        JCT = time.perf_counter() - start_time
        num_decode = model_output.sequences[0].shape[0] - input_ids.shape[-1]
        TPOT = JCT / num_decode  # Include a short period of prefill stage

        model_output = pipe.tokenizer.decode(model_output.sequences[0])
        return model_output, JCT, TPOT, num_decode

        # input = pipe.tokenizer(prompt, return_tensors="pt").to("cuda")
        # with torch.no_grad():

        #     start_time = time.perf_counter()

        #     # Prefill
        #     output = pipe.model(
        #         input_ids=input.input_ids,
        #         past_key_values=pipe.model.past_key_values,
        #         use_cache=True,
        #     )

        #     torch.cuda.synchronize()
        #     prefill_time = time.perf_counter() - start_time

        #     # Store KV cache
        #     past_key_values = output.past_key_values

        #     # Produce the first token
        #     pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        #     generated_content = [pred_token_idx.item()]

        #     decode_time = 0
        #     # Decode autoregressively
        #     for num_decode in range(pipe.model.config.max_position_embeddings - 1):

        #         start_time = time.perf_counter()

        #         outputs = pipe.model(
        #             input_ids=pred_token_idx,
        #             past_key_values=past_key_values,
        #             use_cache=True,
        #         )

        #         torch.cuda.synchronize()
        #         decode_time += time.perf_counter() - start_time

        #         # Store KV cache
        #         past_key_values = outputs.past_key_values

        #         # Produece the next token
        #         pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        #         generated_content += [pred_token_idx.item()]

        #         if pred_token_idx.item() == pipe.tokenizer.eos_token_id:
        #             break

        #     TTFT = prefill_time
        #     JCT = prefill_time + decode_time
        #     TPOT = decode_time / num_decode

        # model_output = pipe.tokenizer.decode(generated_content, skip_special_tokens=True)
        # return model_output, TTFT, JCT, TPOT, num_decode


if __name__ == "__main__":

    configs = E2EConfigs.get_configs_from_cli_args()
    eval_engine = E2EEvalEngine(configs)
    eval_engine.run()
