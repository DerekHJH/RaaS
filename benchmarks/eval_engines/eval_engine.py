import os
import sys
import argparse
from collections import defaultdict
from tqdm.contrib import tenumerate
from dataclasses import dataclass, field
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
logging.basicConfig(level=logging.INFO)

# The following are local imports
from benchmarks.data_sets.utils import str2class

# The following imports are subject to change
from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral

@dataclass
class Configs:

    dataset: str
    model: str
    approach: str
    all_datasets: List[str] = field(default_factory=lambda: ['needle', 'math500'])  # Fixed mutable default
    all_models: List[str] = field(default_factory=lambda: ['peiyi9979/mistral-7b-sft'])  # Fixed mutable default
    all_approaches: List[str] = field(default_factory=lambda: ['full', 'quest'])  # Fixed mutable default
    seed: int = 42
    result_path: str = 'results'

    @classmethod
    def get_configs_from_cli_args(cls) -> 'Configs':
        """
        Parse the command line arguments and return the Configs object.
        """
        # Add the arguments to the parser.
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--approach', type=str, required=True)
        parser.add_argument('--seed', type=int, default=42)
        
        # Parse the arguments.
        args = parser.parse_args()
        configs = cls(**vars(args))
        return configs
        

    def __post_init__(self):
        self._verify_init_args()
        self.result_path = os.path.join(self.result_path, self.dataset, self.model.split('/')[-1])
        os.makedirs(self.result_path, exist_ok=True)

    def _verify_init_args(self):
        assert self.model in self.all_models, f'{self.model} not in {self.all_models}'
        assert self.dataset in self.all_datasets, f'{self.dataset} not in {self.all_datasets}'
        assert self.approach in self.all_approaches, f'{self.approach} not in {self.all_approaches}'
    

class EvalEngine:
    
    def __init__(self, configs: Configs) -> None:
        self.configs = configs

    def run(self):
        logging.info(f'Running \033[32m{self.configs.approach}\033[0m on \033[32m{self.configs.dataset}\033[0m using \033[32m{self.configs.model}\033[0m')
        logging.info(f'Saving the results to \033[32m{self.configs.result_path}\033[0m')

        self._run_inference()

        self._calc_metrics()

        self._plot_figures()


    def _run_inference(self) -> None:

        # Step 1: Load the tokenizer
        # Avoid tokenization warnings (deadlock)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model,
            model_max_length=sys.maxsize,
            padding_side="right",
            trust_remote_code=True,
        )

        # Step 2: Load the dataset
        self.dataset = str2class[self.configs.dataset](tokenizer=self.tokenizer, path=self.configs.result_path)

        # Step 3: Load the model
        if 'llama' in self.configs.model.lower() or 'longchat' in self.configs.model.lower():
            enable_tuple_kv_cache_for_llama()
        if 'mistral' in self.configs.model.lower():
            enable_tuple_kv_cache_for_mistral()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs.model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Step 4: Reload the model according to the approach
        if self.configs.approach == 'quest':
            from evaluation.quest_attention import enable_quest_attention_eval
            # enable_quest_attention_eval(self.model, args)
        elif self.configs.approach == 'RaaS':
            pass
        else: # The "full" approach
            pass

        # Step 5: Assemble the pipeline with the model and the tokenizer
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Step 6: Run the inference and record results
        results = defaultdict(list)
        for i, (prompt, answer) in tenumerate(self.dataset, desc="dataset", leave=False):
            model_output = self._test_model(self.pipe, prompt, answer)
            results[f'output_{self.configs.approach}'].append(model_output)
            # TODO: Also record the time-related metrics
            # results['time'].append(time)
        
        # Step 7: Save the results
        self.dataset.update(results)
        self.dataset.save_dataset(self.configs.result_path)
            

    def _calc_metrics(self) -> None:
        pass

    def _plot_figures(self) -> None:
        pass

    
    def _test_model(self, pipe, prompt, answer) -> str:
        # response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=10)[
        #     0]["generated_text"][len(prompt_text):]

        q_length = 400
        que = prompt[-q_length:]
        text = prompt[:-q_length]
        input = pipe.tokenizer(text, return_tensors="pt").to("cuda")
        q_input = pipe.tokenizer(que, return_tensors="pt").to("cuda")
        q_input.input_ids = q_input.input_ids[:, 1:]

        with torch.no_grad():
            output = pipe.model(
                input_ids=input.input_ids,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            for input_id in q_input.input_ids[0]:
                output = pipe.model(
                    input_ids=input_id.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values

            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(10 - 1):
                outputs = pipe.model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() == pipe.tokenizer.eos_token_id:
                    break

        model_output = pipe.tokenizer.decode(generated_content, skip_special_tokens=True)
        return model_output

