# import logging
# from dataclasses import dataclass, field
# from typing import List

# from benchmarks.eval_engines.eval_engine import Configs, EvalEngine

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


# @dataclass
# class MarkovConfigs(Configs):
#     # Overriding the default values of the parent class.
#     tot_num_data: int = 1
#     all_datasets: List[str] = field(default_factory=lambda: ["math500"])  # Fixed mutable default
#     all_models: List[str] = field(default_factory=lambda: ["peiyi9979/mistral-7b-sft"])
#     all_approaches: List[str] = field(default_factory=lambda: ["full"])


# class MarkovEvalEngine(EvalEngine):

#     def run(self):
#         logging.info(
#             (
#                 f"Evaluate \033[32m{self.configs.approach}\033[0m on"
#                 f" \033[32m{self.configs.model}\033[0m and"
#                 f" \033[32m{self.configs.dataset}\033[0m"
#             )
#         )
#         logging.info(f"Save the results to \033[32m{self.configs.result_path}\033[0m")

#         logger.debug("Step 1: Load the tokenizer")
#         # Avoid tokenization warnings (deadlock)
#         os.environ["TOKENIZERS_PARALLELISM"] = "true"
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.configs.model,
#             model_max_length=sys.maxsize,
#             padding_side="right",
#             trust_remote_code=True,
#         )

#         logger.debug("Step 2: Load the dataset, preprocess the data and save the preprocced data")
#         self.dataset = str2class[self.configs.dataset](
#             tokenizer=self.tokenizer,
#             path=self.configs.result_path,
#             tot_num_data=self.configs.tot_num_data,
#         )
#         # ckpt 1: dataset preprocessed
#         self.dataset.save_dataset(self.configs.result_path)

#         logger.debug("Step 3: Load the model")
#         if "llama" in self.configs.model.lower() or "longchat" in self.configs.model.lower():
#             enable_tuple_kv_cache_for_llama()
#         if "mistral" in self.configs.model.lower():
#             enable_tuple_kv_cache_for_mistral()

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.configs.model,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True,
#         )

#         logger.debug("Step 4: Reload the model according to the approach")
#         if self.configs.approach == "quest":
#             from evaluation.quest_attention import enable_quest_attention_eval

#             enable_quest_attention_eval(self.model, self.configs)
#         elif self.configs.approach == "raas":
#             from evaluation.raas_attention import enable_raas_attention_eval

#             enable_raas_attention_eval(self.model, self.configs)
#         else:  # The "full" approach
#             pass

#         logger.debug("Step 5: Assemble the pipeline with the model and the tokenizer")
#         self.pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             pad_token_id=self.tokenizer.eos_token_id,
#         )

#         logger.debug("Step 6: Run the inference and record results")
#         results = defaultdict(list)
#         for i, (prompt, answer) in tenumerate(self.dataset, desc="dataset", leave=False):
#             model_output, TTFT, JCT, TPOT, num_decode = self._test_model(
#                 self.pipe, prompt, answer)
#             results[f"output_{self.configs.approach}"].append(model_output)
#             # TODO: Also record the time-related metrics
#             results[f"TTFT_{self.configs.approach}"].append(TTFT)
#             results[f"JCT_{self.configs.approach}"].append(JCT)
#             results[f"TPOT_{self.configs.approach}"].append(TPOT)
#             results[f"num_decode_{self.configs.approach}"].append(num_decode)

#         logger.debug("Step 7: Save the results")
#         self.dataset.update(results)

#         # ckpt 2: dataset augmented with inference results
#         self.dataset.save_dataset(self.configs.result_path)

#         logger.debug("Step 8: Calculate the accuracy for the model outputs.")
#         self.dataset.calc_accuracy(self.configs.approach)

#         # ckpt 3: dataset augmented with accuracy
#         self.dataset.save_dataset(self.configs.result_path)

#         # Print some aggregate information
#         accuracy_avg = np.mean(self.dataset.data[f"accuracy_{self.configs.approach}"])
#         TTFT_avg = np.mean(self.dataset.data[f"TTFT_{self.configs.approach}"])
#         JCT_avg = np.mean(self.dataset.data[f"JCT_{self.configs.approach}"])
#         TPOT_avg = np.mean(self.dataset.data[f"TPOT_{self.configs.approach}"])
#         num_decode_avg = np.mean(self.dataset.data[f"num_decode_{self.configs.approach}"])
#         logger.info(f"Average accuracy of {self.configs.approach}: {accuracy_avg:.3f}")
#         logger.info(f"Average TTFT of {self.configs.approach}: {TTFT_avg:.2f} s")
#         logger.info(f"Average JCT of {self.configs.approach}: {JCT_avg:.2f} s")
#         logger.info(f"Average TPOT of {self.configs.approach}: {TPOT_avg:.2f} s")
#         logger.info(f"Average num_decode of {self.configs.approach}: {num_decode_avg:.2f}")


# if __name__ == "__main__":

#     configs = MarkovConfigs.get_configs_from_cli_args()
#     eval_engine = MarkovEvalEngine(configs)
#     eval_engine.run()
