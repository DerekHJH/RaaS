from benchmarks.eval_engines.eval_engine import Configs, EvalEngine
from typing import List
from dataclasses import dataclass, field

@dataclass
class E2EConfigs(Configs):
    # Overriding the default values of the parent class.
    all_datasets: List[str] = field(default_factory=lambda: ['needle', 'math500'])  # Fixed mutable default
    all_models: List[str] = field(default_factory=lambda: ['peiyi9979/mistral-7b-sft']) 
    all_approaches: List[str] = field(default_factory=lambda: ['full', 'quest'])

class E2EEvalEngine(EvalEngine):
    pass

if __name__ == "__main__":

    configs = E2EConfigs.get_configs_from_cli_args()
    eval_engine = E2EEvalEngine(configs)
    eval_engine.run()