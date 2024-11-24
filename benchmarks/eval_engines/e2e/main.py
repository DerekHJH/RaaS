from benchmarks.eval_engines.eval_engine import Configs, EvalEngine

class E2EConfigs(Configs):
    pass
class E2EEvalEngine(EvalEngine):
    pass

if __name__ == "__main__":

    configs = E2EConfigs.get_configs_from_cli_args()
    eval_engine = E2EEvalEngine(configs)
    eval_engine.run()