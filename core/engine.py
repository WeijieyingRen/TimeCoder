"""
Core Engine - Experiment Pipeline Driver

Orchestrates the entire experiment generation pipeline
"""
import sys
import os
from pathlib import Path

# Auto-set project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import torch first to avoid Windows DLL issues
import torch
import numpy as np
import time
import pandas as pd

import hydra
from omegaconf import DictConfig

from llm.client import LLMClient
from models.strategies import get_strategy
from data.loader import load_environment
from tools.logger import setup_logger
from core.evaluator import run_evaluation


class ExperimentEngine:
    """Experiment Pipeline Engine"""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(config)

        # Clean up old data first
        self.cleanup_old_data()

        self.llm = LLMClient(config)

    def cleanup_old_data(self):
        """Clean up all old generated code and result files"""
        import shutil

        self.logger.info("=" * 60)
        self.logger.info("Cleaning up all old data...")
        self.logger.info("=" * 60)

        # Clean old generated code directories
        for env_name in self.config.envs_to_evaluate:
            code_dir = Path("experiments/code") / env_name
            if code_dir.exists():
                self.logger.info(f"Deleting old code directory: {code_dir}")
                shutil.rmtree(code_dir)

        # Clean old result files
        results_dir = Path("experiments/results")
        if results_dir.exists():
            for env_name in self.config.envs_to_evaluate:
                pattern = f"{env_name}*"
                for file_path in results_dir.glob(pattern):
                    if file_path.is_file():
                        self.logger.info(f"Deleting old result file: {file_path}")
                        file_path.unlink()

        # Clean old gen files in root directory
        project_root = Path(__file__).parent.parent
        for gen_file in project_root.glob("gen_*.py"):
            self.logger.info(f"Deleting root gen file: {gen_file}")
            gen_file.unlink()

        self.logger.info("=" * 60)
        self.logger.info("Old data cleanup complete!")
        self.logger.info("=" * 60)

    def run_experiment(self, env_name, method_name, seed):
        """Run a single experiment"""
        self.logger.info(f"Running {env_name} with {method_name} (seed={seed})")

        # Load environment
        env = load_environment(env_name, seed, self.config)

        # Get strategy
        strategy = get_strategy(method_name)

        # Run evaluation
        result = run_evaluation(
            env=env,
            strategy=strategy,
            llm=self.llm,
            config=self.config
        )

        return result


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main function"""
    engine = ExperimentEngine(config)

    results = []
    t0 = time.perf_counter()

    # Run all experiment combinations
    for seed in range(config.setup.seed_start, config.setup.seed_runs + config.setup.seed_start):
        for env_name in config.envs_to_evaluate:
            for method_name in config.methods_to_evaluate:
                result = engine.run_experiment(env_name, method_name, seed)
                results.append(result)

    # Log results
    time_taken = time.perf_counter() - t0
    engine.logger.info(f"Time taken: {time_taken}s")

    # Save results
    df_results = pd.DataFrame(results)
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_dir / "results.csv", index=False)

    engine.logger.info("Experiment completed.")


if __name__ == "__main__":
    main()
