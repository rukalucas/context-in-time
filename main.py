"""Main training script. Create tasks, model, and trainer from OmegaConf config, then train."""

import sys
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig


def create_tasks(conf: DictConfig) -> list:
    """Create task instances from config."""
    from tasks import TransitionTask, InferredTask, InstructedTask, SingleTrialTask

    task_name_to_class = {
        "transition": TransitionTask,
        "inferred": InferredTask,
        "instructed": InstructedTask,
        "single_trial": SingleTrialTask,
    }
    tasks = []
    for task_spec in conf.tasks:
        task_type = task_spec.task_type
        assert task_type in task_name_to_class, f"Unknown task_type: {task_type}"
        task = task_name_to_class[task_type](**task_spec.get("task", {}))
        tasks.append((task, task_type))
    return tasks


def create_model(conf: DictConfig):
    """Create model instance from config."""
    from models.rnn import RNN
    from models.modular_rnn import ModularRNN

    model_type = conf.model.get("model_type", "rnn")

    if model_type == "rnn":
        # Remove model_type from params before passing to constructor
        model_params = {k: v for k, v in conf.model.items() if k != "model_type"}
        return RNN(**model_params)
    elif model_type == "modular_rnn":
        model_params = {k: v for k, v in conf.model.items() if k != "model_type"}
        return ModularRNN(**model_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_trainer(conf: DictConfig, model, log_dir: str = None):
    """Create trainer instance from config."""
    from trainers import ParallelTrainer, SequentialTrainer, OrthogonalSequentialTrainer

    # Get trainer type
    trainer_type = conf.get("trainer_type", "parallel")
    # Create tasks
    tasks_with_names = create_tasks(conf)
    tasks = [t for t, _ in tasks_with_names]
    task_names = [name for _, name in tasks_with_names]
    # Get log_dir
    if log_dir is None:
        log_dir = conf.training.get("log_dir", "logs")

    # Extract eval flags (default True for each task)
    task_eval_flags = [spec.get("eval", True) for spec in conf.tasks]

    # Build trainer params dict
    trainer_params = {
        "model": model,
        "tasks": tasks,
        "task_names": task_names,
        "task_eval_flags": task_eval_flags,
        "log_dir": log_dir,
        "config": OmegaConf.to_container(conf, resolve=True),
        **conf.training,
    }

    # Create appropriate trainer
    if trainer_type == "parallel":
        weights = [spec.weight for spec in conf.tasks]
        assert abs(sum(weights) - 1.0) < 1e-2, "Task weights must roughly sum to 1.0"
        return ParallelTrainer(task_weights=weights, **trainer_params)
    elif trainer_type in ["sequential", "orthogonal_sequential"]:
        num_steps = [spec.num_steps for spec in conf.tasks]
        schedules = [spec.get("schedule") for spec in conf.tasks]
        total_steps = conf.training.get("total_steps", 0)
        assert sum(num_steps) == total_steps, (
            "Sum of task num_steps must equal total_steps"
        )
        if trainer_type == "sequential":
            return SequentialTrainer(
                task_num_steps=num_steps,
                task_param_schedules=schedules,
                **trainer_params,
            )
        else:  # orthogonal_sequential
            return OrthogonalSequentialTrainer(
                task_num_steps=num_steps,
                task_param_schedules=schedules,
                **trainer_params,
            )
    else:
        raise ValueError(f"Unknown trainer_type: {trainer_type}")


# ============================================================================
# Main script
# ============================================================================


def set_random_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_log_dir_exists(log_dir: Path) -> bool:
    """Check if log directory exists and prompt user for confirmation.

    Args:
        log_dir: Path to the log directory

    Returns:
        True if user wants to proceed (or dir doesn't exist), False otherwise
    """
    if not log_dir.exists():
        return True

    # Warn user
    print(f"\nWARNING: Log directory already exists: {log_dir}")
    print("Proceeding will overwrite existing files.")

    # Prompt for confirmation
    response = input("Do you want to proceed? (y/n): ").strip().lower()

    if response == "y":
        return True
    else:
        print("\nTraining cancelled.")
        return False


def main():
    """Main training script with OmegaConf CLI interface.

    Usage:
        # Fresh training
        python main.py configs/instructed.yaml

        # With parameter overrides
        python main.py configs/instructed.yaml model.hidden_size=256 training.total_steps=1000

        # Load checkpoint as initialization (new wandb run, step=0)
        python main.py configs/instructed.yaml training.from_checkpoint=path/to/checkpoint.pt

        # Resume from checkpoint (continue same wandb run and step count)
        # New checkpoints (with wandb_run_id saved):
        python main.py configs/instructed.yaml training.from_checkpoint=path/to/checkpoint.pt training.resume=true

        # Resume from old checkpoint (without wandb_run_id, manually specify run ID):
        python main.py configs/instructed.yaml \
            training.from_checkpoint=path/to/old_checkpoint.pt \
            training.resume=true \
            training.resume_run_id=abc123xyz
    """
    # Parse CLI
    if len(sys.argv) < 2 or "=" in sys.argv[1]:
        print(
            "Error: Config file required!\nUsage: python main.py configs/instructed.yaml [key=value overrides...]"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    cli_args = sys.argv[2:]

    # Load config
    print(f"Loading config from {config_path}")
    conf = OmegaConf.load(config_path)

    # Merge CLI overrides
    if cli_args:
        conf = OmegaConf.merge(conf, OmegaConf.from_cli(cli_args))

    # Set random seed
    random_seed = conf.get("random_seed", 42)
    set_random_seed(random_seed)
    print(f"Random seed: {random_seed}")

    # Setup log directory
    log_dir = Path(conf.training.get("log_dir", "logs/test"))

    # Check if log_dir exists and get user confirmation (skip if resuming)
    resume = conf.training.get("resume", False)
    if not resume and not check_log_dir_exists(log_dir):
        sys.exit(1)

    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config (skip if resuming - preserve original config)
    if not resume:
        OmegaConf.save(conf, log_dir / "config.yaml")
    else:
        print(f"Resuming run - using existing config from {log_dir / 'config.yaml'}")

    # Create model and trainer using factory functions
    model = create_model(conf)
    trainer = create_trainer(conf, model)

    # Train!
    trainer.train()


if __name__ == "__main__":
    main()
