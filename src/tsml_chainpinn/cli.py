import argparse
import json
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="TSML-ChainPINN paper implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Validate explicit configuration")
    check.add_argument("--config", required=True)
    prepare = sub.add_parser("prepare", help="Convert actual CFD meshes with explicit field/unit mapping")
    prepare.add_argument("--source-manifest", required=True)
    prepare.add_argument("--output", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--mode", choices=["selection", "final"], default="final")
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume", action="store_true", help="Continue output/latest.pt at an epoch boundary")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=4096)
    evaluate.add_argument("--save-predictions", action="store_true")
    sub.add_parser("doctor", help="Show local runtime, GPU and optional dependencies")
    demo = sub.add_parser("demo", help="Run a reduced synthetic workflow, never a paper benchmark")
    demo.add_argument("--output", required=True)
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--plot", action="store_true")
    inspect = sub.add_parser("inspect-data", help="Validate all prepared snapshots")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--verify-hashes", action="store_true")
    experiment = sub.add_parser("experiment", help="Train/evaluate independent seeds and aggregate")
    experiment.add_argument("--config", required=True)
    experiment.add_argument("--manifest", required=True)
    experiment.add_argument("--output", required=True)
    experiment.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 3407])
    experiment.add_argument("--mode", choices=["selection", "final"], default="final")
    experiment.add_argument("--device", default="cpu")
    aggregate = sub.add_parser("aggregate", help="Summarize existing independent-seed evaluations")
    aggregate.add_argument("--summaries", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    plot = sub.add_parser("plot", help="Plot saved training logs and evaluation metrics")
    plot.add_argument("--run", required=True)
    plot.add_argument("--evaluation", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            cfg = load_config(args.config)
            from .model import ChainPINN
            model = ChainPINN(**cfg["model"])
            print(json.dumps({"status": "configuration_valid_not_a_reproduction_claim",
                              "parameters": sum(p.numel() for p in model.parameters())}, indent=2))
        elif args.command == "prepare":
            from .prepare import prepare as run
            print(run(args.source_manifest, args.output))
        elif args.command == "train":
            from .engine import train as run
            print(run(load_config(args.config), args.manifest, args.output, args.mode, args.seed, args.device, args.resume))
        elif args.command == "evaluate":
            from .engine import evaluate as run
            print(json.dumps(run(args.checkpoint, args.manifest, args.output, args.device,
                                 args.batch_size, args.save_predictions), indent=2))
        elif args.command == "doctor":
            from .diagnostics import doctor
            print(json.dumps(doctor(), indent=2))
        elif args.command == "inspect-data":
            from .diagnostics import inspect_data
            print(json.dumps(inspect_data(args.manifest, args.verify_hashes), indent=2))
        elif args.command == "demo":
            from .demo import run_demo
            print(json.dumps(run_demo(args.output, args.device, args.plot), indent=2))
        elif args.command == "experiment":
            from .experiments import run_experiment
            print(json.dumps(run_experiment(load_config(args.config), args.manifest, args.output,
                                            args.seeds, args.mode, args.device), indent=2))
        elif args.command == "aggregate":
            from .experiments import aggregate
            print(json.dumps(aggregate(args.summaries, args.output), indent=2))
        elif args.command == "plot":
            from .reporting import plot_run
            print(plot_run(args.run, args.evaluation, args.output))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
