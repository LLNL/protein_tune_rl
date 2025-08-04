# This is a python script to generate the config files.
# It generates te json files needed for to run dso

# Usage:
# python make_config.py -r percentage_beta_sheets -o ppo -b mean -lr 1e-5 -klw 1e-2

import json
import argparse


def parse_args():
    # Optional arguments
    parser = argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument(
        "-r",
        "--reward",
        help="reward to be optimized",
        default="percentage_beta_sheets",
    )
    parser.add_argument("-o", "--optimizer", help="Optimizer for RL", default="ppo")
    parser.add_argument("-b", "--baseline", help="Baseline for PPO", default="mean")
    parser.add_argument(
        "-lr",
        "--learning-rate",
        help="Learning rate for optimizer",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "-klw",
        "--KL-weight",
        help="Weight for KL penalty term",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "-klt", "--KL-target", help="Target for KL penalty term", default=None
    )
    args = parser.parse_args()

    if args.KL_target is not None:
        args.KL_target = float(args.KL_target)

    return args


def generate_common(metric, metric_params, tuner_type):
    # Create config file
    data = "training_data_v3.csv" if tuner_type == "trainer" else "eval_data_50_v3.csv"
    return {
        tuner_type: {
            "batch_size": 8 if tuner_type == "trainer" else 1,
            "max_length": 22
        },
        "experiment_directory": "output",
        "fixed_experiment_directory": True,
        "dataset": {
            "name": "infilling",
            "data_directory": f"/g/g90/lee1029/workspace/OptLM/protein_tune_rl/data/nos_data/{data}",
            "chain" : "HC",
            "region" : "HCDR3"
        },
        "tokenizer": {
            "name": "iglm_tokenizer",
            "tokenizer_config": "/usr/workspace/vaccines/abag_seq/weights/trained/iglm",
            "padding_side": "left",
        },
        "collator": {"name": "infilling"},
        "policy_model": {
            "name": "iglm",
            "dir": "/usr/workspace/vaccines/abag_seq/weights/trained/iglm"
        },
        "metric": [
            {"name": metric, "params": metric_params},
        ],
    }


def main():
    # Read arguments
    args = parse_args()

    metric_params = {}
    if args.reward in ["sasa", "folding_confidence"]:
        metric_params["folding_tool"] = "igfold"
        metric_params["options"] = {
            "num_models": 1,
            "do_refine": False,
            "use_openmm": True,
            "do_renum": False,
        }

    if args.reward == "sasa":
        metric_params["mean"] = 11059.921829543871
        metric_params["std"] = 283.3540114234261
    elif args.reward == "ss_perc_sheet":
        metric_params["mean"] = 0.3784639359426365
        metric_params["std"] = 0.017445864484741506
    elif args.reward == "folding_confidence":
        metric_params["mean"] = 0.7233366161042375
        metric_params["std"] = 0.031452952774115195

    train_config = generate_common(args.reward, metric_params, "trainer")
    train_config["trainer"]["check_point_freq"] = 500_000
    train_config["trainer"]["total_optimization_steps"] = 1000
    train_config["optimizer"] = {
        "name": args.optimizer,
        "learning_rate": args.learning_rate,
        "entropy_weight": 0e-3,
    }
    if args.optimizer == "ppo":
        train_config["optimizer"]["normalize_advantage"] = False
        train_config["optimizer"]["baseline"] = args.baseline
        train_config["optimizer"]["minibatch_size"] = 2

    if args.optimizer == "dro":
        train_config["trainer"]["name"] = "dro"
        train_config["trainer"]["learning_rate"] = 1e-4
        train_config["trainer"]["tau"] = args.KL_weight # 0.75
        train_config["trainer"]["optimizer"] = "adafactor"
        train_config["trainer"]["rescaling"] = True
        train_config["trainer"]["mean_loss"] = True
        train_config["tokenizer"]["padding_side"] = "right"
        train_config["collator"]["name"] = "dro_infilling"
        train_config["dataset"]["name"] = "dro"
        train_config["dataset"]["reward"] = f"norm_{args.reward}"
        train_config["value_model"] = {
            "name" : "iglm_w_linear_head",
            "dir" : "/usr/workspace/vaccines/abag_seq/weights/trained/iglm",
            "train_all_params" : True
        }
    else:
        train_config["trainer"]["name"] = "online_rl_trainer"
        train_config["KL_penalty"] = {"weight": args.KL_weight, "target": args.KL_target}

    if args.optimizer == "reinforce":
        train_config["policy_model"]["attn_implementation"] = "sdpa"

    with open("config_train.json", "w") as f:
        json.dump(train_config, f, indent=4)

    metric_params.pop("mean")
    metric_params.pop("std")

    eval_config = generate_common(args.reward, metric_params, "evaluator")
    eval_config["evaluator"]["name"] = "iglm"
    eval_config["evaluator"]["model_name"] ="iglm"
    eval_config["tokenizer"]["name"] = "iglm_tokenizer"
    eval_config["experiment_directory"] = "output/ref"
    eval_config["metric"].append(
        {
            "name": "prot_gpt2_scoring",
            "params": {
                "model": "/usr/workspace/vaccines/abag_seq/weights/pretrained/protgpt2"
            },
        }
    )
    eval_config["metric"].append({"name": "progen2_scoring", "params": {}})
    eval_config["metric"].append(
        {
            "name": "iglm_kl_scoring",
            "params": {
                "model": "/usr/workspace/vaccines/abag_seq/weights/trained/iglm",
                "ref_model": "/usr/workspace/vaccines/abag_seq/weights/trained/iglm",
                "tokenizer": "/usr/workspace/vaccines/abag_seq/weights/trained/iglm",
            },
        }
    )

    eval_config["generator"] = {
        "num_to_generate" : 10,
        "top_p" : 1,
        "temperature" :  1,
        "max_length" : 150,
        "bad_word_ids" :
                            [[0],
                            [1],
                            [3],
                            [4],
                            [25],
                            [26],
                            [27],
                            [28],
                            [29],
                            [30],
                            [31],
                            [32]]
    }

    with open("config_eval_ref.json", "w") as f:
        json.dump(eval_config, f, indent=4)

    eval_config["experiment_directory"] = "output/ft"
    eval_config["policy_model"]["dir"] = "models/final"
    eval_config["metric"][-1]["params"]["model"] = "models/final"

    with open("config_eval_ft.json", "w") as f:
        json.dump(eval_config, f, indent=4)


if __name__ == "__main__":
    main()
