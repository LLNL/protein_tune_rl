# This is a python script to generate the config files.
# It generates te json files needed for to run dso

# Usage: 
# python make_config.py -r percentage_beta_sheets -o ppo -b mean -lr 1e-5 -klw 1e-2

import json
import argparse

def parse_args():
    # Optional arguments
    parser=argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument("-r", "--reward", help="reward to be optimized", default="percentage_beta_sheets")
    parser.add_argument("-o", "--optimizer", help="Optimizer for RL", default="ppo")
    parser.add_argument("-b", "--baseline", help="Baseline for PPO", default="mean")
    parser.add_argument("-lr", "--learning-rate", help="Learning rate for optimizer", type=float, default=1e-5)
    parser.add_argument("-klw", "--KL-weight", help="Weight for KL penalty term", type=float, default=1e-2)
    parser.add_argument("-klt", "--KL-target", help="Target for KL penalty term", default=None)
    args=parser.parse_args()

    if args.KL_target is not None:
        args.KL_target = float(args.KL_target)

    return args


def generate_common(metric, metric_params, tuner_type):
    # Create config file
    data = "training_data.csv" if tuner_type == "trainer" else "eval_data.csv"
    config = {
        tuner_type:
        {
            "batch_size": 8,
            "max_length": 22,
            "use_cuda": True,
            "check_point_freq": 50,
            "total_optimization_steps": 150
        },

        "experiment_directory": "output",
        "fixed_experiment_directory": True,

        "dataset":
        {
            "name": "infilling",
            "data_directory": "/g/g90/lee1029/workspace/OptLM/protein_tune_rl/data/nos_data/" + data
        },

        "tokenizer":
        {
            "vocab_file": "/usr/workspace/vaccines/abag_seq/weights/trained/iglm",
            "do_lower_case": False,
            "padding_side": "left"
        },

        "collator":
        {
            "mask_region": "HCDR3"
        },

        "policy_model":
        {
            "name": "iglm",
            "path": "/usr/workspace/vaccines/abag_seq/weights/trained/iglm"
        },

        "metric":
        [
            {
                "name": metric,
                "params": metric_params
            },
        ]
    }
    return config


def main():
    # Read arguments
    args = parse_args()

    metric_params = { }
    if args.reward == "sasa" or args.reward == "folding_confidence":
        metric_params = {
            "folding_tool": "igfold",
            "options":
            {
                "num_models": 1,
                "do_refine": False,
                "use_openmm": True,
                "do_renum": False
            }
        }

    train_config = generate_common(args.reward, metric_params, "trainer")
    
    train_config["optimizer"] = {
        "name": args.optimizer,
        "learning_rate": args.learning_rate,
        "entropy_weight": 0e-3
    }
    if args.optimizer == "ppo":
        train_config["optimizer"]["normalize_advantage"] = True
        train_config["optimizer"]["baseline"] = args.baseline
        train_config["optimizer"]["minibatch_size"] = 2

    train_config["trainer"]["name"] = "online_rl_trainer"
    train_config["KL_penalty"] = {
        "weight": args.KL_weight,
        "target": args.KL_target
    }

    with open("config_train.json", "w") as f:
        json.dump(train_config, f, indent=4)

    eval_config = generate_common(args.reward, metric_params, "evaluator")
    eval_config["evaluator"]["name"] = "online_rl_evaluator"
    eval_config["experiment_directory"] =  "output/ref"
    eval_config["metric"].append({
        "name": "prot_gpt2_scoring",
        "params": { }
    })
    eval_config["metric"].append({
        "name": "progen2_scoring",
        "params": { }
    })

    with open("config_eval_ref.json", "w") as f:
        json.dump(eval_config, f, indent=4)

    eval_config["experiment_directory"] =  "output/ft"
    eval_config["model"]["path"] = "models/final"
    with open("config_eval_ft.json", "w") as f:
        json.dump(eval_config, f, indent=4)

        
if __name__ == "__main__":
    main()
