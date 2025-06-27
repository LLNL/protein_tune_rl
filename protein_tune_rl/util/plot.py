import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pareto_frontier(Xs, Ys, color, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        elif pair[1] <= pareto_front[-1][1]:
            pareto_front.append(pair)

    '''Plotting process'''
    plt.scatter(Xs, Ys, marker='.', color=color, alpha=0.5)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y, marker='o', color=color)


def plot_pareto_frontiers(
    ref_model_path, ft_model_path, output_dir, feature, likelihood
):

    ref_scores = pd.read_csv(ref_model_path / 'iglm_eval.csv').to_dict("list")
    ft_scores = pd.read_csv(ft_model_path / 'iglm_eval.csv').to_dict("list")

    ref_log_likelihood = ref_scores[likelihood]
    ref_rewards = ref_scores[feature]
    ft_log_likelihood = ft_scores[likelihood]
    ft_rewards = ft_scores[feature]

    fig = plt.figure()
    plot_pareto_frontier(ref_log_likelihood, ref_rewards, plt.cm.Blues(0.9))
    plot_pareto_frontier(ft_log_likelihood, ft_rewards, plt.cm.Oranges(0.5))

    legend = plt.legend(
        ["IgLM", "IgLM PF", "Fine-tuned IgLM", "Fine-tuned IgLM PF"],
        loc="lower right",
        fontsize=11,
    )
    for obj in legend.legend_handles:
        obj.set_linewidth(2.0)
    xlabel = "ProtGPT2" if likelihood == "prot_gpt2_scoring" else "ProGen2"
    plt.xlabel(rf'{xlabel} Log Likelihood  $\rightarrow$', fontsize=15)
    plt.ylabel(rf'{feature}  $\rightarrow$', fontsize=15)
    plt.title(f'{len(ft_rewards)} samples', fontsize=15)
    plt.savefig(
        output_dir / f"plot_pareto_front_{feature}_{likelihood}.pdf", format="pdf"
    )
    plt.close(fig)


def plot_distribution(ref_model_path, ft_model_path, output_dir):
    ref_scores = pd.read_csv(ref_model_path / 'iglm_eval.csv').to_dict("list")
    ft_scores = pd.read_csv(ft_model_path / 'iglm_eval.csv').to_dict("list")

    for metric in ref_scores:
        print("metric:", metric)
        if metric in ['Unnamed: 0', 'completion', 'HC', 'LC', 'prompts']:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        fill = True

        y_pd = pd.DataFrame({rf'{metric}  $\rightarrow$': ref_scores[metric]})
        sns.kdeplot(
            data=y_pd, x=rf'{metric}  $\rightarrow$', label='IgLM', fill=fill, ax=ax
        )
        x_pd = pd.DataFrame({rf'{metric}  $\rightarrow$': ft_scores[metric]})
        sns.kdeplot(
            data=x_pd,
            x=rf'{metric}  $\rightarrow$',
            label='Fine-tuned IgLM',
            fill=fill,
            ax=ax,
        )

        ax.legend(loc='upper left')
        plt.tight_layout()

        plt.savefig(output_dir / f"plot_dist_{metric}.pdf", format="pdf")
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rm",
        "--ref-model",
        type=str,
        default="ref_model",
        help="Path to the reference model data.",
    )
    parser.add_argument(
        "-fm",
        "--ft-model",
        type=str,
        default="ft_model",
        help="Path to the fine-tuned model data.",
    )
    parser.add_argument(
        "-pff",
        "--pareto-front-feature",
        type=str,
        default="percentage_beta_sheets",
        help="Path to the fine-tuned model data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Path where the results should be saved",
    )
    args = parser.parse_args()
    print(args)

    args.ref_model = Path(args.ref_model)
    args.ft_model = Path(args.ft_model)
    args.output_dir = Path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_distribution(args.ref_model, args.ft_model, args.output_dir)
    plot_pareto_frontiers(
        args.ref_model,
        args.ft_model,
        args.output_dir,
        args.pareto_front_feature,
        "prot_gpt2_scoring",
    )
    plot_pareto_frontiers(
        args.ref_model,
        args.ft_model,
        args.output_dir,
        args.pareto_front_feature,
        "progen2_scoring",
    )
