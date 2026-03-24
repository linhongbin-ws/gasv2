import os
import yaml
import numpy as np
from collections import defaultdict

EXP_DIR = "data/exp_result"


def parse_filename(fname):
    """
    Examples:
      GASv2-s0@seed11@2026_03_23-16_32_23.yml
      GASv2-BC-s0@seed11@2026_03_23-17_59_40.yml
      object_scale1-GASv2-BC-s0@seed11@2026_03_24-06_07_31.yml

    Returns:
      task_type, baseline, seed
    """
    name = fname.removesuffix(".yml").removesuffix(".yaml")
    main = name.split("@")[0]
    parts = main.split("-")

    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {fname}")

    seed = parts[-1]

    if parts[0].startswith("object_scale"):
        task_type = parts[0]
        baseline = "-".join(parts[1:-1])
    else:
        task_type = "default"
        baseline = "-".join(parts[:-1])

    return task_type, baseline, seed


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_results(exp_dir):
    results = defaultdict(lambda: defaultdict(list))

    for fname in os.listdir(exp_dir):
        if not (fname.endswith(".yml") or fname.endswith(".yaml")):
            continue

        path = os.path.join(exp_dir, fname)
        task_type, baseline, seed = parse_filename(fname)
        data = load_yaml(path)

        results[task_type][baseline].append({
            "seed": seed,
            "success_rate": float(data.get("success_rate", 0.0)),
            "score_mean": float(data.get("score_mean", 0.0)),
        })

    return results


def compute_stats(results):
    final_stats = {}

    for task_type, baselines in results.items():
        final_stats[task_type] = {}

        for baseline, runs in baselines.items():
            success_rates = np.array([r["success_rate"] for r in runs], dtype=float)
            scores = np.array([r["score_mean"] for r in runs], dtype=float)

            final_stats[task_type][baseline] = {
                "success_rate_mean": float(np.mean(success_rates)),
                "success_rate_std": float(np.std(success_rates)),
                "score_mean_mean": float(np.mean(scores)),
                "score_mean_std": float(np.std(scores)),
                "num_seeds": len(runs),
                "seeds": [r["seed"] for r in runs],
            }

    return final_stats


def print_summary(final_stats):
    for task_type, baselines in final_stats.items():
        print(f"\n=== Task: {task_type} ===")
        for baseline, stats in sorted(baselines.items()):
            print(f"{baseline}:")
            print(f"  success_rate: {stats['success_rate_mean']:.4f} ± {stats['success_rate_std']:.4f}")
            print(f"  score_mean  : {stats['score_mean_mean']:.4f} ± {stats['score_mean_std']:.4f}")
            print(f"  seeds       : {stats['num_seeds']} ({', '.join(stats['seeds'])})")


def format_cell(task_key, baseline, final_stats, metric="success_rate", scale_100=True):
    """
    Returns:
      e.g. '33.58(17.54)' or 'xx(xx)'
    """
    if task_key not in final_stats:
        return "xx(xx)"
    if baseline not in final_stats[task_key]:
        return "xx(xx)"

    stats = final_stats[task_key][baseline]

    if metric == "success_rate":
        mean = stats["success_rate_mean"]
        std = stats["success_rate_std"]
    elif metric == "score_mean":
        mean = stats["score_mean_mean"]
        std = stats["score_mean_std"]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if scale_100:
        mean *= 100.0
        std *= 100.0

    return f"{mean:.2f}({std:.2f})"


def compute_avg_row(final_stats, baseline, task_order, metric="success_rate", scale_100=True):
    """
    Average across available task rows only.
    Missing rows are ignored. If no row exists, return xx(xx).
    """
    means = []
    stds = []

    for task_key in task_order:
        if task_key in final_stats and baseline in final_stats[task_key]:
            stats = final_stats[task_key][baseline]
            if metric == "success_rate":
                means.append(stats["success_rate_mean"])
                stds.append(stats["success_rate_std"])
            elif metric == "score_mean":
                means.append(stats["score_mean_mean"])
                stds.append(stats["score_mean_std"])

    if len(means) == 0:
        return "xx(xx)"

    mean = float(np.mean(means))
    std = float(np.mean(stds))

    if scale_100:
        mean *= 100.0
        std *= 100.0

    return f"{mean:.2f}({std:.2f})"


def generate_latex_table(
    final_stats,
    metric="success_rate",
    caption="Grasping score (\\%) in simulating experiments over 5 training seeds",
    label="table:simulation_success_rate",
):
    """
    You can edit task_mapping to match your paper row names.
    Missing rows/baselines become xx(xx).
    """

    baselines = [
        "GASv2",
        "GASv1",
        "DreamerV2",
        "PPO",
        "GASv2-BC",
        "GASv2-Idle",
        "GASv2-DR",
        "GASv2-RawVR",
        "GASv2-PID",
    ]

    # Map paper rows -> task keys in filenames/stat dict
    # Edit these if your naming changes
    task_mapping = {
        "In-Dist": "default",
        "OOD-Large": "object_scale2",
        "OOD-Small": "object_scale1",
        "OOD-Shape": "object_shape",   # currently missing -> xx(xx)
        "MoveCam": "movecam",          # currently missing -> xx(xx)
        "Re-grasp": "regrasp",         # currently missing -> xx(xx)
    }

    avg_task_order = [
        task_mapping["In-Dist"],
        task_mapping["OOD-Large"],
        task_mapping["OOD-Small"],
        task_mapping["OOD-Shape"],
        task_mapping["MoveCam"],
        task_mapping["Re-grasp"],
    ]

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\renewcommand{\arraystretch}{1.6}")
    lines.append(r"\fontsize{7pt}{7pt}\selectfont")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{|l|l|c|c|c|c|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\multirow{2}{*}{\textbf{Stud}} & \multirow{2}{*}{\textbf{Type}} & \multicolumn{9}{c|}{\textbf{Baselines}} \\")
    lines.append(r"\cline{3-11}")
    lines.append(r"& & \textbf{GASv2} & \textbf{GASv1} & \textbf{DreamerV2} & \textbf{PPO} & \textbf{GASv2-BC} & \textbf{GASv2-Idle} & \textbf{GASv2-DR} & \textbf{GASv2-RawVR} & \textbf{GASv2-PID} \\")
    lines.append(r"\hline")

    def row_line(stud, row_name, task_key, multirow_prefix=None):
        vals = [format_cell(task_key, b, final_stats, metric=metric, scale_100=True) for b in baselines]
        prefix = multirow_prefix if multirow_prefix is not None else stud
        return f"{prefix}\n& {row_name}\n& " + " & ".join(vals) + r" \\"

    # Perf
    lines.append(row_line(
        r"\multirow{1}{*}{\textbf{Perf}}",
        "In-Dist",
        task_mapping["In-Dist"],
    ))
    lines.append(r"\hline")

    # Gen
    lines.append(row_line(
        r"\multirow{3}{*}{\textbf{Gen}}",
        "OOD-Large",
        task_mapping["OOD-Large"],
    ))
    lines.append(r"\cline{2-11}")
    lines.append(row_line(
        "",
        "OOD-Small",
        task_mapping["OOD-Small"],
    ))
    lines.append(r"\cline{2-11}")
    lines.append(row_line(
        "",
        "OOD-Shape",
        task_mapping["OOD-Shape"],
    ))
    lines.append(r"\hline")

    # Rob
    lines.append(row_line(
        r"\multirow{2}{*}{\textbf{Rob}}",
        "MoveCam",
        task_mapping["MoveCam"],
    ))
    lines.append(r"\cline{2-11}")
    lines.append(row_line(
        "",
        "Re-grasp",
        task_mapping["Re-grasp"],
    ))
    lines.append(r"\hline")
    lines.append(r"\hline")

    avg_vals = [
        compute_avg_row(final_stats, b, avg_task_order, metric=metric, scale_100=True)
        for b in baselines
    ]
    lines.append(r"\multicolumn{2}{|c|}{\textbf{Avg}}")
    lines.append(r"& " + " & ".join(avg_vals) + r" \\")
    lines.append(r"\hline")

    lines.append(r"")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item \textit{Note:} Mean (std) over seeds, 100 rollouts each.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    results = collect_results(EXP_DIR)
    final_stats = compute_stats(results)

    # keep current output
    print_summary(final_stats)

    # LaTeX table for success_rate
    latex_success = generate_latex_table(
        final_stats,
        metric="success_rate",
        caption="Grasping success rate (\\%) in simulating experiments over training seeds",
        label="table:simulation_success_rate",
    )

    print("\n" + "=" * 80)
    print("LATEX TABLE: SUCCESS RATE")
    print("=" * 80)
    print(latex_success)

    # Optional: another LaTeX table for score_mean
    latex_score = generate_latex_table(
        final_stats,
        metric="score_mean",
        caption="Grasping score (\\%) in simulating experiments over training seeds",
        label="table:simulation_score",
    )

    print("\n" + "=" * 80)
    print("LATEX TABLE: SCORE MEAN")
    print("=" * 80)
    print(latex_score)


if __name__ == "__main__":
    main()
