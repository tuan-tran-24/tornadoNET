import os
from pathlib import Path
import matplotlib.pyplot as plt

colors = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green']

def count_recovery_state_folders(
    root,
    recovery_states,
):
    ex = {e.lower() for e in "Surrounding"}
    root = Path(root)

    counts = []
    for rs in recovery_states:
        p = root / rs
        if not p.is_dir():
            counts.append(0)
            continue
        counts.append(sum(1 for d in p.iterdir() if d.is_dir() and d.name.lower() not in ex))
    return counts

def plot_counts(
    roots,
    recovery_states,
    title,
    save_path,
):
    
    years = [year for year, _ in roots]
    data = [
        count_recovery_state_folders(r, recovery_states=recovery_states)
        for _, r in roots
    ]  
    
    Y, S = len(years), len(recovery_states)

    x = list(range(Y))
    group_w = 0.82
    bar_w = group_w / max(S, 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for rs_index, rs in enumerate(recovery_states):
        vals = [data[yi][rs_index] for yi in range(Y)]
        offs = [xi - group_w / 2 + (rs_index + 0.5) * bar_w for xi in x]
        ax.bar(offs,
               vals,
               width=bar_w,
               color=colors[rs_index % len(colors)],
               )

        bump = max(0.5, 0.01 * max(vals or [1]))
        for xo, v in zip(offs, vals):
            ax.text(xo,
                    v + bump,
                    str(v),
                    ha="center",
                    va="bottom",
                    fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Buildings")
    ax.set_title(title)

    ax.legend(title="Recovery States", labels=recovery_states)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
    plt.show()
