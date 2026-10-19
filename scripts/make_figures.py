"""Generate manuscript figures from actual front-end result tables.

Use the project's analysis environment, not a model runtime.
python -B make_figures.py --output-dir <temporary figure directory> --initial
Omit --initial after the frozen confirmation result is available.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'results'
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "svg.fonttype": "none", "savefig.dpi": 300,
})
BLUE = "#1259AE"
TEAL = "#0070C0"
ORANGE = "#5699D4"
GREY = "#9CBFDF"
NAVY = "#00329F"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(output / (name + ".svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def panels(output):
    rows = read_csv(ROOT / "mechanism_panel/panel_summary.csv")
    by = {r["method"]: r for r in rows}
    keys=["construct_top10","unique_sequence_top10","equal_feature_unique_top10",
          "rule_based_unique_top10","fixed_weight_unique_top10",
          "mmr_ranking_score_lambda_0.75","mmr_ranking_score_lambda_0.5",
          "mmr_percentile_lambda_0.75","mmr_percentile_lambda_0.5"]
    labels=["Construct top ten","Sequence top ten","Equal features",
            "Rule based","Fixed weights","Score MMR 0.75","Score MMR 0.50",
            "Percentile MMR 0.75","Percentile MMR 0.50"]
    fig, axes = plt.subplots(1,2,figsize=(7.1,4.2),
                             gridspec_kw={"width_ratios":[1.05,1.22]})
    y=np.arange(len(keys))
    ax=axes[0]
    scores=np.array([float(by[k]["mean_prior_score"]) for k in keys])
    ax.hlines(y,.81,scores,color="#D6E5F5",lw=2.2,zorder=1)
    ax.scatter(scores,y,s=42,color=[NAVY,BLUE,TEAL,GREY,GREY,
                                   "#438AC6","#438AC6","#83B5DC","#83B5DC"],zorder=3,
               edgecolor="white",linewidth=.6)
    for i,val in enumerate(scores):
        ax.text(val+.006,i,f"{val:.3f}",va="center",fontsize=8,color="#173B68")
    ax.set(yticks=y,yticklabels=labels,xlim=(.805,.976),ylim=(8.7,-.7),
           xticks=[.82,.86,.90,.94],
           xlabel="Mean original priority")
    ax.tick_params(axis="y",length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#B6C9DF")
    ax.grid(axis="x",color="#E5EDF5",linewidth=.6)
    ax.set_title("A  Prior based screening score",loc="left",
                 fontweight="bold",color=NAVY,pad=13)
    ax=axes[1]
    fields=["unique_sequences","helpers","linkers","cores","terminal_states",
            "mean_pairwise_3mer_distance"]
    limits=np.array([10,4,5,10,4,1])
    matrix=np.array([[float(by[k][f]) for f in fields] for k in keys])
    colors=matrix/limits
    from matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list("maven_blues",
                                         ["#F2F7FD","#C6DFF5","#6EA8D7","#1259AE","#00329F"])
    ax.imshow(colors,vmin=0,vmax=1,cmap=cmap,aspect="auto",interpolation="nearest")
    ax.set(xticks=np.arange(6),
           xticklabels=["Seq.","Helper","Linker","Core","Ends","Distance"],
           yticks=y,yticklabels=[],ylim=(8.7,-.7))
    ax.tick_params(axis="both",length=0,pad=7)
    ax.set_xticks(np.arange(-.5,6,1),minor=True)
    ax.set_yticks(np.arange(-.5,9,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=3)
    ax.tick_params(which="minor",length=0)
    for i in range(len(keys)):
        for j in range(6):
            text=f"{matrix[i,j]:.3f}" if j==5 else f"{int(matrix[i,j])}"
            ax.text(j,i,text,ha="center",va="center",fontsize=8,
                    color="white" if colors[i,j]>.65 else "#123C68")
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title("B  Panel coverage and distance",loc="left",
                 fontweight="bold",color=NAVY,pad=13)
    fig.subplots_adjust(left=.22,right=.99,bottom=.17,top=.88,wspace=.23)
    fig.text(.22,.035,"Ten constructs per panel. Colour intensity is scaled within each coverage metric.",
             fontsize=8,color="#446384")
    save(fig,output,"figure5_panels")


def confirmation(output):
    path=ROOT / "model_confirmation/compact_summary.json"
    data=json.loads(path.read_text("utf-8"))
    models=data["models"]
    names=["base","old","candidate","public_only_deterministic_rule"]
    labels=["Qwen3-8B","R1","R6","Public rules"]
    assert all(key in models for key in names), "Public rule comparator is required"
    tasks=["plan_discovery","resolve_measurement","identity_gate","claim_scope","interpret_ranking","feedback_gate"]
    tasklabels=["Plan","Units","Identity","Evidence","Ranking","Feedback"]
    fig=plt.figure(figsize=(7.1,4.1))
    axrad=fig.add_axes([.05,.25,.36,.63],projection="polar")
    axheat=fig.add_axes([.665,.31,.32,.46])
    matrix=np.array([[models[k]["by_task"][t]["success"] for t in tasks] for k in names])
    theta=np.linspace(0,2*np.pi,6,endpoint=False)
    theta_closed=np.r_[theta,theta[0]]
    axrad.set_theta_offset(np.pi/2)
    axrad.set_theta_direction(-1)
    for idx in [1,0,3,2]:
        values=matrix[idx]/8*100
        color=[TEAL,"#B4CDE5",NAVY,"#6C9CC9"][idx]
        style=["--",":","-","-."][idx]
        axrad.plot(theta_closed,np.r_[values,values[0]],color=color,lw=2 if idx==2 else 1.2,
                   ls=style,marker="o",markersize=3,label=labels[idx])
        if idx==2:axrad.fill(theta_closed,np.r_[values,values[0]],color=color,alpha=.08)
    axrad.set(xticks=theta,xticklabels=tasklabels,ylim=(0,100),
              yticks=[25,50,75,100],yticklabels=["25","50","75","100"])
    axrad.tick_params(axis="x",labelsize=8.3,pad=5)
    axrad.tick_params(axis="y",labelsize=7,colors="#6B8199")
    axrad.grid(color="#D6E1EF",linewidth=.7)
    axrad.spines["polar"].set_color("#B8CAE1")
    handles,legend_labels=axrad.get_legend_handles_labels()
    handle_by_label=dict(zip(legend_labels,handles))
    fig.legend([handle_by_label[label] for label in labels],labels,
               loc="lower center",bbox_to_anchor=(.5,.055),ncol=4,
               frameon=False,fontsize=8,columnspacing=1.4,handlelength=2.2)
    fig.text(.035,.95,"A  Decisions across six tasks",fontweight="bold",fontsize=10,color=NAVY)
    fig.text(.56,.95,"B  Correct first answers",fontweight="bold",fontsize=10,color=NAVY)
    from matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list("maven_confirm",["#F3F7FC","#B7D7F1","#5397D0","#00329F"])
    axheat.imshow(matrix,vmin=0,vmax=8,cmap=cmap,aspect="auto")
    axheat.set(xticks=np.arange(6),xticklabels=tasklabels,
               yticks=np.arange(4),
               yticklabels=[f"{label} ({models[key]['success']}/48)"
                            for label,key in zip(labels,names)])
    axheat.tick_params(axis="x",rotation=35,labelsize=8)
    axheat.tick_params(axis="y",length=0,labelsize=8)
    axheat.set_xticks(np.arange(-.5,6,1),minor=True)
    axheat.set_yticks(np.arange(-.5,4,1),minor=True)
    axheat.grid(which="minor",color="white",linewidth=3)
    axheat.tick_params(which="minor",length=0)
    for i in range(4):
        for j in range(6):
            axheat.text(j,i,f"{matrix[i,j]}/8",ha="center",va="center",
                        color="white" if matrix[i,j]>=5 else "#173B68",fontsize=8)
    for spine in axheat.spines.values():spine.set_visible(False)
    save(fig,output,"figure3_confirmation")


def retention(output):
    rows=read_csv(ROOT/"retention_ablation/development_summary.csv")
    by={(r["arm"],r["suite"]):r for r in rows}
    assert len(by)==4 and all(int(r["actual_updates"])==128 for r in rows)
    fig,axes=plt.subplots(1,3,figsize=(7.1,3.2))
    settings=[
        ("dev","strict_success",240,"A  Decision tasks","Strict success (%)"),
        ("legacy39","strict_success",39,"B  Legacy tasks","Strict success (%)"),
        ("legacy39","legacy_field_score",1,"C  Field delivery","Partial credit score"),
    ]
    for ax,(suite,field,denom,title,ylabel) in zip(axes,settings):
        values=[float(by[arm,suite][field])/denom*(100 if denom>1 else 1) for arm in ("A","B")]
        ax.bar([0,1],values,width=.55,color=["#82B0D9",NAVY],edgecolor="white",linewidth=.8)
        ax.set(ylim=(0,110),xticks=[0,1],xticklabels=["Domain only","Replay 20%"],ylabel=ylabel,
               yticks=[0,25,50,75,100])
        ax.tick_params(axis="x",rotation=25)
        ax.spines["left"].set_color("#B3C8DF")
        ax.spines["bottom"].set_color("#B3C8DF")
        ax.set_axisbelow(True);ax.grid(axis="y",color="#E2ECF6",lw=.65)
        for i,arm in enumerate(("A","B")):
            value=by[arm,suite][field]
            label=f"{int(value)}/{denom}" if denom>1 else f"{float(value):.2f}"
            ax.text(i,values[i]+2,label,ha="center",fontsize=8,color="#173B68")
        ax.set_title(title,loc="left",fontweight="bold",color=NAVY,pad=12)
    fig.tight_layout(w_pad=2)
    save(fig,output,"figure4_retention")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--initial",action="store_true")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    panels(args.output_dir)
    if not args.initial:
        confirmation(args.output_dir)
        retention(args.output_dir)
    print("Generated figures in",args.output_dir)


if __name__=="__main__":
    main()
