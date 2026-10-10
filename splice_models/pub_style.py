# shared palette + style for the publication figures
import matplotlib.pyplot as plt
PALETTE = {
    "ink":    "#22313F",
    "rf":     "#264653",
    "splice": "#2A9D8F",
    "splice2":"#1F7A6E",
    "gold":   "#E9C46A",
    "sand":   "#F4A261",
    "canon":  "#E76F51",
    "grey":   "#C7CDD1",
    "greyd":  "#9AA5AD",
    "cream":  "#FBF7EF",
    "line":   "#DCE3E6",
}
def set_style():
    plt.rcParams.update({
        "font.family":"sans-serif",
        "font.sans-serif":["DejaVu Sans","Arial","Helvetica"],
        "font.size":9,"axes.titlesize":10.5,"axes.labelsize":9.5,
        "legend.fontsize":8.5,"xtick.labelsize":8.5,"ytick.labelsize":8.5,
        "axes.linewidth":0.9,"axes.edgecolor":PALETTE["ink"],
        "figure.dpi":150,"savefig.dpi":300,"pdf.fonttype":42,"ps.fonttype":42,
        "axes.titlecolor":PALETTE["ink"],"text.color":PALETTE["ink"],
        "axes.labelcolor":PALETTE["ink"],"xtick.color":PALETTE["ink"],"ytick.color":PALETTE["ink"],
    })
def clean(ax, grid=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3,width=0.9)
    if grid:
        ax.yaxis.grid(True,color=PALETTE["line"],lw=0.8,zorder=0); ax.set_axisbelow(True)