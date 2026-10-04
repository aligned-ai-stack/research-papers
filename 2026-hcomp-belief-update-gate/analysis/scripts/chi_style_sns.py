# chi_style_sns.py
import matplotlib as mpl
import seaborn as sns
from cycler import cycler

# --- Your palettes ---
PALETTE = [
    "#FF0000","#0072B2","#D3D3D3","#1A1A1A","#00FFFF","#800080",
    "#FFD700","#006400","#E6E6FA","#000080","#006400","#FFC0CB",
    "#F4A460","#1A1A1A"
]
LINE_PALETTE = [
    "#FF0000","#0072B2","#000000","#A9A9A9",
    "#00FFFF","#800080","#FFD700","#006400",
    "#E6E6FA","#000080","#006400","#FFC0CB",
    "#F4A460","#000000"
]
CUSTOM_PALETTE = [
    "#FF0000","#000080","#0072B2","#000000","#A9A9A9",
    "#00FFFF","#800080","#FFD700","#006400","#E6E6FA",
    "#006400","#FFC0CB","#F4A460","#000000"
]
PALETTE_AD_SOURCE = ["#FF0000", "#87CEEB"]

def set_chi_style_sns(palette="custom", base_font=12, dpi=300):
    """Apply CHI-ready Seaborn + Matplotlib rcParams with bold axis labels."""
    pal = {
        "palette": PALETTE,
        "line": LINE_PALETTE,
        "custom": CUSTOM_PALETTE,
        "ad_source": PALETTE_AD_SOURCE,
    }.get(palette, CUSTOM_PALETTE)

    # Matplotlib rcParams
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.transparent": False,
        "figure.autolayout": True,

        "font.size": base_font,
        "axes.titlesize": base_font + 3,
        "axes.labelsize": base_font + 2,
        "axes.labelweight": "bold",    # << Bold axis labels
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,

        "axes.edgecolor": "#1A1A1A",
        "axes.linewidth": 1.2,
        "axes.prop_cycle": cycler(color=pal),
    })

    # Seaborn theme
    sns.set_theme(
        style="whitegrid",
        font_scale=1.0,
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.3,
        }
    )
    sns.set_palette(pal)
