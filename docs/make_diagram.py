"""One-off script to render docs/architecture.png for the README.
Not part of the installed package — dev/documentation tooling only."""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

BOX_STYLE = "round,pad=0.3,rounding_size=0.15"


def box(x, y, w, h, text, fc, ec="#333333", fontsize=10, fontweight="normal", text_color="black"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=BOX_STYLE, linewidth=1.4, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, color=text_color, wrap=True,
    )
    return patch


def arrow(x1, y1, x2, y2, color="#444444"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.3, color=color, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


def label(x, y, text, color="#444444", fontsize=8):
    ax.text(x, y, text, fontsize=fontsize, color=color, style="italic", ha="center", va="center")


# Title
ax.text(6, 6.7, "Product Catalog — Hybrid Multimodal Search on Azure", ha="center", fontsize=15, fontweight="bold")

# Ingestion row
box(0.3, 5.0, 2.3, 1.0, "Product Catalog\n(name, desc, price,\nimage file)", "#eef2ff")
box(3.1, 5.0, 2.3, 1.0, "Azure Blob\nStorage\n(product images)", "#dbeafe")
box(5.9, 5.0, 2.6, 1.0, "Azure AI Vision\nImage Retrieval API\n(multimodal embeddings)", "#dcfce7")
box(9.0, 5.0, 2.6, 1.0, "Azure AI Search\nindex\n(text + 2 vector fields)", "#fef9c3")

arrow(2.6, 5.5, 3.1, 5.5)
arrow(5.4, 5.5, 5.9, 5.5)
arrow(8.5, 5.5, 9.0, 5.5)
label(2.85, 4.75, "upload image")
label(5.65, 4.75, "vectorizeImage /\nvectorizeText")
label(8.75, 4.75, "upload doc")

ax.text(0.3, 4.3, "ingest.py", fontsize=9, color="#555555", style="italic")

# Query row
box(0.3, 1.8, 2.3, 1.0, "User query\n(text and/or\nuploaded photo)", "#eef2ff")
box(3.1, 1.8, 2.6, 1.0, "Azure AI Vision\n(embed query text\nor query photo)", "#dcfce7")
box(6.2, 1.8, 2.6, 1.0, "Azure AI Search\nhybrid query\n(BM25 + vector RRF +\nsemantic re-rank)", "#fef9c3")
box(9.3, 1.8, 2.3, 1.0, "Ranked results\n(name, price,\nimage, score)", "#fde2e2")

arrow(2.6, 2.3, 3.1, 2.3)
arrow(5.7, 2.3, 6.2, 2.3)
arrow(8.8, 2.3, 9.3, 2.3)
label(5.95, 1.55, "query vector(s)")

ax.text(0.3, 1.1, "search.py  (search_by_text / search_by_image / search_combined)", fontsize=9, color="#555555", style="italic")

# Connect index to query search (vertical)
arrow(10.3, 5.0, 7.5, 2.8, color="#999999")
label(9.9, 4.0, "same index\nused for queries", color="#777777")

legend_patches = [
    mpatches.Patch(color="#dbeafe", label="Storage"),
    mpatches.Patch(color="#dcfce7", label="AI / Embeddings"),
    mpatches.Patch(color="#fef9c3", label="Search index"),
    mpatches.Patch(color="#fde2e2", label="Output"),
]
ax.legend(handles=legend_patches, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=160, bbox_inches="tight")
print("Saved docs/architecture.png")
