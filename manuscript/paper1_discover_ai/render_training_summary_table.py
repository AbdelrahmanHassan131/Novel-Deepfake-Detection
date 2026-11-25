import matplotlib.pyplot as plt

data = [
    ["1", "Spatial RGB Expert", "Unfrozen (End-to-End)", "Sub-model Train/Val"],
    ["1", "Frequency Expert", "Unfrozen (End-to-End)", "Sub-model Train/Val"],
    ["2", "Spatial RGB Expert", "Frozen", "N/A"],
    ["2", "Frequency Expert", "Frozen", "N/A"],
    ["2", "MHA Fusion Module", "Unfrozen (Trained)", "MHA Train/Val (80/20)"],
    ["3", "End-to-End Pipeline", "Evaluation Only", "MHA Test/Validation"]
]
columns = ["Phase", "Component", "State", "Dataset Split"]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis('tight')
ax.axis('off')

plt.title("Compact summary of the training protocol,\ndetailing the freezing stages and dataset utilization.", 
          pad=20, weight='bold', fontsize=13)

table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Automatically adjust column widths based on text
table.auto_set_column_width(col=list(range(len(columns))))

for (row, col), cell in table.get_celld().items():
    # Make header bold with soft gray background
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e0e0e0')
    else:
        # Phase column styling
        if col == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f9f9f9')
        # Bolding for 'Frozen' state
        if col == 2 and data[row - 1][2] == "Frozen":
            cell.set_text_props(weight='bold')
        # Alternating row background for readability
        if row % 2 == 0 and col != 0:
            cell.set_facecolor('#fcfcfc')

plt.tight_layout()
output_path = "table_training_summary.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved successfully as {output_path}")
