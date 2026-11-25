import matplotlib.pyplot as plt

data = [
    ["Baseline (Validation Set)", "0.9882", "0.9992", "0.0127"],
    ["FGSM (Adversarial Attack)", "0.9070", "0.8920", "0.1626"]
]
columns = ["Evaluation Setting", "Accuracy", "AUC", "EER"]

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis('tight')
ax.axis('off')

plt.title("Table 11: Inference-Time Robustness Stress Test\nunder baseline validation conditions and white-box FGSM adversarial attack.", 
          pad=15, weight='bold', fontsize=12)

table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Automatically adjust column widths based on text
table.auto_set_column_width(col=list(range(len(columns))))

for (row, col), cell in table.get_celld().items():
    # Make header bold and gray
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e0e0e0')
    # Make first column bold
    if col == 0 and row > 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f9f9f9')
        
plt.tight_layout()
plt.savefig("table_11.png", dpi=300, bbox_inches='tight')
print("Image saved as table_11.png")
