import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the confusion matrix
confusion_matrix = np.array([
    [24, 0, 1, 0],
    [0, 15, 0, 0],
    [1, 0, 17, 0],
    [0, 0, 0, 6]
])

# Define the labels
labels = ["Government", "Enterprise", "Citizen", "Social Organization"]

# Create the heatmap
plt.figure(figsize=(11, 7.7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 8, "fontname": "Times New Roman"})

# Customize the font size and style for labels and title
plt.xlabel('Predicted', fontsize=8, fontname="Times New Roman")
plt.ylabel('Actual', fontsize=8, fontname="Times New Roman")
plt.title('Confusion Matrix', fontsize=8, fontname="Times New Roman")

# Customize the tick labels font size and style
plt.xticks(fontsize=8, fontname="Times New Roman")
plt.yticks(fontsize=8, fontname="Times New Roman")

# Display the plot
plt.show()
