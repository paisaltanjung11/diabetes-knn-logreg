import matplotlib.pyplot as plt
import numpy as np

metrics = ['Accuracy', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']

knn_values = [0.72, 0.68, 0.39, 0.49]

logreg_values = [0.701, 0.59, 0.50, 0.54]

x = np.arange(len(metrics)) 
width = 0.35 

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, knn_values, width, label='KNN', color='skyblue')
bars2 = ax.bar(x + width/2, logreg_values, width, label='Logistic Regression', color='salmon')

ax.set_ylabel('Score')
ax.set_title('Perbandingan Performa KNN vs Logistic Regression')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend()

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.show()
