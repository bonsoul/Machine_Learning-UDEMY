import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# --- Left: ReLU vs Sigmoid/Tanh ---
axes[0].set_title("Activation Functions", fontsize=12, weight="bold")

# Draw comparison table
rows = ["Computation speed", "Vanishing gradient", "Sparsity", "Non-linearity"]
relu = ["Very fast", "Mostly avoided", "Yes (many zeros)", "Yes"]
sigmoid_tanh = ["Slower (exp ops)", "Severe issue", "No", "Yes"]

# Display text as a grid
for i, row in enumerate(rows):
    axes[0].text(0.05, 1-(i+1)*0.2, row, fontsize=9, va="center", ha="left", weight="bold")
    axes[0].text(0.5, 1-(i+1)*0.2, relu[i], fontsize=9, va="center", ha="center", color="blue")
    axes[0].text(0.85, 1-(i+1)*0.2, sigmoid_tanh[i], fontsize=9, va="center", ha="center", color="red")

axes[0].text(0.5, 1.05, "ReLU", ha="center", fontsize=10, color="blue", weight="bold")
axes[0].text(0.85, 1.05, "Sigmoid / Tanh", ha="center", fontsize=10, color="red", weight="bold")
axes[0].axis("off")

# --- Right: Adam vs SGD ---
axes[1].set_title("Optimizers", fontsize=12, weight="bold")

rows2 = ["Learning rate", "Convergence speed", "Handles sparse gradients", "Hyperparameter tuning"]
adam = ["Adaptive (per param)", "Fast & stable", "Yes", "Minimal (default works)"]
sgd = ["Fixed (global)", "Slow (needs tuning)", "No", "Needs careful tuning"]

for i, row in enumerate(rows2):
    axes[1].text(0.05, 1-(i+1)*0.2, row, fontsize=9, va="center", ha="left", weight="bold")
    axes[1].text(0.5, 1-(i+1)*0.2, adam[i], fontsize=9, va="center", ha="center", color="blue")
    axes[1].text(0.85, 1-(i+1)*0.2, sgd[i], fontsize=9, va="center", ha="center", color="red")

axes[1].text(0.5, 1.05, "Adam", ha="center", fontsize=10, color="blue", weight="bold")
axes[1].text(0.85, 1.05, "SGD", ha="center", fontsize=10, color="red", weight="bold")
axes[1].axis("off")

plt.suptitle("ReLU vs Sigmoid/Tanh   |   Adam vs SGD", fontsize=14, weight="bold")
plt.show()
