import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Arquivos CSV
activation_files = {
    "relu": "l1_relu.csv",
    "tanh": "l1_tanh.csv",
    "sigmoid": "l1_sigmoid.csv",
    "cauchy": "l1_cauchy.csv",
}

# Carrega os DataFrames
dfs = {act: pd.read_csv(path) for act, path in activation_files.items()}

# Métricas para plotar
metrics = ["loss", "avg_reward", "grad_norm", "grad_to_param_ratio"]

for metric in metrics:
    plt.figure(figsize=(10, 6))

    for act, df in dfs.items():
        plt.plot(df["step"], df[metric], label=act)

    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Step (Todas as ativações)")

    plt.yscale("log")  # Escala log no eixo Y

    # ---- Ajuste dos ticks ----
    # X: tick a cada 25
    max_x = max(df["step"].max() for df in dfs.values())
    plt.xticks(np.arange(0, max_x + 1, 25))

    # Y: ticks automáticos (mas garantimos grade pontilhada)
    ax = plt.gca()

    # ---- Grade ----
    ax.grid(which="major", axis="both", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.4)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric}_combined.png", dpi=300)
    plt.close()

print("Feito!")