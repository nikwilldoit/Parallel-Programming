import pandas as pd
import matplotlib.pyplot as plt


rows = [
    [4, 1, 0.00465],
    [8, 1, 0.00433],
    [16, 1, 0.00281],
    [32, 1, 0.00324],
    [64, 1, 0.00395],

    [4, 2, 0.00487],
    [8, 2, 0.00365],
    [16, 2, 0.00277],
    [32, 2, 0.00314],
    [64, 2, 0.00394],

    [4, 3, 0.00440],
    [8, 3, 0.00346],
    [16, 3, 0.00275],
    [32, 3, 0.00334],
    [64, 3, 0.00395],

    [4, 4, 0.00431],
    [8, 4, 0.00329],
    [16, 4, 0.00280],
    [32, 4, 0.00310],
    [64, 4, 0.00383],
]

df = pd.DataFrame(rows, columns=["tasks", "threads", "time"])


plt.figure(figsize=(9, 6))
for th in sorted(df["threads"].unique()):
    sub = df[df["threads"] == th].sort_values("tasks")
    plt.plot(sub["tasks"], sub["time"], marker="o", label=f"threads={th}")

plt.xscale("log", base=2)
plt.xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
plt.xlabel("Tasks")
plt.ylabel("Μέσος χρόνος")
plt.title("Recursive tasks: Μέσος χρόνος ως προς τον αριθμό tasks")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("recursive_time_vs_tasks.png", dpi=300)
plt.show()


plt.figure(figsize=(9, 6))
for nt in sorted(df["tasks"].unique()):
    sub = df[df["tasks"] == nt].sort_values("threads")
    plt.plot(sub["threads"], sub["time"], marker="o", label=f"tasks={nt}")

plt.xticks([1, 2, 3, 4])
plt.xlabel("Threads")
plt.ylabel("Μέσος χρόνος")
plt.title("Recursive tasks: Μέσος χρόνος ως προς τον αριθμό threads")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("recursive_time_vs_threads.png", dpi=300)
plt.show()


sub16 = df[df["tasks"] == 16].sort_values("threads")
base_t = sub16[sub16["threads"] == 1]["time"].values[0]
speedup = base_t / sub16["time"].values

plt.figure(figsize=(7, 5))
plt.plot(sub16["threads"], speedup, marker="o")
plt.xticks([1, 2, 3, 4])
plt.xlabel("Threads")
plt.ylabel("Speedup (16 tasks)")
plt.title("Recursive tasks: Speedup ως προς threads (16 tasks)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("recursive_speedup_16tasks.png", dpi=300)
plt.show()
