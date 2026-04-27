import matplotlib.pyplot as plt

# Άξονας x: chunk sizes
chunks = [1, 10, 100, 1000]

# ========== STATIC, irregular different f(x) ==========
# Μέσοι χρόνοι από τα πειράματα σου
static_1 = [9.98, 9.46, 9.95, 9.64]
static_2 = [6.29, 6.14, 6.16, 6.20]
static_3 = [5.61, 5.67, 5.65, 5.77]
static_4 = [5.15, 5.06, 5.20, 5.18]

# ========== DYNAMIC, irregular different f(x) ==========
dynamic_1 = [15.51, 10.69, 9.88, 9.79]
dynamic_2 = [11.33, 6.86, 6.31, 6.33]
dynamic_3 = [10.42, 6.27, 5.77, 5.69]
dynamic_4 = [10.07, 5.42, 5.01, 5.07]

# ========== GUIDED, irregular different f(x) ==========
guided_1 = [9.79, 9.52, 9.56, 9.21]
guided_2 = [6.77, 6.80, 6.89, 7.04]
guided_3 = [5.80, 5.87, 5.85, 5.76]
guided_4 = [5.17, 5.18, 5.55, 5.24]


def plot_schedule(title, filename, t1, t2, t3, t4):
    plt.figure(figsize=(8, 5))
    plt.plot(chunks, t1, marker='o', label='1 thread')
    plt.plot(chunks, t2, marker='o', label='2 threads')
    plt.plot(chunks, t3, marker='o', label='3 threads')
    plt.plot(chunks, t4, marker='o', label='4 threads')
    plt.xscale('log')
    plt.xlabel('Chunk size')
    plt.ylabel('Execution time (s)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


# 1) Static – irregular
plot_schedule(
    'Static scheduling - irregular workload (different f(x))',
    'static_irregular.png',
    static_1, static_2, static_3, static_4
)

# 2) Dynamic – irregular
plot_schedule(
    'Dynamic scheduling - irregular workload (different f(x))',
    'dynamic_irregular.png',
    dynamic_1, dynamic_2, dynamic_3, dynamic_4
)

# 3) Guided – irregular
plot_schedule(
    'Guided scheduling - irregular workload (different f(x))',
    'guided_irregular.png',
    guided_1, guided_2, guided_3, guided_4
)

# 4) Σύγκριση των τριών για 4 threads – irregular
plt.figure(figsize=(8, 5))
plt.plot(chunks, static_4, marker='o', label='static')
plt.plot(chunks, dynamic_4, marker='o', label='dynamic')
plt.plot(chunks, guided_4, marker='o', label='guided')
plt.xscale('log')
plt.xlabel('Chunk size')
plt.ylabel('Execution time (s)')
plt.title('Static vs Dynamic vs Guided (4 threads, irregular different f(x))')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('comparison_irregular_4threads.png', dpi=200)
plt.show()