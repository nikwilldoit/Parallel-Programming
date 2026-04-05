import matplotlib.pyplot as plt

means = {
    (1, 100): 0.4029705,
    (1, 1000): 0.38808875,
    (1, 10000): 0.40770325,
    (2, 100): 0.21463725,
    (2, 1000): 0.20868425,
    (2, 10000): 0.194547,
    (4, 100): 0.119066,
    (4, 1000): 0.10109303,
    (4, 10000): 0.10634975,
    (6, 100): 0.1152155,
    (6, 1000): 0.07442542,
    (6, 10000): 0.07793078,
}

threads = [1, 2, 4, 6]
K_values = [100, 1000, 10000]

plt.figure(figsize=(6, 4))
for K in K_values:
    y = [means[(t, K)] for t in threads]
    plt.plot(threads, y, marker='o', label=f'K = {K}')

plt.title('Dynamic queue')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('dynamic_queue_times.png', dpi=200)
plt.show()