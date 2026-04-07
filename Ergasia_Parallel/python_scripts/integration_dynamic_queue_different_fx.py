import matplotlib.pyplot as plt

threads = [1, 2, 4, 6]
K_values = [100, 1000, 10000]
means = {
    (1, 100): 0.4001405,
    (1, 1000): 0.388314,
    (1, 10000): 0.38983725,
    (2, 100): 0.21697525,
    (2, 1000): 0.19625825,
    (2, 10000): 0.198756,
    (4, 100): 0.12006575,
    (4, 1000): 0.104541,
    (4, 10000): 0.106349,
    (6, 100): 0.1202375,
    (6, 1000): 0.072921025,
    (6, 10000): 0.07582015,
}

plt.figure(figsize=(6, 4))
for K in K_values:
    y = [means[(t, K)] for t in threads]
    plt.plot(threads, y, marker='o', label=f'K = {K}')

plt.title('Dynamic queue different f(x)')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('dynamic_queue_modified_fx_times.png', dpi=200)
plt.show()