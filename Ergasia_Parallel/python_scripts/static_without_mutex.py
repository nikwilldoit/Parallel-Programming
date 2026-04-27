import matplotlib.pyplot as plt

threads = [1, 2, 3, 4]
mean_times = [0.8264, 0.4339, 0.4246, 0.4222]

plt.figure(figsize=(6, 4))
plt.plot(threads, mean_times, marker='o')

plt.title('Parallel for without reduction (critical)')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)

plt.tight_layout()
plt.savefig('parallel_for_without_reduction_times.png', dpi=200)
plt.show()