import matplotlib.pyplot as plt

threads = [1, 2, 4, 6]
mean_times = [0.39679196, 0.19769661, 0.10887460, 0.09161740]

plt.figure(figsize=(6, 4))
plt.plot(threads, mean_times, marker='o')

plt.title('Static integration without mutex')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)

plt.tight_layout()
plt.savefig('static_without_mutex_times.png', dpi=200)
plt.show()