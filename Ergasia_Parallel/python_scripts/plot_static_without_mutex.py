import matplotlib.pyplot as plt

threads = [1, 2, 4, 6]
mean_times = [0.45376125, 0.19777325, 0.11030275, 0.093134525]

plt.figure(figsize=(6, 4))
plt.plot(threads, mean_times, marker='o')

plt.title('Static integration without mutex')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)

plt.tight_layout()
plt.savefig('static_without_mutex_times.png', dpi=200)
plt.show()