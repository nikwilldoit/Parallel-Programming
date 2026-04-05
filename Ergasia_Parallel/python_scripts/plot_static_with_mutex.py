import matplotlib.pyplot as plt

threads = [1, 2, 4, 6]
mean_times_mutex = [0.388435, 0.197482, 0.119154, 0.088419]

plt.figure(figsize=(6, 4))
plt.plot(threads, mean_times_mutex, marker='o')

plt.title('Static integration with mutex')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)

plt.tight_layout()
plt.savefig('static_with_mutex_times.png', dpi=200)
plt.show()