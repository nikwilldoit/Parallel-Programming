import matplotlib.pyplot as plt

threads = [1, 2, 4, 6]
mean_times_jumps = [0.390083, 0.219739, 0.127647, 0.088956]

plt.figure(figsize=(6, 4))
plt.plot(threads, mean_times_jumps, marker='o')

plt.title('Integration with jumps')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)

plt.tight_layout()
plt.savefig('integration_with_jumps_times.png', dpi=200)
plt.show()