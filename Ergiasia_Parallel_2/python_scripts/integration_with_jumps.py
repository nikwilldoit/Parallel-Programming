import matplotlib.pyplot as plt

threads = [1, 2, 4, 6]
mean_times_jumps = [0.3898718, 0.2169086, 0.1265182, 0.0873636]

plt.figure(figsize=(6, 4))
plt.plot(threads, mean_times_jumps, marker='o')

plt.title('Integration with jumps')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)

plt.tight_layout()
plt.savefig('integration_with_jumps_times.png', dpi=200)
plt.show()