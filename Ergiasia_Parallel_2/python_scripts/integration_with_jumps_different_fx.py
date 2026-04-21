import matplotlib.pyplot as plt

threads = [1, 2, 4, 6]
mean_times_jumps_fx = [13.87676, 7.06789, 3.66302, 2.47942]

plt.figure(figsize=(6, 4))
plt.plot(threads, mean_times_jumps_fx, marker='o')

plt.title('Integration with jumps different f(x)')
plt.xlabel('# threads')
plt.ylabel('Time (sec)')
plt.grid(True)

plt.tight_layout()
plt.savefig('integration_with_jumps_modified_fx_times.png', dpi=200)
plt.show()