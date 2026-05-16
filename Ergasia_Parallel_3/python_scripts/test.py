import matplotlib.pyplot as plt

# Block sizes
block_sizes = [32, 64, 128, 256, 512]

# Times for N = 100,000,000 (ΝΕΑ δεδομένα)
times_large_N = [0.534862, 0.494854, 0.489835, 0.497177, 0.485564]

# Times for N = 1,000,000 (ΝΕΑ δεδομένα)
times_small_N = [0.0059252, 0.0059711, 0.00515, 0.0049818, 0.0049477]

# ----------------------------
# Plot 1: Large N
# ----------------------------
plt.figure()
plt.plot(block_sizes, times_large_N, marker='o')
plt.title("CUDA Performance vs Block Size (N = 10^8)")
plt.xlabel("Block Size (threads per block)")
plt.ylabel("Execution Time (sec)")
plt.grid(True)
plt.xticks(block_sizes)
plt.savefig("block_size_large_N.png", dpi=300)
plt.show()

# ----------------------------
# Plot 2: Small N
# ----------------------------
plt.figure()
plt.plot(block_sizes, times_small_N, marker='o')
plt.title("CUDA Performance vs Block Size (N = 10^6)")
plt.xlabel("Block Size (threads per block)")
plt.ylabel("Execution Time (sec)")
plt.grid(True)
plt.xticks(block_sizes)
plt.savefig("block_size_small_N.png", dpi=300)
plt.show()