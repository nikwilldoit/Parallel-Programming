import matplotlib.pyplot as plt

# Block sizes
block_sizes = [32, 64, 128, 256, 512]

# Times for N = 100,000,000 (ΝΕΑ δεδομένα)
times_large_N = [
0.0753,
0.0733,
0.0736,
0.0696,
0.0585

]

# Times for N = 1,000,000 (ΝΕΑ δεδομένα)
times_small_N = [0.00128,
0.00099,
0.00100,
0.00094,
0.00099

]

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