import matplotlib.pyplot as plt


block_sizes = [32, 64, 128, 256, 512]


times_large_N = [
0.0753,
0.0733,
0.0736,
0.0696,
0.0585

]


times_small_N = [0.00128,
0.00099,
0.00100,
0.00094,
0.00099

]


plt.figure()
plt.plot(block_sizes, times_large_N, marker='o')
plt.title("CUDA Performance vs Block Size (N = 10^8)")
plt.xlabel("Block Size (threads per block)")
plt.ylabel("Execution Time (sec)")
plt.grid(True)
plt.xticks(block_sizes)
plt.savefig("block_size_large_N.png", dpi=300)
plt.show()


plt.figure()
plt.plot(block_sizes, times_small_N, marker='o')
plt.title("CUDA Performance vs Block Size (N = 10^6)")
plt.xlabel("Block Size (threads per block)")
plt.ylabel("Execution Time (sec)")
plt.grid(True)
plt.xticks(block_sizes)
plt.savefig("block_size_small_N.png", dpi=300)
plt.show()
