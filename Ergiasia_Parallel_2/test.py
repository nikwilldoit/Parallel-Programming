import matplotlib.pyplot as plt

chunks = [1, 10, 100, 1000]

static_1 = [0.9436, 0.9075, 0.9381, 0.9449]
static_2 = [0.4788, 0.4647, 0.4768, 0.4713]
static_3 = [0.4576, 0.4796, 0.4759, 0.4570]
static_4 = [0.4744, 0.5029, 0.4674, 0.4646]

dynamic_1 = [5.9997, 1.2981, 0.8976, 0.8664]
dynamic_2 = [5.9710, 0.9444, 0.5383, 0.4488]
dynamic_3 = [6.0870, 0.9374, 0.4870, 0.4429]
dynamic_4 = [5.9761, 0.9346, 0.4870, 0.4488]

guided_1 = [0.8694, 0.8804, 0.8718, 0.9014]
guided_2 = [0.4553, 0.4602, 0.4727, 0.4762]
guided_3 = [0.4496, 0.4639, 0.5054, 0.4628]
guided_4 = [0.4712, 0.4539, 0.4569, 0.4456]

def plot_schedule(title, filename, t1, t2, t3, t4):
    plt.figure(figsize=(8, 5))
    plt.plot(chunks, t1, marker='o', label='1 thread')
    plt.plot(chunks, t2, marker='o', label='2 threads')
    plt.plot(chunks, t3, marker='o', label='3 threads')
    plt.plot(chunks, t4, marker='o', label='4 threads')
    plt.xscale('log')
    plt.xlabel('Chunk size')
    plt.ylabel('Execution time (s)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

plot_schedule(
    'Static scheduling f(x)=x^2',
    'static_uniform.png',
    static_1, static_2, static_3, static_4
)

plot_schedule(
    'Dynamic scheduling f(x)=x^2',
    'dynamic_uniform.png',
    dynamic_1, dynamic_2, dynamic_3, dynamic_4
)

plot_schedule(
    'Guided scheduling f(x)=x^2',
    'guided_uniform.png',
    guided_1, guided_2, guided_3, guided_4
)

plt.figure(figsize=(8, 5))
plt.plot(chunks, static_4, marker='o', label='static')
plt.plot(chunks, dynamic_4, marker='o', label='dynamic')
plt.plot(chunks, guided_4, marker='o', label='guided')
plt.xscale('log')
plt.xlabel('Chunk size')
plt.ylabel('Execution time (s)')
plt.title('Comparison of scheduling policies (4 threads, uniform workload)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('comparison_uniform_4threads.png', dpi=200)
plt.show()