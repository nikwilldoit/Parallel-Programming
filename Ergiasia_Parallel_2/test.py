import pandas as pd
import matplotlib.pyplot as plt

raw_data = {
    (4,1):  [0.0217629, 0.0082179, 0.0117415, 0.0088972, 0.0083316],
    (8,1):  [0.0097927, 0.0086169, 0.0128117, 0.016631, 0.0077858],
    (16,1): [0.0124419, 0.0262228, 0.011046, 0.0118701, 0.0162189],
    (32,1): [0.0168831, 0.0129237, 0.0114057, 0.0134544, 0.0140843],
    (64,1): [0.0180246, 0.0166085, 0.0179545, 0.0179173, 0.0158211],

    (4,2):  [0.0094093, 0.0084569, 0.0075833, 0.0087922, 0.0200049],
    (8,2):  [0.0080903, 0.0113883, 0.0085847, 0.0073171, 0.0207018],
    (16,2): [0.0153195, 0.0199227, 0.0095514, 0.0180468, 0.0108123],
    (32,2): [0.0194886, 0.0234973, 0.0202455, 0.0149817, 0.0247354],
    (64,2): [0.051629, 0.0159555, 0.020325, 0.0385895, 0.181385],

    (4,3):  [0.0115854, 0.0108148, 0.0133522, 0.0125535, 0.0193672],
    (8,3):  [0.0088569, 0.0096509, 0.0080587, 0.0079019, 0.0090149],
    (16,3): [0.0187584, 0.011338, 0.012443, 0.0097019, 0.0232578],
    (32,3): [0.0115882, 0.0133703, 0.0179451, 0.0101408, 0.0106149],
    (64,3): [0.0286079, 0.0178594, 0.0375508, 0.0133957, 0.016455],

    (4,4):  [0.0088563, 0.0097663, 0.0083165, 0.007282, 0.0080134],
    (8,4):  [0.0108864, 0.0154619, 0.0102648, 0.0090855, 0.0136355],
    (16,4): [0.0117409, 0.0115863, 0.015053, 0.0091482, 0.0097055],
    (32,4): [0.0139805, 0.0119713, 0.0107065, 0.0210389, 0.0114356],
    (64,4): [0.0161648, 0.0239692, 0.107556, 0.048871, 0.0192675],
}

rows = []
for (tasks, threads), times in raw_data.items():
    avg_time = sum(times) / len(times)
    rows.append([tasks, threads, avg_time])

df = pd.DataFrame(rows, columns=['tasks', 'threads', 'avg_time'])
df = df.sort_values(['threads', 'tasks'])

print("\nΜέσοι χρόνοι εκτέλεσης:")
print(df.to_string(index=False))

# Diagram 1: Avg time vs tasks
plt.figure(figsize=(10,6))
for th in sorted(df['threads'].unique()):
    subset = df[df['threads'] == th]
    plt.plot(subset['tasks'], subset['avg_time'], marker='o', label=f'{th} threads')

plt.title('Μέσος χρόνος (αναδρομικά tasks) ως προς τον αριθμό tasks')
plt.xlabel('Αριθμός tasks')
plt.ylabel('Μέσος χρόνος')
plt.xticks([4,8,16,32,64])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('diagram_rec_tasks_vs_time.png', dpi=300)
plt.show()

# Diagram 2: Avg time vs threads
plt.figure(figsize=(10,6))
for t in sorted(df['tasks'].unique()):
    subset = df[df['tasks'] == t]
    plt.plot(subset['threads'], subset['avg_time'], marker='o', label=f'{t} tasks')

plt.title('Μέσος χρόνος (αναδρομικά tasks) ως προς τον αριθμό threads')
plt.xlabel('Αριθμός threads')
plt.ylabel('Μέσος χρόνος')
plt.xticks([1,2,3,4])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('diagram_rec_threads_vs_time.png', dpi=300)
plt.show()

# Diagram 3: Speedup vs threads
speedup_rows = []
for t in sorted(df['tasks'].unique()):
    subset = df[df['tasks'] == t].sort_values('threads')
    t1 = subset[subset['threads'] == 1]['avg_time'].values[0]
    for _, row in subset.iterrows():
        speedup = t1 / row['avg_time']
        speedup_rows.append([t, row['threads'], speedup])

speedup_df = pd.DataFrame(speedup_rows, columns=['tasks', 'threads', 'speedup'])

plt.figure(figsize=(10,6))
for t in sorted(speedup_df['tasks'].unique()):
    subset = speedup_df[speedup_df['tasks'] == t]
    plt.plot(subset['threads'], subset['speedup'], marker='o', label=f'{t} tasks')

plt.title('Speedup (αναδρομικά tasks) ως προς τον αριθμό threads')
plt.xlabel('Αριθμός threads')
plt.ylabel('Speedup')
plt.xticks([1,2,3,4])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('diagram_rec_speedup.png', dpi=300)
plt.show()