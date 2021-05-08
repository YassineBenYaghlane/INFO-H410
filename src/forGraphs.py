import matplotlib.pyplot as plt
import numpy as np
import csv

NUM_SAMPLES = 10

# %%
file_path = './out1/'
names = ['astar', 'inverse', 'sshaped', 'weighted']

big_dictionary = {}
for name in names:
    big_dictionary[name] = {}

    for i in range(NUM_SAMPLES):
        filename = file_path+name+'_'+str(i)+'.csv'
        with open(filename, 'r') as csvfile:
            csvfile.readline()
            reader = csv.reader(csvfile)
            data = []
            big_dictionary[name][i] = [[int(row[0]), int(row[1])] for row in reader if row]
            big_dictionary[name][i] = np.asarray(big_dictionary[name][i])

# %%

mean_dictionary = {}

for name in names:
    maxlen = max([len(big_dictionary[name][i]) for i in range(NUM_SAMPLES)])
    arr = np.ma.empty((maxlen, 2, NUM_SAMPLES))
    arr.mask = True

    for i in range(NUM_SAMPLES):
        x = big_dictionary[name][i]
        arr[:x.shape[0], :x.shape[1], i] = x

    mean_masked_array = (arr.mean(axis=2))
    mean_dictionary[name] = mean_masked_array.data

# %%

plt.figure()
for name in names:
    plt.plot(mean_dictionary[name][:, 0], mean_dictionary[name][:, 1], label=name)
plt.legend()
plt.show()

