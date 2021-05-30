import matplotlib.pyplot as plt
import numpy as np
import csv

NUM_SAMPLES = 30

# %%
file_path = './out4/'
names = ['random', 'sshaped', 'astar', 'inverse', 'weighted']
colors = ['k', 'g', 'b', 'm', 'c']

big_dictionary = {}
time_dictionary = {}
for name in names:
    big_dictionary[name] = {}
    time_dictionary[name] = []

    for i in range(NUM_SAMPLES):
        filename = file_path+name+'_'+str(i)+'.csv'
        with open(filename, 'r') as csvfile:
            csvfile.readline()
            reader = csv.reader(csvfile)
            data = []
            big_dictionary[name][i] = [[int(row[0]), float(row[1])] for row in reader if row]
            time_dictionary[name].append(big_dictionary[name][i].pop(-1))
            big_dictionary[name][i] = np.asarray(big_dictionary[name][i])

# %%

mean_dictionary = {}
mean_time_dictionary = {}

for name in names:
    maxlen = max([len(big_dictionary[name][i]) for i in range(NUM_SAMPLES)])
    arr = np.ma.empty((maxlen, 2, NUM_SAMPLES))
    arr.mask = True

    for i in range(NUM_SAMPLES):
        x = big_dictionary[name][i]
        arr[:x.shape[0], :x.shape[1], i] = x

    mean_masked_array = (arr.mean(axis=2))
    mean_dictionary[name] = mean_masked_array.data

    temp_array = []
    for two in time_dictionary[name]:
        one_mean = two[1]/two[0]
        temp_array.append(one_mean)

    mean_time_dictionary[name] = np.mean(temp_array)*1000000

# %%
print(big_dictionary['random'])

# %%

plt.figure(figsize=(13, 8))
for name, color in zip(names, colors):
    if name == 'sshaped':
        continue
    for i in range(NUM_SAMPLES):
        plt.plot(big_dictionary[name][i][:, 0], big_dictionary[name][i][:, 1], color, linewidth=0.17)
    plt.plot(mean_dictionary[name][:, 0], mean_dictionary[name][:, 1], color, label=name, linewidth=1)
plt.legend()
plt.xlabel('Number of moves')
plt.ylabel('Score')
plt.title('Evolution of the score for \nthe different algorithms proposed', fontsize=18)
plt.show()

