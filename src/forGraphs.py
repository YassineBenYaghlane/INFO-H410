import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.patches import Ellipse

NUM_SAMPLES = 30

# %%
file_path = './compare/'
#names = ['random', 'sshaped', 'astar', 'inverse', 'weighted']
names = ['survival', 'sshaped', 'astar', 'inverse', 'weighted']

colors = ['k', 'g', 'b', 'm', 'c']

big_dictionary = {}
time_dictionary = {}
speed_dictionary = {}
for name in names:
    big_dictionary[name] = {}
    speed_dictionary[name] = {}
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
        speed_dictionary[name][i] = [[tup[0], (tup[1] / tup[0])] for tup in big_dictionary[name][i] if tup[0] != 0]
        speed_dictionary[name][i] = np.asarray(speed_dictionary[name][i])

# %%

mean_dictionary = {}
mean_speed_dictionary = {}
mean_time_dictionary = {}

for name in names:

    ### MEAN
    maxlen = max([len(big_dictionary[name][i]) for i in range(NUM_SAMPLES)])
    arr = np.ma.empty((maxlen, 2, NUM_SAMPLES))
    arr2 = np.ma.empty((maxlen, 2, NUM_SAMPLES))
    arr.mask = True
    arr2.mask = True

    for i in range(NUM_SAMPLES):
        x = big_dictionary[name][i]
        y = speed_dictionary[name][i]
        arr[:x.shape[0], :x.shape[1], i] = x
        arr2[:y.shape[0], :y.shape[1], i] = y


    mean_masked_array = (arr.mean(axis=2))
    mean_speed_masked_array = (arr2.mean(axis=2))
    mean_dictionary[name] = mean_masked_array.data
    mean_speed_dictionary[name] = mean_speed_masked_array.data

    ### TIME
    temp_array = []
    for two in time_dictionary[name]:
        one_mean = two[1]/two[0]
        temp_array.append(one_mean)

    mean_time_dictionary[name] = np.mean(temp_array)*1000000



# %%
print(mean_time_dictionary['astar'])

# %%

plt.figure(figsize=(13, 11))
for name, color in zip(names, colors):

    for i in range(NUM_SAMPLES):
        plt.plot(big_dictionary[name][i][:, 0], big_dictionary[name][i][:, 1], color, linewidth=0.17)
    plt.plot(mean_dictionary[name][:, 0], mean_dictionary[name][:, 1], color, label=name, linewidth=1)
plt.legend(prop={'size': 20})
plt.xlabel('Number of moves', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Evolution of the score for \nthe different algorithms proposed', fontsize=24)

plt.savefig('thefigure.png', bbox_inches='tight')
plt.show()

# %%
### SPEED

plt.figure(figsize=(13, 11))
for name, color in zip(names, colors):
    plt.plot(mean_speed_dictionary[name][:-2, 0], mean_speed_dictionary[name][:-2, 1], color, label=name, linewidth=1)
plt.legend(prop={'size': 20})
plt.xlabel('Number of moves', fontsize=14)
plt.ylabel('Speed', fontsize=14)
plt.title('Evolution of the speed for \nthe different algorithms proposed', fontsize=24)

plt.savefig('thespeed.png', bbox_inches='tight')
plt.show()

# %%
### VARIANCE
end_dictionary = {}

for name in names:
    end_dictionary[name] = {}
    end_dictionary[name]['moves mean'] = np.mean([big_dictionary[name][i][-1, 0] for i in range(NUM_SAMPLES)])
    end_dictionary[name]['score mean'] = np.mean([big_dictionary[name][i][-1, 1] for i in range(NUM_SAMPLES)])
    end_dictionary[name]['moves var'] = np.var([big_dictionary[name][i][-1, 0] for i in range(NUM_SAMPLES)])
    end_dictionary[name]['score var'] = np.var([big_dictionary[name][i][-1, 1] for i in range(NUM_SAMPLES)])

plt.figure(figsize=(13, 11))
ax = plt.gca()
for name, color in zip(names, colors):
    a = end_dictionary[name]['moves mean']
    b = end_dictionary[name]['score mean']
    mvar = end_dictionary[name]['moves var']
    svar = end_dictionary[name]['score var']
    ax.plot(a, b, color+'o', label=name, linewidth=5)
    ax.add_patch(Ellipse((a, b), np.sqrt(mvar), np.sqrt(svar), edgecolor=color, fc='None', lw=2))
ax.legend(prop={'size': 20})
plt.xlabel('Number of moves', fontsize=14)
plt.ylabel('Final score', fontsize=14)
plt.title('Mean and variance of the final score for \nthe different algorithms proposed', fontsize=24)

plt.savefig('thevar.png', bbox_inches='tight')
plt.show()

# %%
### TIME
with open('times.txt', 'w') as file:
    file.write('\\underline{\\textbf{Av. time per move}} & \\textbf{Time ($\\mus$)} \\\\\n')
    for name in names:
        file.write(name+'&'+str(mean_time_dictionary[name])+'\\\\\n')