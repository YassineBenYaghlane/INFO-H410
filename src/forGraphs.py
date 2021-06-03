### ---------- EXPLANATION ------------- ###
# File to create the different graphs shown 
# in the report.

import matplotlib.pyplot as plt
import numpy as np
import csv

NUM_SAMPLES = 30 # Defines the number of files per algorithm (set in main.py)


# The "#%%" makes separate cells that can be ran individually

# %%
file_path = './out4/' # Directory with the csv files
names = ['random', 'sshaped', 'astar', 'inverse', 'weighted'] # The names of the files (without "_[number]")
colors = ['k', 'g', 'b', 'm', 'c'] # Some colors for the graphs


# Here we put all the values in a big dictionary. For example, big_dictionary['astar'][1] will
# be a 2-dim numpy array with the values of the corresponding file (astar_1.csv)
# The time_dictionary is of the form: time_dictionary['astar'] is a list of 2-list where each 
# 2-list is [max number of moves, time taken]

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

# Here we create dictionaries with the mean values. Note that the operation is a little bit 
# more complex than a simple "np.mean" because not all numpy arrays have the same sizes, 
# hence the use of masked arrays.

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
# Here we create the main graphs, showing the evolution of the score in function of the 
# number of moves.

plt.figure(figsize=(13, 11))
for name, color in zip(names, colors):

    for i in range(NUM_SAMPLES):
        plt.plot(big_dictionary[name][i][:, 0], big_dictionary[name][i][:, 1], color, linewidth=0.17)
    plt.plot(mean_dictionary[name][:, 0], mean_dictionary[name][:, 1], color, label=name, linewidth=1)
plt.legend(prop={'size': 20})
plt.xlabel('Number of moves', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Evolution of the score for \nthe algorithms and their survival edition', fontsize=24)

plt.savefig('thefigure2.png', bbox_inches='tight')
plt.show()

# %%
### SPEED
# Here the graphs for the "speed"

plt.figure(figsize=(13, 11))
for name, color in zip(names, colors):
    plt.plot(mean_speed_dictionary[name][:-2, 0], mean_speed_dictionary[name][:-2, 1], color, label=name, linewidth=1)
plt.legend(prop={'size': 20})
plt.xlabel('Number of moves', fontsize=14)
plt.ylabel('Speed', fontsize=14)
plt.title('Evolution of the speed for \nthe algorithms and their survival edition', fontsize=24)

plt.savefig('thespeed2.png', bbox_inches='tight')
plt.show()

# %%
### VARIANCE
# And here the graphs for the end mean and variance
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
plt.title('Mean and variance of the final score for \nthe algorithms and their survival edition', fontsize=24)

plt.savefig('thevar2.png', bbox_inches='tight')
plt.show()

# %%
### TIME
# Finally, here we store in a text file the computation times, already writing them
# in a latex format for the report.

with open('times2.txt', 'w') as file:
    file.write('\\underline{\\textbf{Av. time per move}} & \\textbf{Time ($\\mus$)} \\\\\n')
    for name in names:
        file.write(name+'&'+str(mean_time_dictionary[name])+'\\\\\n')
        
