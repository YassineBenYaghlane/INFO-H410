### ---------------- DISCLAIMER ----------------- ###
# Make sure to have more than 4 logical cores before
# running  this code, otherwise (if 4 only), comment 
# the lines marked with "#*****", and edit lines
# marked with "#-----".

### ---------------- EXPLANATIONS --------------- ###
# This file simply launches the snakeAI.py in training
# mode with different flags, and writes the results
# in files with adequate names, and this is done in 
# parallel with the multiprocessing package.

import multiprocessing
import os


def runMP(core_name, core_id):
    file_path = './survival/' # Make sure to first create this directory
    if core_id == 1:
        command = 'python ./src/snakeAI.py -t -s -o '
        filename_base = file_path+'sshaped_'

    elif core_id == 2:
        command = 'python ./src/snakeAI.py -t -a -o '
        filename_base = file_path+'astar_'

    elif core_id == 3:
        command = 'python ./src/snakeAI.py -t -w -o '
        filename_base = file_path+'weighted_'

    elif core_id == 4:
        command = 'python ./src/snakeAI.py -t -n -o '
        filename_base = file_path+'inverse_'

    elif core_id == 5: #*****
        command = 'python ./src/snakeAI.py -t -r -o ' #*****
        filename_base = file_path+'random_' #*****


    else:
        print('Wrong Core ID. Aborting')
        return -1

    # Each algorithm is ran 30 times
    for count in range(30): 
        filename = filename_base+str(count)+'.csv'
        print(core_name+'running on '+filename+', count: '+str(count))
        os.system(command+filename)


if __name__ == '__main__':

    processes = []
    for i in range(1, 6): #-----: change "6" in "5" if only 4 cores
        p = multiprocessing.Process(target=runMP, args=("Core-%i"%i, i, ))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

