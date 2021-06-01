import multiprocessing
import os


def runMP(core_name, core_id):
    file_path = './compare/'
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

    elif core_id == 5:
        command = 'python ./src/snakeAI.py -t -w -z -o '
        filename_base = file_path+'survival_'


    else:
        print('Wrong Core ID. Aborting')
        return -1

    for count in range(30):
        filename = filename_base+str(count)+'.csv'
        print(core_name+'running on '+filename+', count: '+str(count))
        os.system(command+filename)


if __name__ == '__main__':

    processes = []
    for i in range(1, 6):
        p = multiprocessing.Process(target=runMP, args=("Core-%i"%i, i, ))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

