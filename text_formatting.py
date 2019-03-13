import os
import numpy as np

def reformat_files(in_dir, out_dir):

    files = os.listdir(in_dir)

    # remove any hidden system files (.DS_Store etc.)
    _ = [files.remove(i) for i in files if i.startswith('.')]

    for file in files:
        data = np.genfromtxt(in_dir + file, delimiter=',')

        result_array = np.zeros((len(data)*2, len(data.T)))

        for i, entry in enumerate(data):
            i_array = np.array([i])
            line_1 = np.concatenate((i_array, entry[[0]], entry[2:]))
            line_2 = np.concatenate((i_array, entry[[1]], entry[2:]))

            result_array[2*i,:] = line_1
            result_array[(2*i)+1, :] = line_2

        result_array[:,[2,3]] = result_array[:,[3,2]]
        np.savetxt(out_dir + file, result_array, delimiter=',',
            fmt=['%d', '%1.3f', '%d', '%d'])


def strip_files(in_dir, out_dir):

    files = os.listdir(in_dir)

    # remove any hidden system files (.DS_Store etc.)
    _ = [files.remove(i) for i in files if i.startswith('.')]

    for file in files:
        a = open(in_dir + file).readlines()[1:]
        b = []

        for i in a:
            b.append(i[i.find(',')+1:i.rfind(',')]+'\n')

        file_to_write = open(out_dir + file[:file.find('.')] +'.csv', 'w')
        file_to_write.writelines(b)
        file_to_write.close()
