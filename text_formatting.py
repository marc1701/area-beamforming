import os
import numpy as np

def strip_files(in_dir, out_dir):

    files = os.listdir(in_dir)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

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
