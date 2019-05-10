import progressbar as pb
import numpy as np
import os

from shbeamforming import *
from utilities import *


def batch_srp_map(dir, **kwargs):
# this function is for getting from batches of Ambisonic files

    file_list = os.listdir(dir)

    progbar = pb.ProgressBar(max_value=len(file_list))
    progbar.start()
    maps = {} # empty dict

    for i, filename in enumerate(file_list):
        progbar.update(i)

        # calculate power map for file
        filepath = 'SEL/noiseless_audio/ov1/test/' + filename
        pwr_tup = SRP_map(filepath, **kwargs)

        # add to dictionary of power maps
        name_start = filepath.rfind('/')+1
        maps[filepath[name_start:]] = pwr_tup

    progbar.finish()

    return maps
