import time
import os 


def timestr():
    t0 = time.time() + 60 * 60 * 2
    return time.strftime("%Y%m%d-%I_%M_%p",time.localtime(t0))

def get_fname(file_path):
    full_name = os.path.basename(file_path)
    file_name = os.path.splitext(full_name)
    return file_name[0]