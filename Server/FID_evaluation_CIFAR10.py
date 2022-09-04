
import numpy as np
from fid_score import calculate_fid_given_paths
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

check_points = np.arange(4, 100, 5)
folder_name = ['data/cifar10-epoch{}'.format(j) for j in check_points]

num_avail_cpus = len(os.sched_getaffinity(0))
num_workers = min(num_avail_cpus, 8)
# result_list = []
# for dir in folder_name:
result_list = calculate_fid_given_paths(['data/cifar10/', folder_name],
                                        500,
                                        'cuda:0',
                                        2048,
                                        num_workers)
# result_list.append(fid_value)
# print(fid_value)


print(result_list)


# Open Files
resultFyle = open("FID_CIFAR10_1BENIGN_1ATTACKER_ROUND1.csv",'w')

# Write data to file
for r in result_list:
    resultFyle.write(str(r) + "\n")
resultFyle.close()
