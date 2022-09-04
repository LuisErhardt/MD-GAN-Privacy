import pandas as pd
import numpy as np
import sys

### Loop the data lines
file_path = sys.argv[1]
# "CIFAR100/cifar100_old_single_defense/IGNORE_CLIENTS_5000ROWS_5CLIENT_5attacker_ROUND1.csv"
with open(file_path, 'r') as temp_f:
    # get No of columns in each line
    col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
column_names = [i for i in range(0, max(col_count))]
### Read csv
df = pd.read_csv(file_path, header=None, delimiter=",", names=column_names)

attacker_number = int(sys.argv[2])
attacker = [5,6,7,8,9]
benign = [0,1,2,3,4]
precision_list = []


count_total_ignore = 0
count_correct_ignore = 0
count_wrong_ignore = 0


_, column_width = df.shape 

for ind in df.index:
    round_precision_counter = 0
    for col in range(column_width):
        if df.loc[ind, col] in attacker:
            round_precision_counter += 1
            count_correct_ignore += 1
            count_total_ignore += 1
        elif df.loc[ind, col] in benign:
            count_wrong_ignore += 1
            count_total_ignore += 1
    precision_list.append(round_precision_counter/attacker_number)        

print("precision, recall: ",count_correct_ignore*100/count_total_ignore, np.mean(precision_list)*100)


