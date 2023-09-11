from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import csv

def array_of_epochs(interval, multiple):
    if multiple <= 0 or interval <= 0:
        return []

    result = []
    epoch = interval - 1 # epochs start at 0

    for i in range(multiple):
        result.append(str(epoch))
        epoch += interval

    return result

def main():
    dirs = []
    for epoch in array_of_epochs(20, 15):
        dir = 'data/cifar10-epoch{}'.format(epoch)
        dirs.append(dir)

    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)
    path_of_test_images = "data/imgs"

    result_fids = []
    for dir in dirs:
        fid = calculate_fid_given_paths([path_of_test_images, dir ], 500, 'cuda:0', 2048, num_workers)
        result_fids.append([dir, fid])

    print(result_fids)

    csv_filename = "result_fids.csv"

    # Open the CSV file in write mode
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # writer.writerow(["Dir", "FID"])  # Uncomment this line if you want headers

        for row in result_fids:
            writer.writerow(row)

if __name__ == "__main__":
    main()