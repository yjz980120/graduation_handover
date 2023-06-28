import numpy as np
import os
import glob
import pickle

def read_labels(fname):
    labels = os.path.splitext(os.path.split(fname)[-1])[0].split("_")[0]
    labels_list = []
    for l in labels:
        labels_list.append(int(l))
    return np.array(labels_list)


def dataset_split(data_dir, valid_ratio=0.1, test_ratio=0.1, seed=1024):
    
    #import ipdb; ipdb.set_trace()
    samples = []

    for i,train_file in enumerate(glob.glob(os.path.join(data_dir,'train_input', "*tiff"))):
        label = read_labels(train_file)
        samples.append((train_file,label))

    # data splitting
    #import ipdb; ipdb.set_trace()
    num_tot = len(samples)
    print(f"Number of samples of total for test are {num_tot}")

    splits = {}
    splits["test"] = samples[:]

    # write to file
    with open(os.path.join(data_dir, "train_input", "data_splits.pkl"),'wb') as fp:
        pickle.dump(splits, fp)

if __name__ == "__main__":
    #label_dir = "/home/yjz/Projects/Auto_tracing/neuronet_new_0519_crossing_sixPathsModality/neuronet/data/task003_moreData/label/label.txt"
    input_dir = "/home/yjz/Project/Auto-tracing/neuronet/neuronet_new_0519_crossing_threeUniteModality/neuronet/data/task001_onlyForTest"
    #labels = read_ndarray_labels(label_dir)
    #import ipdb; ipdb.set_trace()
    dataset_split(input_dir)
