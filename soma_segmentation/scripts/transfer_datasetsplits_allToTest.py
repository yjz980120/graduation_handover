import pickle
import os


def transfer(pkl_dir,output_dir):
    pkl_file = os.path.join(pkl_dir,"data_splits.pkl")
    bf = open(pkl_file,"rb")
    pkl = pickle.load(bf)
    pkl["test"].extend(pkl["train"])
    pkl["test"].extend(pkl["val"])
    output_file = os.path.join(output_dir,"data_splits_allTest.pkl")
    with open(output_file,"wb") as wf:
        pickle.dump(pkl,wf)

if __name__ == "__main__":
    pkl_dir = "/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/data/task003_withNoRandomCrop_withTiffImageForRefer"
    output_dir = "/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/data/task003_withNoRandomCrop_withTiffImageForRefer"
    transfer(pkl_dir,output_dir)
