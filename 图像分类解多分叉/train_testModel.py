import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import json

from neuronet.models import encoder

def train(model, loss, optimizer, num_epochs=200):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = sitk.GetArrayFromImage(sitk.ReadImage("/home/yjz/Projects/Auto_tracing/neuronet_new_0519_crossing_sixPathsModality/neuronet/datasets/train_input/236174_8754_13036_1936_196.66_60.48_89.21_195.73_56.35_87.06inp.tiff"))
    y = np.array([0,0,0,1,0,1])
    #import ipdb; ipdb.set_trace()
    
    inps = torch.from_numpy(X)
    labs = torch.from_numpy(y)
    inps = inps.to(device)
    labs = labs.to(device)
    inps = inps.unsqueeze(0)
    inps = inps.float()
    labs = labs.float()
    labs = labs.view(1,-1)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)
        train_loss = 0
        val_loss = 0
        optimizer.zero_grad()
        logits = model(inps)
        m = nn.Sigmoid()
        outputs = m(logits)
        l  = loss(outputs, labs)
        l.backward()
        optimizer.step()
        train_loss += l.item()

        print("epoch %d loss:%0.4f" % (epoch,train_loss))

def main():
    net_config = "./models/configs/default_config.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(net_config) as fp:
        net_configs = json.load(fp)
        print("Network configs: ",net_configs)
        model = encoder.UNet(**net_configs)
        print("\n" + "="*10 + "Network Structure" + "="*10)
        print(model)
        print("="*30 + "\n")

    model = model.to(device)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=3e-5)
    train(model, loss, optimizer)

if __name__ == "__main__":
    main()
