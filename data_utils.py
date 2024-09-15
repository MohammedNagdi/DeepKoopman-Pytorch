import pickle
import pandas as pd
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def load_data(df,chunk_size=1):
    X = []
    for i in tqdm(df.index.unique()):
        x = torch.FloatTensor(df.loc[i].values)
        size = x.shape[0]
        if chunk_size > 1:
            size = int(size/chunk_size)
        x = torch.chunk(x,chunk_size)
        X.extend(x)
    X = torch.stack(X, 0)
    return X

def load_dataset(dataset_name = "Repressilator",file_path=r'data',chunk_size=1):

    with open(os.path.join(file_path,f"{dataset_name}/{dataset_name}_train_inputs.pickle"), "rb") as handle:
        train_df = pd.read_pickle(handle)

    with open(os.path.join(file_path,f"{dataset_name}/{dataset_name}_val_inputs.pickle"), "rb") as handle:
        test_df = pd.read_pickle(handle)

    X_train = load_data(train_df,chunk_size)
    X_test = load_data(test_df,chunk_size)

    return X_train,X_test


class differential_dataset(Dataset):

    def __init__(self,X,horizon,samples=1):

        self.X = X # the entire dataset
        self.samples = samples # number of samples per time series
        self.horizon = horizon # the horizon of the forecast
        self.D = X.shape[-1] # the number of dimensions
        self.T = X.shape[1]-self.horizon # should be length - horizon - horizon 
        self.mu = torch.tensor([torch.mean(X[:,:,i]) for i in range(self.D)]) # mean of each dimension
        self.std = torch.tensor([torch.std(X[:,:,i]) for i in range(self.D)]) # std of each dimension
        self.max = torch.tensor([torch.max(X[:,:,i]) for i in range(self.D)]) # max of each dimension
        self.min = torch.tensor([torch.min(X[:,:,i]) for i in range(self.D)]) # min of each dimension

    def __len__(self):
        return self.X.shape[0] * self.samples # number of samples in this case is the number of time series

    def __getitem__(self,idx):
        slice = self.T/self.samples # the size of each slice
        start_slice_t = int((idx % self.samples) * slice) # the starting point of the slice
        end_slice_t = int(start_slice_t + slice) # the ending point of the slice

        if torch.is_tensor(idx):
            idx = idx/self.samples
            idx = idx.tolist()
        if type(idx) is int:
            idx = idx/self.samples
            idx = [idx]
        start = torch.randint(low=start_slice_t,high=end_slice_t,size=(len(idx),)) # generate a random starting point
        windows = torch.tensor([list(range(i,i+self.horizon)) for i in start]).unsqueeze(-1).repeat(1,1,self.D) # generate the window from the starting point to the horizon
        x = torch.gather(self.X[idx],1,windows).squeeze() # make it in a tensor of [1,horizon,D]

        return x

if __name__ == "__main__":
    X_train,X_test = load_dataset(chunk_size=1)
    print(X_train.shape)
    print(X_test.shape)

    dataset = differential_dataset(X_train,10)
    print("hello:")


