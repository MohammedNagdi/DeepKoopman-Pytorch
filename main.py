# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch.optim

from models import Lusch
from models2 import Lusch_mixing
from data_generator import load_dataset,differential_dataset
from loss_functions import koopman_loss,prediction_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    koopman_dim = 64
    # update hidden dimesnion to 500 for MLP and 20-50 for KAN network
    hidden_dim = 500
    input_dim = 3
    delta_t = 0.01
    epochs = 300
    lr = 1e-3
    Sp = 72; horizon = 72; T = max(horizon,Sp)
    batch_size = 256
    load_chkpt = False
    chkpt_filename = "fixed_matrix_checkk"
    save_every = 5
    start_epoch = 1
    device="mps"
    arch = "mlp"
    type = "mixed"

    if type == "fixed":
        number_cj = 0  # number of koopman operators constructed from complex conjugate pairs
        number_real = 2 # number of koopman operators contructed from real eigenvalues
        model = Lusch(input_dim,koopman_dim,hidden_dim = hidden_dim,delta_t=delta_t,device=device,arch=arch,n_com=number_cj,n_real=number_real).to(device)
    else:
        number_cj = 16 # number of complex conjugate pairs
        if number_cj * 2 > koopman_dim:
            raise ValueError("Number of complex conjugate pairs is greater than koopman dimension")
        number_real = koopman_dim - number_cj * 2 # number of koopman operators contructed from real eigenvalues
        model = Lusch_mixing(input_dim,koopman_dim,hidden_dim = hidden_dim,delta_t=delta_t,device=device,arch=arch,n_com=number_cj,n_real=number_real).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    if load_chkpt:
        print("LOAD CHECKPOINTS")
        state_dicts = torch.load(chkpt_filename+".pth")
        model.load_state_dict(state_dicts['model'])
        optimizer.load_state_dict(state_dicts['optimizer'])
        start_epoch=state_dicts["start_epoch"]
        print(state_dicts["start_epoch"])
        print(state_dicts.keys())

    X_train,X_test = load_dataset(chunk_size=1)
    X_train_recon = X_train[:,:-T,:]; X_test_recon = X_test[:,:-T,:]
    X_forecast_train = X_train[:,-T:,:]; X_forecast_test = X_test[:,-T:,:]
    train_dl = DataLoader(differential_dataset(X_train_recon,T),batch_size=batch_size)
    test_dl = DataLoader(differential_dataset(X_test_recon,T),batch_size=batch_size)

    model.mu = train_dl.dataset.mu.to(device)
    model.std = train_dl.dataset.std.to(device)

    for epoch in range(start_epoch+1,epochs):
        train_epoch_loss = []
        test_epoch_loss = []

        model.train()
        for x in tqdm(train_dl):
            optimizer.zero_grad()
            loss_train = koopman_loss(x.to(device), model, Sp=Sp, T=T,alpha1=2,alpha2=1e-8)
            loss_train.backward()
            optimizer.step()
            train_epoch_loss.append(loss_train.item()*x.shape[0])

        with torch.no_grad():
            model.eval()
            for x in tqdm(test_dl):
                loss_test = koopman_loss(x.to(device), model, Sp=Sp, T=T,alpha1=2,alpha2=1e-8)
                test_epoch_loss.append(loss_test.cpu().item() * x.shape[0])

            forecast_loss = prediction_loss(X_test_recon[:,[-1],:].to(device), X_forecast_test.to(device),model)

        if (epoch+1) % save_every == 0:
            torch.save({"model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "start_epoch":(epoch+1)},chkpt_filename+".pth")


        print("\n","="*10,f" EPOCH {epoch} ","="*10)
        print("\nPrediction Loss: {:.4f}".format(forecast_loss))
        # print("Reconstruction Loss: {:.4f}".format(reconstruction_loss))
        print("TRAIN LOSS: ",np.sum(train_epoch_loss)/X_train.shape[0])
        print("TEST LOSS: ",np.sum(test_epoch_loss)/X_test.shape[0])

#* FIDDLE

print("COMPLETE NOW :)")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
