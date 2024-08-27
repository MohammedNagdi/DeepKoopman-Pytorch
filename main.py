# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch.optim
import os
from models import Lusch
from mixing_model import Lusch_mixing
from data_generator import load_dataset,differential_dataset
from loss_functions import koopman_loss,prediction_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # add experiment name to save the results
    Experiment_name = "fixed_matrix"
    # Network Dimensions
    koopman_dim = 16
    hidden_dim = 50
    input_dim = 6
    delta_t = 0.025
    # Training Parameters
    epochs = 200
    lr = 1e-3
    save_every = 5
    start_epoch = 1
    batch_size = 64
    load_chkpt = False  # True if you want to load a checkpoint
    chkpt_filename = "fixed_matrix_checkk" # filename to save the checkpoint

    # Forecasting Parameters
    # SP is the number of steps 
    # T is the maximum of SP and horizon
    Sp =72; horizon = 72; T = max(horizon,Sp)
    
    # Architectural Parameters
    device="mps" # The device to run the model on "cpu" or "cuda" or "mps"
    arch = "ekan" # The architecture of the model "mlp" or "fastkan" or "ekan"
    oper_arch = "ekan" # The architecture of the operator (auxilary) network "mlp" or "kan"
    type = "fixed"  # The type of the operator network "fixed" if all element are either real or complex conjugate pairs or "mixing" if there are both real and complex conjugate pairs in the operator network

    if type == "fixed":
        number_cj = 0  # number of koopman operators constructed from complex conjugate pairs
        number_real = 1 # number of koopman operators contructed from real eigenvalues
        model = Lusch(input_dim,koopman_dim,hidden_dim = hidden_dim,delta_t=delta_t,device=device,arch=arch,n_com=number_cj,n_real=number_real,oper_arch=oper_arch).to(device)
    else:
        number_cj = 6 # number of complex conjugate pairs
        if number_cj * 2 > koopman_dim:
            raise ValueError("Number of complex conjugate pairs is greater than koopman dimension")
        number_real = koopman_dim - number_cj * 2 # number of koopman operators contructed from real eigenvalues
        model = Lusch_mixing(input_dim,koopman_dim,hidden_dim = hidden_dim,delta_t=delta_t,device=device,arch=arch,n_com=number_cj,n_real=number_real,oper_arch=oper_arch).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    if load_chkpt:
        # make sure the model the loaded model is the same as the model you are using
        print("LOAD CHECKPOINTS")
        state_dicts = torch.load(chkpt_filename+".pth")
        model.load_state_dict(state_dicts['model'])
        optimizer.load_state_dict(state_dicts['optimizer'])
        start_epoch=state_dicts["start_epoch"]
        print(state_dicts["start_epoch"])
        print(state_dicts.keys())

    X_train,X_test = load_dataset(dataset_name="Repressilator",chunk_size=1)
    X_train_recon = X_train[:,:-T,:]; X_test_recon = X_test[:,:-T,:]
    X_forecast_train = X_train[:,-T:,:]; X_forecast_test = X_test[:,-T:,:]
    train_dl = DataLoader(differential_dataset(X_train_recon,T),batch_size=batch_size)
    test_dl = DataLoader(differential_dataset(X_test_recon,T),batch_size=batch_size)
    if arch == "mlp":
        model.mu = train_dl.dataset.mu.to(device)
        model.std = train_dl.dataset.std.to(device)
    else:
        model.mu = train_dl.dataset.min.to(device)
        std = train_dl.dataset.max - train_dl.dataset.min
        model.std = std.to(device)

    # losses
    prediction_losses = []
    train_losses = []
    validation_losses = []

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
        prediction_losses.append(forecast_loss)
        train_losses.append(np.sum(train_epoch_loss)/X_train.shape[0])
        validation_losses.append(np.sum(test_epoch_loss)/X_test.shape[0])

    # save the losses
    # create a folder with the exp name to save the results
    if not os.path.exists(Experiment_name):
        os.makedirs(Experiment_name)
    prediction_losses = torch.tensor(prediction_losses).to("cpu")
    train_losses = torch.tensor(train_losses).to("cpu")
    validation_losses = torch.tensor(validation_losses).to("cpu")
    np.save(Experiment_name+"/prediction_losses.npy",prediction_losses)
    np.save(Experiment_name+"/train_losses.npy",train_losses)
    np.save(Experiment_name+"/validation_losses.npy",validation_losses)

#* FIDDLE

print("COMPLETE NOW :)")
