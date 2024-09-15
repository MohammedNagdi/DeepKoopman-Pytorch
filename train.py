import os
import logging
import argparse
import yaml
import torch.optim
import matplotlib.pyplot as plt
from model.network import KoopmanAutoencoder
from data_utils import load_dataset,differential_dataset
from loss_functions import koopman_loss,prediction_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_losses(train_losses, val_losses, prediction_losses=None, save_path=None):
    """
    Plot the training, validation, and optionally prediction losses over epochs on separate plots.
    
    Args:
        train_losses (list): List of training losses over epochs.
        val_losses (list): List of validation losses over epochs.
        prediction_losses (list, optional): List of prediction losses over epochs.
        save_path (str, optional): Path to save the plot as an image. If None, plot will be shown.
    """
    epochs = range(1, len(train_losses) + 1)  # Assuming one loss per epoch

    # Define the number of subplots (2 if prediction_losses is None, otherwise 3)
    num_plots = 2 if prediction_losses is None else 3
    

    plt.figure(figsize=(10, 15))
    
    # Plot 1: Training Loss
    plt.subplot(num_plots, 1, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue', marker='o', markersize=6, linewidth=2)
    plt.yscale('log')  # Set the y-axis to log scale for better visibility
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title('Training Loss (Log Scale)')
    plt.grid(True)
    
    # Plot 2: Validation Loss
    plt.subplot(num_plots, 1, 2)
    plt.plot(epochs, val_losses, label="Validation Loss", color='orange', marker='x', markersize=6, linewidth=2)
    plt.yscale('log')  # Set the y-axis to log scale for better visibility
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title('Validation Loss (Log Scale)')
    plt.grid(True)
    
    # Plot 3 (optional): Prediction Loss
    if prediction_losses is not None:
        plt.subplot(num_plots, 1, 3)
        plt.plot(epochs, prediction_losses, label="Prediction Loss", color='green', marker='s', markersize=6, linewidth=2)
        plt.yscale('log')  # Set the y-axis to log scale for better visibility
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.title('Prediction Loss (Log Scale)')
        plt.grid(True)

    # Adjust layout so the plots don't overlap
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
    else:
        plt.show()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Args:
            val_loss (float): The current validation loss.
        """
        # Initialize best_loss if it's the first validation
        if self.best_loss is None:
            self.best_loss = val_loss
        # If there is an improvement (greater than min_delta), reset counter
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        # If no improvement, increase the counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True



if __name__ == '__main__':
    # read the yaml file from input
    parser = argparse.ArgumentParser(description='Add the path to the config file')
    parser.add_argument('--config', type=str, help='path to the config file')
    args = parser.parse_args()
    config_file = args.config
    config = load_config(config_file)


    

    # create the experiment directory
    Experiment_name = config["Experiment_name"]
    Experiment_name = "experiments/" + Experiment_name
    if not os.path.exists(Experiment_name):
        os.makedirs(Experiment_name)


    # experiment
    training = config["training"]
    forecasting = config["forecasting"]
    architecture = config["architecture"]
    data = config["dataset"]

    # Dataset
    dataset = data["dataset_name"]
    sample_per_traj = data["sample_per_traj"]
    delta_t = data["delta_t"]

    # Network Dimensions
    koopman_dim = config["koopman_dim"]
    hidden_dim = config["hidden_dim"]
    input_dim = config["input_dim"]
    oper_hidden_dim = config["oper_hidden_dim"]
    

    # Training Parameters
    epochs = training["epochs"]
    lr = training["learning_rate"]
    save_every = training["save_every"]
    start_epoch = training["start_epoch"]
    batch_size = training["batch_size"]
    load_chkpt = training["load_checkpoint"]
    chkpt_filename = training["checkpoint_filename"]
    chkpt_filename = Experiment_name + "/" + chkpt_filename
    optimizer_type = training["optimizer"]
    alpha1 = training["alpha1"]
    alpha2 = float(training["alpha2"])

    # add logging 
    if os.path.exists(Experiment_name+"/training.log"):
        os.remove(Experiment_name+"/training.log")
    logging.basicConfig(filename=Experiment_name+"/training.log", level=logging.INFO)
    # Forecasting Parameters
    Sp = forecasting["sp"]
    horizon = forecasting["horizon"]
    T = max(horizon,Sp)

    # Architectural Parameters
    device = architecture["device"]
    arch = architecture["model_arch"]
    oper_arch = architecture["operator_arch"]
    oper_type = architecture["operator_type"]

    # The model
    if oper_type == "fixed":
        number_cj = config["fixed"]["number_of_complex_koopman"]
        number_real = config["fixed"]["number_of_real_koopman"]
        model = KoopmanAutoencoder(input_dim,koopman_dim,hidden_dim = hidden_dim,delta_t=delta_t,device=device,arch=arch,n_com=number_cj,n_real=number_real,oper_arch=oper_arch,oper_type=oper_type,oper_hidden_dim=oper_hidden_dim).to(device)

    else:
        number_cj = config["mixed"]["number_of_conjugate_pairs"]
        number_real = config["mixed"]["number_of_real_koopman"]
        if (number_cj * 2 + number_real) > koopman_dim:
            raise ValueError("Number of eigenvalues is greater than koopman dimension")
        model = KoopmanAutoencoder(input_dim,koopman_dim,hidden_dim = hidden_dim,delta_t=delta_t,device=device,arch=arch,n_com=number_cj,n_real=number_real,oper_arch=oper_arch,oper_type=oper_type,oper_hidden_dim=oper_hidden_dim).to(device)

    # The optimizer
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    
    # Load the checkpoint
    if load_chkpt:
        # make sure the model the loaded model is the same as the model you are using
        print("LOAD CHECKPOINTS")
        state_dicts = torch.load(chkpt_filename+".pth")
        model.load_state_dict(state_dicts['model'])
        optimizer.load_state_dict(state_dicts['optimizer'])
        start_epoch=state_dicts["start_epoch"]
        print(state_dicts["start_epoch"])
        print(state_dicts.keys())
    
    # Load the dataset
    X_train,X_test = load_dataset(dataset_name=dataset,chunk_size=1)
    X_train_recon = X_train
    X_test_recon = X_test[:,:-T,:]
    X_forecast_test = X_test[:,-T:,:]
    train_dl = DataLoader(differential_dataset(X_train_recon,T,samples=sample_per_traj),batch_size=batch_size)
    test_dl = DataLoader(differential_dataset(X_test_recon,T,samples=sample_per_traj),batch_size=batch_size)
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

    # Early Stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0)

    # training loop
    for epoch in range(start_epoch,epochs):
        train_epoch_loss = []
        test_epoch_loss = []

        model.train()
        for x in tqdm(train_dl):
            optimizer.zero_grad()
            loss_train = koopman_loss(x.to(device), model, Sp=Sp, T=T,alpha1=alpha1,alpha2=alpha2)
            loss_train.backward()
            optimizer.step()
            train_epoch_loss.append(loss_train.item()*x.shape[0])

        with torch.no_grad():
            model.eval()
            for x in tqdm(test_dl):
                loss_test = koopman_loss(x.to(device), model, Sp=Sp, T=T,alpha1=alpha1,alpha2=alpha2)
                test_epoch_loss.append(loss_test.cpu().item() * x.shape[0])

            forecast_loss = prediction_loss(X_test_recon[:,[-1],:].to(device), X_forecast_test.to(device),model)


        if (epoch+1) % save_every == 0:
            torch.save({"model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "start_epoch":(epoch+1)},chkpt_filename+".pth")


        train_loss = np.sum(train_epoch_loss)/len(train_dl.dataset)
        test_loss = np.sum(test_epoch_loss)/len(test_dl.dataset)
        print("\n","="*10,f" EPOCH {epoch} ","="*10)
        print("\nPrediction Loss: ",format(forecast_loss))
        # print("Reconstruction Loss: {:.4f}".format(reconstruction_loss))
        print("TRAIN LOSS: ",train_loss)
        print("TEST LOSS: ",test_loss)
        # add in the log file
        logging.info("\n")
        logging.info("="*10+f" EPOCH {epoch} "+"="*10)
        logging.info(f"\nPrediction Loss: {format(forecast_loss)}")
        logging.info(f"TRAIN LOSS: {format(train_loss)}")
        logging.info(f"TEST LOSS: {format(test_loss)}")
        prediction_losses.append(forecast_loss)
        train_losses.append(np.sum(train_epoch_loss)/len(train_dl.dataset))
        validation_losses.append(np.sum(test_epoch_loss)/len(test_dl.dataset))


        # Early Stopping
        early_stopping(np.sum(test_epoch_loss)/len(test_dl.dataset))
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        
    prediction_losses = torch.tensor(prediction_losses).to("cpu")
    train_losses = torch.tensor(train_losses).to("cpu")
    validation_losses = torch.tensor(validation_losses).to("cpu")
    np.save(Experiment_name+"/prediction_losses.npy",prediction_losses)
    np.save(Experiment_name+"/train_losses.npy",train_losses)
    np.save(Experiment_name+"/validation_losses.npy",validation_losses)



    # visualize the results
    plot_losses(train_losses, validation_losses, prediction_losses, save_path=f"{Experiment_name}/losses_plot.png")

    #* FIDDLE
    logging.info("COMPLETE NOW :)")
    print("COMPLETE NOW :)")
    
