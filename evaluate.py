import os
import argparse
import yaml
import torch.optim
import logging
import pickle
import matplotlib.pyplot as plt
from model.network import KoopmanAutoencoder
import torch.nn.functional as F
from data_utils import load_dataset,differential_dataset,load_data
from loss_functions import koopman_loss,prediction_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd



def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    load_chkpt = True
    chkpt_filename = training["checkpoint_filename"]
    chkpt_filename = Experiment_name + "/" + chkpt_filename
    optimizer_type = training["optimizer"]
    alpha1 = training["alpha1"]
    alpha2 = float(training["alpha2"])

    # add logging 
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



    # make sure the model the loaded model is the same as the model you are using
    print("LOAD CHECKPOINTS")
    state_dicts = torch.load(chkpt_filename+".pth")
    model.load_state_dict(state_dicts['model'])

    # load the test data
    if "test_data_path" in config["training"] and config["training"]["test_data_path"] is not None:
        testing_data_path = config["training"]["test_data_path"]
    else:
        testing_data_path = os.path.join("data",f"{dataset}/{dataset}_test_inputs.pickle")
    with open(testing_data_path, "rb") as handle:
        test_df = pd.read_pickle(handle)
    X_test = load_data(test_df,chunk_size=1)

    X_test_recon = X_test[:, :-horizon, :]
    X_forecast_test = X_test[:, -horizon:, :]

    test_dl = DataLoader(differential_dataset(X_test_recon, horizon), batch_size=batch_size)


    # calculate the overall Koopman loss
    with torch.inference_mode():
        model.eval()
        test_loss = []

        for x in tqdm(test_dl):
            loss = koopman_loss(x.to(device), model, Sp=Sp, T=T,alpha1=alpha1,alpha2=alpha2)
            test_loss.append(loss)
        test_loss = np.array([loss.cpu().item() for loss in test_loss])
        Test_Loss =  torch.tensor(np.sum(test_loss)/len(test_dl))
        print(f" Loss: {Test_Loss}")
       
        logging.info("====================================")
        logging.info("Evaluating the model")
        logging.info(f" The overall loss for the testing data is: {Test_Loss}")
    
        # calculate the prediction loss
        forecast_loss = []
        for i in range(X_forecast_test.shape[0]):
            Y = model.koopman_operator(model.embed(X_test_recon[:,[-1],:].to(device)),X_forecast_test.to(device).shape[1])
            f_loss = F.mse_loss(X_forecast_test.to(device),model.recover(Y))
            forecast_loss.append(f_loss.cpu().item())
        forecast_loss = np.mean(forecast_loss)
        print(f"Forecast Loss: {forecast_loss}")
        logging.info(f" The forecast loss for the testing data is: {forecast_loss}")




        # visulize the results
        n = 10
        # the reconstucted data
        x_recon_hat = model(X_test[[n],:-T,:].to(device)).cpu().squeeze(0)
        # the forecasted data
        x_ahead_hat = model.recover(model.koopman_operator(model.embed(X_test[[n],-T,:].to(device).unsqueeze(0)),horizon)).cpu().squeeze(0)

        num_dim = X_test.shape[2]  # Number of dimensions
        T_recon = x_recon_hat.shape[0]  # Length of reconstructed data
        T_forecast = x_ahead_hat.shape[0]  # Length of forecasted data

        for dim in range(num_dim):
            plt.figure(figsize=(10, 6))

            # Plot the original data for this dimension (Reconstructed Part)
            plt.plot(range(T_recon), X_test[n, :-T, dim].cpu().numpy(), label='Original (Reconstructed Part)', color='red', linestyle=':')
            
            # Plot the original data for this dimension (Forecast Part) with 'x' marker
            plt.plot(range(T_recon, T_recon + T_forecast), X_test[n, -T:, dim].cpu().numpy(), label='Original (Forecast Part)', color='red', linestyle='None', marker='x')

            # Plot the generated reconstructed data for this dimension in the same color (green, dotted line)
            plt.plot(range(T_recon), x_recon_hat[:, dim], label='Generated (Reconstructed)', color='green', linestyle=':')

            # Plot the generated forecasted data for this dimension in the same color (green, dotted line)
            plt.plot(range(T_recon, T_recon + T_forecast), x_ahead_hat[:, dim], label='Generated (Forecasted)', color='green', linestyle=':')

            # Add labels, title, and legend
            plt.xlabel('Time')
            plt.ylabel(f'Dimension {dim + 1}')
            plt.title(f'Dimension {dim + 1} - Reconstructed vs Forecasted')
            plt.legend()

            # Save the figure with a unique name
            plt.savefig(os.path.join(Experiment_name, f'dimension_{dim + 1}_reconstruction_vs_forecast.png'))

            # Close the figure to avoid overlap in subsequent plots
            plt.close()


