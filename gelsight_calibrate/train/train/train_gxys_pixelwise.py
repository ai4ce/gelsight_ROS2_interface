import argparse
import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gelsight_calibrate.train.models import BGRXYDataset, BGRXYMLPNet_, BGRXYMLPNet
from gelsight_calibrate.utils import load_csv_as_dict, transfer_weights

"""
Train the gradients prediction network by loading each pixel at a time.
"""


def train_gxys_pixelwise(args):
    calib_dir = args.calib_dir
    device = args.device
    train_mode = args.train_mode
    # Create the model directory
    model_dir = osp.join(calib_dir, train_mode + "_gxy_model")
    if not osp.isdir(model_dir):
        os.makedirs(model_dir)
    # Load the train and test split
    data_path = osp.join(calib_dir, "train_test_split.json")
    with open(data_path, "r") as f:
        data = json.load(f)
        train_reldirs = data["train"]
        test_reldirs = data["test"]

    # Load the background data and get the subtracted bgrxy
    bg_path = os.path.join(calib_dir, "background_data.npz")
    bg_data = np.load(bg_path)
    subtract_bgrxys = np.zeros_like(bg_data["bgrxys"])
    subtract_bgrxys[:, :, :3] = bg_data["bgrxys"][:, :, :3].copy()
    # Prepare data
    train_data = {"all_bgrxys": [], "all_gxyangles": []}
    test_data = {"all_bgrxys": [], "all_gxyangles": []}
    for experiment_reldir in train_reldirs:
        data_path = osp.join(calib_dir, experiment_reldir, "data.npz")
        if not osp.isfile(data_path):
            print("Data file %s does not exist" % data_path)
        data = np.load(data_path)
        # Remove background or not
        if train_mode == "mlp":
            bgrxys = data["bgrxys"]
        elif train_mode == "mlp-nobg":
            bgrxys = data["bgrxys"] - subtract_bgrxys
        train_data["all_bgrxys"].append(bgrxys[data["mask"]])
        train_data["all_gxyangles"].append(data["gxyangles"][data["mask"]])
    for experiment_reldir in test_reldirs:
        data_path = osp.join(calib_dir, experiment_reldir, "data.npz")
        if not osp.isfile(data_path):
            print("Data file %s does not exist" % data_path)
        data = np.load(data_path)
        # Remove background or not
        if train_mode == "mlp":
            bgrxys = data["bgrxys"]
        elif train_mode == "mlp-nobg":
            bgrxys = data["bgrxys"] - subtract_bgrxys
        test_data["all_bgrxys"].append(bgrxys[data["mask"]])
        test_data["all_gxyangles"].append(data["gxyangles"][data["mask"]])

    # Split background data into train and test
    if train_mode == "mlp":
        bgrxys = bg_data["bgrxys"]
    elif train_mode == "mlp-nobg":
        bgrxys = bg_data["bgrxys"] - subtract_bgrxys
    bgrxys = bgrxys[bg_data["mask"]]
    gxyangles = bg_data["gxyangles"][bg_data["mask"]]
    perm = np.random.permutation(len(bgrxys))
    n_train = np.sum([len(bgrxys) for bgrxys in train_data["all_bgrxys"]]) // 5
    n_test = np.sum([len(bgrxys) for bgrxys in test_data["all_bgrxys"]]) // 5
    if n_train + n_test > len(bgrxys):
        n_train = 4 * len(bgrxys) // 5
        n_test = len(bgrxys) // 5
    train_data["all_bgrxys"].append(bgrxys[perm[:n_train]])
    train_data["all_gxyangles"].append(gxyangles[perm[:n_train]])
    test_data["all_bgrxys"].append(bgrxys[perm[n_train : n_train + n_test]])
    test_data["all_gxyangles"].append(gxyangles[perm[n_train : n_train + n_test]])

    # Create train and test Dataloader
    train_bgrxys = np.concatenate(train_data["all_bgrxys"])
    train_gxyangles = np.concatenate(train_data["all_gxyangles"])
    train_dataset = BGRXYDataset(train_bgrxys, train_gxyangles)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_bgrxys = np.concatenate(test_data["all_bgrxys"])
    test_gxyangles = np.concatenate(test_data["all_gxyangles"])
    test_dataset = BGRXYDataset(test_bgrxys, test_gxyangles)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Create the MLP Net for training
    net = BGRXYMLPNet_().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create MLP Net to eventually transfer weights to
    save_net = BGRXYMLPNet().to(device)

    # Initial evaluation
    train_mae = evaluate(net, train_dataloader, device)
    test_mae = evaluate(net, test_dataloader, device)
    naive_mae = np.mean(np.abs(test_gxyangles - np.mean(train_gxyangles, axis=0)))
    traj = {"train_maes": [train_mae], "test_maes": [test_mae], "naive_mae": naive_mae}
    print("Naive MAE (predict as mean): %.4f" % naive_mae)
    print("without train, Train MAE: %.4f, Test MAE: %.4f" % (train_mae, test_mae))

    # Train the model
    for epoch_idx in range(args.n_epochs):
        losses = []
        net.train()
        for bgrxys, gxyangles in train_dataloader:
            bgrxys = bgrxys.to(device)
            gxyangles = gxyangles.to(device)
            optimizer.zero_grad()
            outputs = net(bgrxys)
            loss = criterion(outputs, gxyangles)
            loss.backward()
            optimizer.step()

            # Record loss and size
            diffs = outputs - gxyangles
            losses.append(np.abs(diffs.cpu().detach().numpy()))
        net.eval()
        traj["train_maes"].append(np.mean(np.concatenate(losses)))
        traj["test_maes"].append(evaluate(net, test_dataloader, device))
        print(
            "Epoch %i, Train MAE: %.4f, Test MAE: %.4f"
            % (epoch_idx, traj["train_maes"][-1], traj["test_maes"][-1])
        )
        scheduler.step()

        # Save model every 10 steps
        if (epoch_idx + 1) % 10 == 0:
            # Transfer weights to MLP Net and save
            transfer_weights(net, save_net)
            save_path = os.path.join(model_dir, "nnmini.pth")
            torch.save(save_net.state_dict(), save_path)

    # Save the training curve
    save_path = os.path.join(model_dir, "training_curve.png")
    plt.plot(np.arange(len(traj["train_maes"])), traj["train_maes"], color="blue")
    plt.plot(np.arange(len(traj["test_maes"])), traj["test_maes"], color="red")
    plt.xlabel("Epochs")
    plt.ylabel("MAE (rad)")
    plt.title("MAE Curve")
    plt.savefig(save_path)
    plt.close()


def evaluate(net, dataloader, device):
    """Evaluate the network loss."""
    losses = []
    for bgrxys, gxyangles in dataloader:
        bgrxys = bgrxys.to(device)
        gxyangles = gxyangles.to(device)
        outputs = net(bgrxys)
        diffs = outputs - gxyangles
        losses.append(np.abs(diffs.cpu().detach().numpy()))
    mae = np.mean(np.concatenate(losses))
    return mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Normals.")
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        default="/home/zf540/tactile_recon_ws/src/gelsight_ROS2_interface/gelsight_calibrate/data",
        help="place where the calibration data is stored",
    )
    parser.add_argument(
        "-ne", "--n_epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument("-lr", "--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="the device to train NN",
    )
    parser.add_argument(
        "-tm",
        "--train_mode",
        type=str,
        choices=["mlp", "mlp-nobg"],
        default="mlp",
        help="The method to train the network",
    )
    args = parser.parse_args()
    # Train the gradients
    train_gxys_pixelwise(args)
