import cv2
import csv
import math
import numpy as np
from scipy.interpolate import griddata
from scipy import fftpack


def load_csv_as_dict(csv_path):
    """Load the csv file entries as dictionaries."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        data = list(reader)
        keys = reader.fieldnames
        data_dict = {}
        for key in keys:
            data_dict[key] = []
        for line in data:
            for key in keys:
                data_dict[key].append(line[key])
    return data_dict

def image2bgrxys(image):
    """Convert a bgr image to bgrxy."""
    xys = np.dstack(
        np.meshgrid(
            np.arange(image.shape[1]), np.arange(image.shape[0]), indexing="xy"
        )
    )
    xys = xys.astype(np.float32) / np.array([image.shape[1], image.shape[0]])
    bgrs = image.copy() / 255
    bgrxys = np.concatenate([bgrs, xys], axis=2)
    return bgrxys

def depths2points(depths, imgh, imgw, mmpp=1.0):
    """Convert depth images to point clouds."""
    x = np.arange(imgw)
    y = np.arange(imgh)
    X, Y = np.meshgrid(x, y)
    Z = depths
    points = np.zeros([imgw * imgh, 3])
    points[:, 0] = np.ndarray.flatten(X) * mmpp
    points[:, 1] = np.ndarray.flatten(Y) * mmpp
    points[:, 2] = np.ndarray.flatten(Z) * mmpp
    return points


def transfer_weights(mlp_model, fcn_model):
    """transfer weights between BGRXYMLPNet_ to BGRXYMLPNet"""
    # Copy weights from fc1 to conv1
    fcn_model.conv1.weight.data = mlp_model.fc1.weight.data.view(
        fcn_model.conv1.weight.size()
    )
    fcn_model.conv1.bias.data = mlp_model.fc1.bias.data
    fcn_model.bn1.weight.data = mlp_model.bn1.weight.data
    fcn_model.bn1.bias.data = mlp_model.bn1.bias.data
    fcn_model.bn1.running_mean = mlp_model.bn1.running_mean
    fcn_model.bn1.running_var = mlp_model.bn1.running_var
    # Copy weights from fc2 to conv2
    fcn_model.conv2.weight.data = mlp_model.fc2.weight.data.view(
        fcn_model.conv2.weight.size()
    )
    fcn_model.conv2.bias.data = mlp_model.fc2.bias.data
    fcn_model.bn2.weight.data = mlp_model.bn2.weight.data
    fcn_model.bn2.bias.data = mlp_model.bn2.bias.data
    fcn_model.bn2.running_mean = mlp_model.bn2.running_mean
    fcn_model.bn2.running_var = mlp_model.bn2.running_var
    # Copy weights from fc3 to conv3
    fcn_model.conv3.weight.data = mlp_model.fc3.weight.data.view(
        fcn_model.conv3.weight.size()
    )
    fcn_model.conv3.bias.data = mlp_model.fc3.bias.data
    fcn_model.bn3.weight.data = mlp_model.bn3.weight.data
    fcn_model.bn3.bias.data = mlp_model.bn3.bias.data
    fcn_model.bn3.running_mean = mlp_model.bn3.running_mean
    fcn_model.bn3.running_var = mlp_model.bn3.running_var
    # Copy weights from fc4 to conv4
    fcn_model.conv4.weight.data = mlp_model.fc4.weight.data.view(
        fcn_model.conv4.weight.size()
    )
    fcn_model.conv4.bias.data = mlp_model.fc4.bias.data
    return fcn_model


def poisson_dct_neumaan(gx, gy):
    """2D integration using Poisson solver."""
    gxx = 1 * (
        gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])]
        - gx[:, ([0] + list(range(gx.shape[1] - 1)))]
    )
    gyy = 1 * (
        gy[(list(range(1, gx.shape[0])) + [gx.shape[0] - 1]), :]
        - gy[([0] + list(range(gx.shape[0] - 1))), :]
    )

    f = gxx + gyy

    ### Right hand side of the boundary condition
    b = np.zeros(gx.shape)
    b[0, 1:-2] = -gy[0, 1:-2]
    b[-1, 1:-2] = gy[-1, 1:-2]
    b[1:-2, 0] = -gx[1:-2, 0]
    b[1:-2, -1] = gx[1:-2, -1]
    b[0, 0] = (1 / np.sqrt(2)) * (-gy[0, 0] - gx[0, 0])
    b[0, -1] = (1 / np.sqrt(2)) * (-gy[0, -1] + gx[0, -1])
    b[-1, -1] = (1 / np.sqrt(2)) * (gy[-1, -1] + gx[-1, -1])
    b[-1, 0] = (1 / np.sqrt(2)) * (gy[-1, 0] - gx[-1, 0])

    ## Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    f[0, 1:-2] = f[0, 1:-2] - b[0, 1:-2]
    f[-1, 1:-2] = f[-1, 1:-2] - b[-1, 1:-2]
    f[1:-2, 0] = f[1:-2, 0] - b[1:-2, 0]
    f[1:-2, -1] = f[1:-2, -1] - b[1:-2, -1]

    ## Modification near the corners (Eq. 54 in [1])
    f[0, -1] = f[0, -1] - np.sqrt(2) * b[0, -1]
    f[-1, -1] = f[-1, -1] - np.sqrt(2) * b[-1, -1]
    f[-1, 0] = f[-1, 0] - np.sqrt(2) * b[-1, 0]
    f[0, 0] = f[0, 0] - np.sqrt(2) * b[0, 0]

    ## Cosine transform of f
    tt = fftpack.dct(f, norm="ortho")
    fcos = fftpack.dct(tt.T, norm="ortho").T

    # Cosine transform of z (Eq. 55 in [1])
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = 4 * (
        (np.sin(0.5 * math.pi * x / (f.shape[1]))) ** 2
        + (np.sin(0.5 * math.pi * y / (f.shape[0]))) ** 2
    )

    # 4 * ((sin(0.5 * pi * x / size(p, 2))). ^ 2 + (sin(0.5 * pi * y / size(p, 1))). ^ 2)

    f = -fcos / denom
    # Inverse Discrete cosine Transform
    tt = fftpack.idct(f, norm="ortho")
    img_tt = fftpack.idct(tt.T, norm="ortho").T

    img_tt = img_tt.mean() + img_tt
    # img_tt = img_tt - img_tt.min()

    return img_tt


def find_marker(gray):
    mask = cv2.inRange(gray, 0, 70)
    # kernel = np.ones((2, 2), np.uint8)
    # dilation = cv2.dilate(mask, kernel, iterations=1)
    return mask


def dilate(img, ksize=5, iter=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iter)


def matching_rows(A, B):
    ### https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    matches = [i for i in range(B.shape[0]) if np.any(np.all(A == B[i], axis=1))]
    if len(matches) == 0:
        return B[matches]
    return np.unique(B[matches], axis=0)


def interpolate_grad(img, mask):
    # pixel around markers
    mask_around = (dilate(mask, ksize=3, iter=2) > 0) & ~(mask != 0)
    mask_around = mask_around.astype(np.uint8)

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    yy, xx = np.meshgrid(y, x)

    mask_zero = mask_around == 1
    mask_x = xx[mask_around == 1]
    mask_y = yy[mask_around == 1]
    points = np.vstack([mask_x, mask_y]).T
    values = img[mask_x, mask_y]
    markers_points = np.vstack([xx[mask != 0], yy[mask != 0]]).T
    method = "nearest"
    # method = "linear"
    # method = "cubic"
    x_interp = griddata(points, values, markers_points, method=method)
    x_interp[x_interp != x_interp] = 0.0
    ret = img.copy()
    ret[mask != 0] = x_interp
    return ret


def demark(gx, gy, markermask):
    gx_interp = interpolate_grad(gx.copy(), markermask)
    gy_interp = interpolate_grad(gy.copy(), markermask)
    return gx_interp, gy_interp
