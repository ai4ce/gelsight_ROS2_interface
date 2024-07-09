import torch
import numpy as np
import os
import os.path as osp
import cv2
from .utils import poisson_dct_neumaan, image2bgrxys
from .models import BGRXYMLPNet


class Reconstruction3D:
    """This class is used to provide surface information based on GelSight images."""

    def __init__(
        self,
        resource_dir,
        gxy_mode="mlp",
        contact_mode="naive",
        height_mode="naive",
        device="cuda",
    ):
        """
        :param resource_dir: str; the directory that holds the resources, including the background image and the gxy model.
        :param gxy_mode: str; the mode to get the gradients.
        :param contact_mode: str; the mode to get the contact mask.
        :param height_mode: str; the mode to get the height map.
        :param device: str; the device to run the model.
        """
        self.resource_dir = resource_dir
        self.gxy_mode = gxy_mode
        self.contact_mode = contact_mode
        self.height_mode = height_mode
        self.device = torch.device(device)
        # Load the gxy model
        gxy_model_path = osp.join(resource_dir, gxy_mode + "_gxy_model.pth")
        if not os.path.isfile(gxy_model_path):
            raise ValueError("Error opening " + gxy_model_path + " does not exist")
        self.gxy_net = BGRXYMLPNet()
        self.gxy_net.load_state_dict(torch.load(gxy_model_path))
        self.gxy_net.to(self.device)
        self.gxy_net.eval()

        bg_image = cv2.imread(osp.join(resource_dir, "background.png"))
        self.load_bg(bg_image)

    def load_bg(self, bg_image):
        """
        Load the background image.
        :param bg_image: np.array (H, W, 3); the background image.
        """
        self.bg_image = bg_image
        if self.gxy_mode == "mlp" or self.gxy_mode == "unet":
            bgrxys = image2bgrxys(bg_image)
            bgrxys = bgrxys.transpose(2, 0, 1)
            features = (
                torch.from_numpy(bgrxys[np.newaxis, :, :, :]).float().to(self.device)
            )
            with torch.no_grad():
                gxyangles = self.gxy_net(features)
                gxyangles = gxyangles[0].cpu().detach().numpy()
                self.bg_G = np.tan(gxyangles.transpose(1, 2, 0))
        elif self.gxy_mode == "mlp-nobg" or self.gxy_mode == "unet-nobg":
            self.subtract_bgrxys = image2bgrxys(bg_image)
            self.subtract_bgrxys[:, :, 3:] = 0.0

    def get_surface_info(self, image, mmpp, mask_markers=False, cm=None):
        """
        Get the surface information including height map (H), gradients (G), and contact mask (C).
        :param image: np.array (H, W, 3); the gelsight image.
        :param mmpp: float; the pixel per mm.
        :return G: np.array (H, W, 2); the gradients.
                H: np.array (H, W); the height map.
                C: np.array (H, W); the contact mask.
        """
        # Calculate the gradients
        bgrxys = image2bgrxys(image)
        if self.gxy_mode == "mlp-nobg" or self.gxy_mode == "unet-nobg":
            bgrxys = bgrxys - self.subtract_bgrxys
        bgrxys = bgrxys.transpose(2, 0, 1)
        features = (
            torch.from_numpy(bgrxys[np.newaxis, :, :, :]).float().to(self.device)
        )
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles[0].cpu().detach().numpy()
            G = np.tan(gxyangles.transpose(1, 2, 0))
        if self.gxy_mode == "mlp" or self.gxy_mode == "unet":
            G = G - self.bg_G
        # Calculate the height map
        if self.height_mode == "naive":
            H = poisson_dct_neumaan(G[:, :, 0], G[:, :, 1])
        # Calculate the contact mask
        if self.contact_mode == "naive":
            # Find the contact mask based on color difference
            diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > 15
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
            )
            color_mask = cv2.erode(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )
            # Filter by height
            cutoff = np.percentile(H, 85) - 0.2 / mmpp
            height_mask = H < cutoff
            C = np.logical_and(color_mask, height_mask)
        elif self.contact_mode == "naive_flat":
            # Find the contact mask based on color difference
            diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > 10
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )
            C = cv2.erode(
                color_mask.astype(np.uint8), np.ones((25, 25), np.uint8)
            ).astype(np.bool_)
        return G, H, C

        # # cm is contact_mask
        # if cm is None:
        #     cm = np.ones(frame.shape[:2])
        # if mask_markers:
        #     """find marker mask"""
        #     markermask = find_marker(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        #     cm = ~markermask
        #     """intersection of cm and markermask """
        #     # cmmm = np.zeros(img.shape[:2])
        #     # ind1 = np.vstack(np.where(cm)).T
        #     # ind2 = np.vstack(np.where(markermask)).T
        #     # ind2not = np.vstack(np.where(~markermask)).T
        #     # ind3 = matching_rows(ind1, ind2)
        #     # cmmm[(ind3[:, 0], ind3[:, 1])] = 1.
        #     # cmandmm = (np.logical_and(cm, markermask)).astype("uint8")
        #     # cmandnotmm = (np.logical_and(cm, ~markermask)).astype("uint8")
        # if mask_markers:
        #     dilated_mm = dilate(markermask, ksize=3, iter=2)
        #     gxs_interp, gys_interp = demark(gxs, gys, dilated_mm)
        # else:
        #     gxs_interp, gys_interp = gxs, gys
        # # normalize gradients for plotting purpose
        # print(gx.min(), gx.max(), gy.min(), gy.max())
        # gx = (gx - gx.min()) / (gx.max() - gx.min())
        # gy = (gy - gy.min()) / (gy.max() - gy.min())
        # gx_interp = (gx_interp - gx_interp.min()) / (gx_interp.max() - gx_interp.min())
        # gy_interp = (gy_interp - gy_interp.min()) / (gy_interp.max() - gy_interp.min())
