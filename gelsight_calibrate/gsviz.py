import numpy as np
import open3d


class Visualize3D:
    def __init__(self, imgh, imgw, save_path=""):
        self.imgh, self.imgw = imgh, imgw
        self.init_open3D()
        self.cnt = 212
        self.save_path = save_path

    def init_open3D(self):
        x = np.arange(self.imgw)
        y = np.arange(self.imgh)
        self.X, self.Y = np.meshgrid(x, y)
        Z = np.sin(self.X)

        self.points = np.zeros([self.imgw * self.imgh, 3])
        self.points[:, 0] = np.ndarray.flatten(self.X)  # / self.imgh
        self.points[:, 1] = np.ndarray.flatten(self.Y)  # / self.imgw

        self.depth2points(Z)

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.imgw, self.imgh, 3]))
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)
        self.vis.add_geometry(self.pcd)

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)

    def update(self, Z, dx, dy):
        self.depth2points(Z)
        dx, dy = dx * 0.5, dy * 0.5

        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])
        for _ in range(3):
            colors[:, _] = np_colors

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        #### SAVE POINT CLOUD TO A FILE
        if self.save_path != "":
            open3d.io.write_point_cloud(
                self.save_path + "/pc_{}.pcd".format(self.cnt), self.pcd
            )

        self.cnt += 1

    def save_pointcloud(self):
        open3d.io.write_point_cloud(
            self.save_path + "pc_{}.pcd".format(self.cnt), self.pcd
        )

    def close(self):
        self.vis.close()


def plot_gradients(fig, ax, gx, gy, mask = None, n_skip=5, scale=10.0):
    """
    Plot the gradients of the surface on the image using quiver.
    If the mask is not none, only plot the gradients on the mask.
    """
    imgh, imgw = gx.shape
    X, Y = np.meshgrid(np.arange(imgw)[::n_skip], np.arange(imgh)[::n_skip])
    U = gx[::n_skip, ::n_skip] * scale
    V = -gy[::n_skip, ::n_skip] * scale
    if mask is None:
        mask = np.ones_like(gx)
    else:
        mask = np.copy(mask)
    mask = mask[::n_skip, ::n_skip]
    ax.quiver(X[mask], Y[mask], U[mask], V[mask], units="xy", scale=1, color="red")
