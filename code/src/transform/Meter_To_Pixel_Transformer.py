import numpy as np
import cv2
import torch

class MeterToPixelInterface():
    def __init__(self, homography_file_path) -> None:
        self.homography_file_path = homography_file_path

    def get_pixel_positions(self, positions: np.ndarray) -> np.ndarray:
        """Transforms the given positions to the pixel positions"""
        raise NotImplementedError('This method is abstract and should be overridden')

    def get_positions(self, pixel_positions: np.ndarray):
        raise NotImplementedError('This method is abstract and should be overridden')

    def change_x_and_y(self, pixel_positions):
        """Changes the x and y axis independent of the input shape"""
        orig_shape = pixel_positions.shape
        pixel_positions = pixel_positions.reshape(-1, 2)
        pixel_positions = pixel_positions[:, [1, 0]]
        return pixel_positions.reshape(orig_shape)

    def apply_homography_matrix(self, positions: np.ndarray, homography_matrix: np.ndarray):
        """Transforms the given positions to the pixel positions along the homography matrix"""
        pixel_positions = cv2.perspectiveTransform(positions.reshape(1, -1, 2).astype(float), homography_matrix)
        pixel_positions = np.squeeze(pixel_positions).reshape(positions.shape)
        return pixel_positions
    
    
    def apply_homography_matrix_torch(self, positions, homography_matrix):
            #traj_w = torch.tensor(traj_w)
        #H_inv = torch.tensor(H_inv)

        # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
        traj_homog = torch.cat((positions, torch.ones((positions.shape[0], 1))), dim=1).t()
        
        # to camera frame
        traj_cam = torch.matmul(homography_matrix, traj_homog).to(torch.float32)
        
        # to pixel coords
        traj_uvz = traj_cam / traj_cam[2]
        traj_uvz = traj_uvz.t()
        
        return traj_uvz[:, :2]
        