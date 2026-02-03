from src.transform.Meter_To_Pixel_Transformer import MeterToPixelInterface
import cv2
import numpy as np
import os
import torch


class ETHTransformer(MeterToPixelInterface):

    def __init__(self, homography_file_path) -> None:
        super().__init__(homography_file_path)
        # load homography matrix
        #homography_file_path = os.path.join(homography_file, 'H.txt')
        _, self.h = cv2.invert(np.loadtxt(homography_file_path))
        _, self.h_inv = cv2.invert(self.h)
        self.h_torch = torch.Tensor(self.h)
        self.h_inv_torch = torch.Tensor(self.h_inv)

    def get_pixel_positions(self, positions: np.array) -> np.array:
        """Transforms the given positions to the pixel positions"""
        pixel_positions = self.apply_homography_matrix(positions, self.h).astype(int)
        return self.change_x_and_y(pixel_positions)

    def get_positions(self, pixel_positions: np.ndarray):
        _, inverse_homography = cv2.invert(self.h)
        positions = self.apply_homography_matrix(self.change_x_and_y(pixel_positions), inverse_homography)
        return positions
    
    def get_pixel_positions_torch(self, positions) -> torch.Tensor:
        positions = self.apply_homography_matrix_torch(positions, self.h_torch).to(torch.int)
        return self.change_x_and_y(positions)
    
    def get_positions_torch(self, positions) -> torch.Tensor:
        positions = self.apply_homography_matrix_torch(self.change_x_and_y(positions), self.h_inv_torch)
        return positions

    
    
if __name__ == "__main__":
    transformer = ETHTransformer('eth')
    result = transformer.get_pixel_positions(np.array([50, 50]))
    print(result)