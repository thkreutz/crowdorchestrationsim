from src.transform.Meter_To_Pixel_Transformer import MeterToPixelInterface
import cv2
import numpy as np
import os
import torch

class UCYTransformer(MeterToPixelInterface):

    def __init__(self, homography_file_path) -> None:
        super().__init__(homography_file_path)
        """
        Parameters:
        dataset_name -- name of the dataset
        """

        # load homography matrix
        #folder_path = self.dataset.folder_path
       # homography_file_path = os.path.join(folder_path, 'H.txt')
        _, self.h = cv2.invert(np.loadtxt(homography_file_path))
        _, self.h_inv = cv2.invert(self.h)
        self.h_torch = torch.Tensor(self.h)
        self.h_inv_torch = torch.Tensor(self.h_inv)
        
        
    def get_pixel_positions(self, positions: np.array) -> np.array:
        """Transforms the given positions to the pixel positions"""
        pixel_positions = self.apply_homography_matrix(positions, self.h)

        # homography matrix is not exact, so the following adjustments need to be done
        
        # 
        # tkr * np.array([1, -1]) + np.array([0, 576]) + np.array([365, -270])
        # old: return pixel_positions * np.array([1, -1]) + np.array([0, 576]) + np.array([370, -280])
        return (pixel_positions * np.array([1, -1]) + np.array([0, 576]) + np.array([365, -280])).astype(np.int32)

    def get_positions(self, pixel_positions: np.ndarray):
        _, inverse_homography = cv2.invert(self.h)

        # old
        #pixel_positions = pixel_positions - (np.array([0, 576]) + np.array([370, -280]))
        #pixel_positions = pixel_positions / np.array([1, -1])

        ## tkr
        pixel_positions = pixel_positions - (np.array([0, 576]) + np.array([365, -280]))
        pixel_positions = pixel_positions / np.array([1, -1])

        positions = self.apply_homography_matrix(pixel_positions, inverse_homography)
        return positions
    
    
    def get_pixel_positions_torch(self, positions) -> torch.Tensor:
        positions = self.apply_homography_matrix_torch(positions, self.h_torch).to(torch.int)
        return self.change_x_and_y(positions)
    
    
    def get_positions_torch(self, positions) -> torch.Tensor:
        positions = self.apply_homography_matrix_torch(self.change_x_and_y(positions), self.h_inv_torch)
        return positions

if __name__ == "__main__":
    transformer = UCYTransformer('students003')
    result = transformer.get_pixel_positions(np.array([50, 50]))
    print(result)