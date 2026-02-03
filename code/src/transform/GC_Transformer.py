from src.transform.Meter_To_Pixel_Transformer import MeterToPixelInterface
import numpy as np
import cv2
import os
import torch

class GCTransformer(MeterToPixelInterface):

    def __init__(self, homography_file_path=None) -> None:
        super().__init__(homography_file_path)
        # load homography matrix
        # homography_file_path = os.path.join(homography_file, 'H.txt')
        
        
        #_, self.h = cv2.invert(np.loadtxt(homography_file_path)) 
        
        # from H.json -> only one scene, so this does not change..
        _, self.h = cv2.invert(np.array([[4.97412897e-02, -4.24730883e-02, 7.25543911e+01],
                                        [1.45017874e-01, -3.35678711e-03, 7.97920970e+00],
                                        [1.36068797e-03, -4.98339188e-05, 1.00000000e+00]]))
        
        _, self.h_inv = cv2.invert(self.h)
        
        #print(self.h, self.h_inv)
        
        self.h_torch = torch.Tensor(self.h)
        self.h_inv_torch = torch.Tensor(self.h_inv)

    def get_pixel_positions(self, positions: np.array) -> np.array:
        """Transforms the given positions to the pixel positions"""
        return self.change_x_and_y(self.apply_homography_matrix(positions / 0.8, self.h)) 
    
    def get_positions(self, pixel_positions: np.array):
        return self.apply_homography_matrix(self.change_x_and_y(pixel_positions), self.h_inv) * 0.8
        
    
    def get_pixel_positions_torch(self, positions) -> torch.Tensor:
        
        
        ### This line in lodar_gcs.py l92 took my 5 hours for w2pixel transform.
        ### raw_dataset[["pos_x", "pos_y"]] = pd.DataFrame(world_coords * 0.8 <<<<----- )
        ### => Must divide by 0.8 because loader in OpenTraj multiplies by 0.8 when loading the data...
        positions = self.apply_homography_matrix_torch(positions / 0.8 , self.h_torch).to(torch.int)    
        return self.change_x_and_y(positions)
    
    def get_positions_torch(self, positions) -> torch.Tensor:
        positions = self.apply_homography_matrix_torch(self.change_x_and_y(positions), self.h_inv_torch)
        return positions * 0.8