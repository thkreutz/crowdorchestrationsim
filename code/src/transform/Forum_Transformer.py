from src.transform.Meter_To_Pixel_Transformer import MeterToPixelInterface
import numpy as np
import cv2
import os
import torch

class ForumTransformer(MeterToPixelInterface):

    def __init__(self, homography_file_path=None) -> None:
        super().__init__(homography_file_path)
        # load homography matrix
        # homography_file_path = os.path.join(homography_file, 'H.txt')
        
        
        #_, self.h = cv2.invert(np.loadtxt(homography_file_path)) 
        
        # from H.json -> only one scene, so this does not change..
        _, self.h = cv2.invert(np.array([[ 2.44923560e-02,  1.10790505e-03,  6.62031019e-03],
                                         [-1.26360413e-03,  2.48863885e-02,  6.06410516e-03],
                                        [-7.01804022e-05,  4.26454048e-05,  1.00000000e+00]]))
        
        _, self.h_inv = cv2.invert(self.h)
        
        #print(self.h, self.h_inv)
        
        self.h_torch = torch.Tensor(self.h)
        self.h_inv_torch = torch.Tensor(self.h_inv)

    def get_pixel_positions(self, positions: np.array) -> np.array:
        """Transforms the given positions to the pixel positions"""

        # we stay in pixel space..

        return positions
        #return self.change_x_and_y(positions) 
    
    def get_positions(self, pixel_positions: np.array):
        return self.change_x_and_y(pixel_positions)
        
    
    def get_pixel_positions_torch(self, positions) -> torch.Tensor:
        
        
        ### This line in lodar_gcs.py l92 took my 5 hours for w2pixel transform.
        ### raw_dataset[["pos_x", "pos_y"]] = pd.DataFrame(world_coords * 0.8 <<<<----- )
        ### => Must divide by 0.8 because loader in OpenTraj multiplies by 0.8 when loading the data...
        #positions = self.apply_homography_matrix_torch(positions, self.h_torch).to(torch.int)    
        return self.change_x_and_y(positions)
    
    def get_positions_torch(self, positions) -> torch.Tensor:
        #positions = self.apply_homography_matrix_torch(self.change_x_and_y(positions), self.h_inv_torch)

        # we stay in pixel space...

        return self.change_x_and_y(positions)