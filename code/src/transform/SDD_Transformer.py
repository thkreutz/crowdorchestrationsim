from src.transform.Meter_To_Pixel_Transformer import MeterToPixelInterface
import numpy as np
import yaml
import os
import torch


class SDDTransformer(MeterToPixelInterface):

    def __init__(self, scale) -> None:
        super().__init__(scale)
        self.scale = scale

    def get_pixel_positions(self, positions: np.array) -> np.array:
        """Transforms the given positions to the pixel positions
        """
        return (positions / self.scale).astype(np.int32)

    def get_positions(self, pixel_positions: np.ndarray):
        return pixel_positions * self.scale
    
    
    def get_pixel_positions_torch(self, positions: torch.Tensor) -> torch.Tensor:

        return (positions / self.scale).int()
    
    def get_positions_torch(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Transforms given positions to the pixel positions in torch style
        """
        return positions * self.scale

if __name__ == "__main__":
    transformer = SDDTransformer('bookstore', 0)
    result = transformer.get_pixel_positions(np.array([50, 50]))
    print(result)