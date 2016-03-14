import numpy as np
import scipy.ndimage as ndimage

def test_func(values):
    return values.sum()

class game:
    def __init__(self):
        self.board = np.zeros((10,10),np.uint)

    def __next_iteration__(self):
        footprint = np.array([[1,1,1],[1,0,1],[1,1,1]])

        alive_neigbhours = ndimage.generic_filter(self.board, test_func, footprint=footprint)
        alive = np.where(self.board == 1 & (alive_neigbhours == 3) | (alive_neigbhours == 2))
        born = np.where(self.board == 0 & (alive_neigbhours == 3))

        self.board = np.zeros((10,10),np.uint)
        self.board[alive] = 1
        self.board[born] = 1
