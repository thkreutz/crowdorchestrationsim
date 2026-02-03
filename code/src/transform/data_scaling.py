########################
#### stack all the scenes together and scale.
# 
def min_max_scale(x):
    x_min = x.min()
    x_max = x.max()
    new_x = (x - x_min) / (x_max - x_min)
    return new_x, x_min, x_max

def min_max_scale_given(x, x_min, x_max):
    new_x = (x - x_min) / (x_max - x_min)
    return new_x

def reverse_min_max_scale(x, x_min, x_max):
    return (x * (x_max - x_min)) + x_min


class MinMaxScaler:
    def __init__(self):
        self.min_maxscale_dict = {}
    
    def fit(self, x, key):
        x_, x_min, x_max = min_max_scale(x)
        self.min_maxscale_dict[key] = (x_min, x_max)

    def fit_transform(self, x, key):
        x_, x_min, x_max = min_max_scale(x)
        self.min_maxscale_dict[key] = (x_min, x_max)
        return x_
        
    def transform(self, x, key):
        x_min, x_max = self.min_maxscale_dict[key]
        return min_max_scale_given(x, x_min, x_max)
    
    def inverse_transform(self, x, key):
        x_min, x_max = self.min_maxscale_dict[key]
        return reverse_min_max_scale(x, x_min, x_max)