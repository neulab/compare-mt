class Formatter(object):

    def __init__(self, decimals=4):
        self.set_decimals(decimals)

    def set_decimals(self, decimals):
        self.decimals = decimals
    
    def __call__(self, x):
        """Convert object to string with controlled decimals"""
        if isinstance(x, str):
            return x
        elif isinstance(x, int):
            return f"{x:d}"
        elif isinstance(x, float):
            return f"{x:.{self.decimals}f}"
        else:
            str(x)

fmt = Formatter(decimals=4)