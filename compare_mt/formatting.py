import re

class Formatter(object):
    pat_square_open  = re.compile("\[")
    pat_square_closed  = re.compile("\]")
    pat_lt  = re.compile("<")
    pat_gt  = re.compile(">")
    latex_substitutions = {
        pat_square_open: "{[}",
        pat_square_closed: "{]}",
        pat_lt: r"\\textless",
        pat_gt: r"\\textgreater"
    }

    def __init__(self, decimals=4):
        self.set_decimals(decimals)

    def set_decimals(self, decimals):
        self.decimals = decimals
    
    def escape_latex(self, x):
        """Adds escape sequences wherever needed to make the output
        LateX compatible"""
        for pat, replace_with in self.latex_substitutions.items():
            x = pat.sub(replace_with, x)
        return x

    def __call__(self, x):
        """Convert object to string with controlled decimals"""
        if isinstance(x, str):
            return self.escape_latex(x)
        elif isinstance(x, int):
            return f"{x:d}"
        elif isinstance(x, float):
            return f"{x:.{self.decimals}f}"
        else:
            str(x)

fmt = Formatter(decimals=4)