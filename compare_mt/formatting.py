import re

class Formatter(object):

    latex_substitutions = {
        re.compile("\["): "{[}",
        re.compile("\]"): "{]}",
        re.compile("<"): r"\\textless",
        re.compile(">"): r"\\textgreater"
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

    def __call__(self, x, latex=True):
        """Convert object to string with controlled decimals"""
        if isinstance(x, str):
            return self.escape_latex(x) if latex else x
        elif isinstance(x, int):
            return f"{x:d}"
        elif isinstance(x, float):
            return f"{x:.{self.decimals}f}"
        else:
            str(x)

fmt = Formatter(decimals=4)
