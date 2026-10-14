"""
setting up python PATH at runtime to allow for access of modules in subdirectories

"""

import sys

def setupPATH():
    sys.path.append('./util')
    sys.path.append('./config')