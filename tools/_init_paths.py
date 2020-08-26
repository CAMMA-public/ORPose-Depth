from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', 'lib')
add_path(lib_path)
