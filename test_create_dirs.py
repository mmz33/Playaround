#!/usr/bin/env python3

import sys
import os

fname=sys.argv[1]

if '/' in fname:
  dir_name = os.path.dirname(fname)
  os.makedirs(dir_name, exist_ok=True) # python3 way

with open(fname, 'w') as f:
  f.write('worked!')

