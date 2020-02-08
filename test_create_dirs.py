#!/usr/bin/env python3

import sys
import os

fname=sys.argv[1]

dir_name = os.path.dirname(fname)
if os.path.isdir(dir_name):
  if not os.path.exists(dir_name):
    print('Creating {}'.format(dir_name))
    os.makedirs(dir_name)
  else:
    print('{} is already created'.format(dir_name))

with open(fname, 'w') as f:
  f.write('worked!')
