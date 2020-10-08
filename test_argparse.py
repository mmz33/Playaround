#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--append-action', default=[], action='append')
parser.add_argument('--list-type-nargs', type=list, nargs='+')
parser.add_argument('--int-type-nargs', type=int, nargs='+')
parser.add_argument('--str-type-nargs', type=str, nargs='+')
parser.add_argument('--list-type', type=list)
args = parser.parse_args()

print(args)
