#!/usr/bin/env python3

# ----- test locals ----- #
def f():
  x = 1
  y = 2
  l = locals()
  print(l)
  z = 4
  l = locals()
  print(l)

# {'x': 1, 'y': 2}
# {'x': 1, 'y': 2, 'l': {...}, 'z': 4}
f()

# ----- test vars ----- #
class A:
    def __init__(self, x, y):
        self.x = x
        self.y = y

a = A(1, 2)

# vars accepts an object as parameter
print(vars(a)) # {'x': 1, 'y': 2}

# ----- test globals ----- #

"""
{'__name__': '__main__', '__doc__': None, '__package__': None,
'__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x10f7d6310>,
'__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>,
'__file__': './test_globals_locals.py', '__cached__': None, 'f': <function f at 0x10f9454d0>,
'A': <class '__main__.A'>, 'a': <__main__.A object at 0x10f89a690>}
"""
print(globals())
