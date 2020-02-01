# implementing a decorator for the fibonacci function in order to apply
# memoization for computation speed up

def memorize(f):
    cache = {}
    def fmem(x):
        if x in cache:
            return cache[x]
        y = f(x)
        cache[x] = y;
        return y;
    return fmem

@memorize
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

for x in [5, 35, 50, 80]:
    print('fib{%i} = %i' % (x, fib(x)))
