def tryarg1(a=0, b=0, c=0):
    return a+b+c

def tryarg2(d=0, e=0, f=0):
    return d+e+f

def manyArgs(g, **kwargs):
    print(kwargs)
    keys1 = ["a", "b", "c"]
    kwargs1 = {i: kwargs[i] for i in keys1 if i in kwargs.keys()}
    keys2 = ["d", "e", "f"]
    kwargs2 = {i: kwargs[i] for i in keys2 if i in kwargs.keys()}
    return g + tryarg1(**kwargs1) + tryarg2(**kwargs2)

def many2Args(y, **kwargs):
    return manyArgs(**kwargs)

p = many2Args(y=1, g=2,  d=0, a=1, b=2, c=3)
print(p)
