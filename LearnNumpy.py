#using numpy 
#http://www.tuicool.com/articles/r2yyei
from numpy import *
a = arange(15).reshape(5,3)
print a.shape
print a
print a.ndim
print a.dtype.name
print a.itemsize
print a.size
print type(a)

a+=a
print a
print a.max(axis=1)
a.shape = (3,5)
print a