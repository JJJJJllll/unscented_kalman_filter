dt = 1
pret = 10
dt, pret, v = 20 - pret, 20, 10/dt
print(dt, pret, v)

import numpy as np
a = np.array([[1],[2],[3],[4]])

print(np.take(a, [0,2]).reshape(2,1))
# print(a[0,1,2])

import numpy as np

a = np.array([1,3,5])
b = np.array([2,4,6])

c = np.empty((a.size + b.size,), dtype=a.dtype)
c[0::2] = a
c[1::2] = b
print(c)
print(c[0::2])

a = np.array([[1],[2],[3],[4],[5],[6]])
print(a[0::2])
a[0::2] += a[1::2] * 1
print(a)

a = np.array([[1,2,3],[3,4,5],[5,6,7]])
print('\n',a[0:2,0:2])


ukf_filter = np.zeros(shape=(6,6))
ukf_filter[0:2, 0:2]=ukf_filter[2:4, 2:4]=ukf_filter[4:6, 4:6] = np.array([[1,2],[3,4]])
print(ukf_filter)

pret = 10
nowt = 20
dt = 1
dt, pret = nowt - pret, nowt
print(dt, pret)
pret, dt = nowt, nowt - pret
print(dt, pret)