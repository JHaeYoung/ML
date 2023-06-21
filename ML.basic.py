# %%

import numpy as np
def val_show(v,name):
    print(name,' = ',v,type(v), v.shape)


a = np.array([1,3])
b = np.array([2,1])
c= a+b
val_show(a,'a')
val_show(b,'b')
val_show(c,'c')

# %%
# transpose .. 전치
a= np.array(
[
    [1,2,3],
    [7,4,5] ,
    [531,12,55]
]
)

a = a.T
val_show(a,'a')
# %%
#전치행렬
a= a.T
#스칼라 곱
a= 10*a
# %%
a = np.array([1,3])
b = np.array([2,1])
c= a.dot(b)
val_show(c,'c')
# %%
# 벡터의 크기
b = np.array([5,4])
scale = np.linalg.norm(b)
print(scale)

# %%
