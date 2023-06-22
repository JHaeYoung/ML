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

import matplotlib.pyplot as plt

def f(w0,w1):
    return w0**2+2*w0*w1+3
def df_dw0(w0,w1):
    return 2*w0+2*w1
def df_dw1(w0,w1):
    return 2*w0

w_range = 2
dw = 0.25

w0= np.arange(-w_range, w_range+dw,dw)
w1= np.arange(-w_range, w_range+dw,dw)
wn = w0.shape[0]   
ww0, ww1 = np.meshgrid(w0,w1) 

plt.figure(figsize=(4,4))
y=f(ww0,ww1)
ax = plt.subplot(1,1,1,projection = '3d')
ax.plot_surface(ww0,ww1,y,rstride =1, cstride =1, alpha = 0.3
                ,color = 'blue', edgecolor ='black')

plt.xticks(range(-w_range, w_range+1,1))
plt.yticks(range(-w_range, w_range+1,1))

plt.xlim(-w_range-0.5,w_range+0.5)
plt.ylim(-w_range-0.5,w_range+0.5)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)

#ax.view_init(40, 80)


# ax.set_zticks((0,0.2))
# ax.view_init(0,-180)

{
# ff= np.zeros((len(w0),len(w1)))
# dff_dw0 = np.zeros((len(w0),len(w1)))
# dff_dw1 = np.zeros((len(w0),len(w1)))

# for i0 in range(wn):
#     for i1 in range(wn):
#         ff[i1,i0] = f(w0[i0],w1[i1])
#         dff_dw0[i1,i0] = df_dw0(w0[i0],w1[i1])
#         dff_dw1[i1,i0] = df_dw1(w0[i0],w1[i1])

# plt.figure(figsize=(9,4))
# plt.subplots_adjust(wspace=0.3)
# plt.subplot(1,2,1)  
# cont = plt.contour(ww0,ww1,ff,10,colors = 'k')  
# cont.clabel(fmt='%2.0f', fontsize=8)
# plt.xticks(range(-w_range,w_range+1,1))
# plt.yticks(range(-w_range,w_range+1,1))
# plt.xlim(-w_range-0.5, w_range+0.5)
# plt.ylim(-w_range-0.5, w_range+0.5)
# plt.xlabel('$w_0$',fontsize =14)
# plt.ylabel('$w_1$',fontsize =14)

# plt.subplot(1,2,2)  
# plt.quiver(ww0,ww1,dff_dw0,dff_dw1)
# plt.xlabel('$w_0$',fontsize =14)
# plt.ylabel('$w_1$',fontsize =14)
# plt.xticks(range(-w_range,w_range+1,1))
# plt.yticks(range(-w_range,w_range+1,1))
# plt.xlim(-w_range-0.5, w_range+0.5)
# plt.ylim(-w_range-0.5, w_range+0.5)
# plt.show()
}
# %%
import numpy as np
import matplotlib.pyplot as plt

def f(w0, w1):
    return w0**2 + 2*w0*w1 + 3 #  x^2 + 2xy + 3
def df_dw0(w0, w1):
    #f = 2*w0 + 2*w1
    return 2*w0 + 2*w1
def df_dw1(w0, w1):
    #f = 2*w0
    return 2*w0

w_range = 10
dw = 1
w0 = np.arange(-w_range, w_range+dw, dw)
w1 = np.ones(len(w0))
w1 *= -1
print('w0 = ', w0)
print('w1 = ', w1)
w0 = np.linspace(-w_range, w_range, 50)

_w1 = 2

w1 = np.full(w0.shape, _w1)
print('w0 = ', w0)
print('w1 = ', w1)

#print('df = ', f(w0, w1))
#print('df_dw0 = ', df_dw0(w0, w1))
print('df_dw1 = ', df_dw1(w0, w1))

plt.figure(figsize=(5,5))
plt.plot(w0,f(w0,w1), 'black', linewidth = 3, label = '$w_1=-1$')
plt.plot(w0,df_dw0(w0,w1), 'cornflowerblue', linewidth = 3, label = '$dw_1$')
plt.title('$w_1=-1$')
plt.xlabel('$w_0$')
plt.ylabel('$y$')
plt.ylim(-10,10)
plt.xlim(-5,5)
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('fig4_12_w1(-1).png')

# %%
import matplotlib.pyplot as plt

def f(w0,w1):
    return w0**2+2*w0*w1+3
def df_dw0(w0,w1):
    return 2*w0+2*w1
def df_dw1(w0,w1):
    return 2*w0

w_range = 2
dw = 0.25

w0= np.arange(-w_range, w_range+dw,dw)
w1= np.arange(-w_range, w_range+dw,dw)
wn = w0.shape[0]   
ww0, ww1 = np.meshgrid(w0,w1) 

ff= np.zeros((len(w0),len(w1)))
dff_dw0 = np.zeros((len(w0),len(w1)))
dff_dw1 = np.zeros((len(w0),len(w1)))

for i0 in range(wn):
    for i1 in range(wn):
        ff[i1,i0] = f(w0[i0],w1[i1])
        dff_dw0[i1,i0] = df_dw0(w0[i0],w1[i1])
        dff_dw1[i1,i0] = df_dw1(w0[i0],w1[i1])

plt.figure(figsize=(9,4))
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,2,1)  
cont = plt.contour(ww0,ww1,ff,10,colors = 'k')  
cont.clabel(fmt='%2.0f', fontsize=8)
plt.xticks(range(-w_range,w_range+1,1))
plt.yticks(range(-w_range,w_range+1,1))
plt.xlim(-w_range-0.5, w_range+0.5)
plt.ylim(-w_range-0.5, w_range+0.5)
plt.xlabel('$w_0$',fontsize =14)
plt.ylabel('$w_1$',fontsize =14)

plt.subplot(1,2,2)  
plt.quiver(ww0,ww1,dff_dw0,dff_dw1)
plt.xlabel('$w_0$',fontsize =14)
plt.ylabel('$w_1$',fontsize =14)
plt.xticks(range(-w_range,w_range+1,1))
plt.yticks(range(-w_range,w_range+1,1))
plt.xlim(-w_range-0.5, w_range+0.5)
plt.ylim(-w_range-0.5, w_range+0.5)
plt.show()

# %%

import numpy as np

A= np.array(
    [
        [1,2],
        [3,4]
    ]
)

A_Inv =np.linalg.inv(A)
print(A_Inv)
I_Inv = A_Inv.dot(A)
I= np.round(I_Inv,2) # 소수점 이하 절삭
print(I)

# %%
A = np.array(
    [
        [2,-1],
        [1,1]
    ]
)
B = np.array(
    [
        [0],
        [3]
    ]
)
C = np.array(
    [
        [0],
        [0]
    ]
)
A_Inv =np.linalg.inv(A)
C= A_Inv.dot(B)
I= np.round(C,2)
print(C)
# %%


