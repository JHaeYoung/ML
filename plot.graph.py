# %%
import numpy as np
import matplotlib.pyplot as plt
def val_show(v,name):
    print(name,' = ',v,type(v), v.shape)

np.random.seed(1)   
x= np.arange(10)
y=np.random.rand(10)

val_show(x,'x')
val_show(y,'y')


# %%
plt.plot(x,y)
plt.show()
# %%

def f(x): 
    y= (x-2)*x*(x+2)
    return y

#x= np.arange(10)
x= np.arange(-3,3,0.1)

y=f(x)

plt.plot(x,y)
plt.show()

# %%
def f(x,w): 
    y= (x-w)*x*(x+w)
    return y

#x= np.arange(10)
#x= np.arange(-3,3,0.1)
x= np.linspace(-100,100,10000)

y1=f(x,6)
y2=f(x,2)


plt.plot(x,y1,color ='blue', label = '$f1$')
plt.plot(x,y2,color ='orange', label = '$f2$')

plt.legend(loc = 'upper left')

plt.ylim(-100,100)
plt.xlim(-10,10)
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.show()
# %%

plt.figure(figsize=(10,3))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.title(i+1)
    plt.plot(x,f(x,i),'k')
    plt.xlim(-10,10)
    plt.ylim(-20,20)
    plt.grid(True)
plt.show()    
# %%
#구구단
def f2(x,w): 
    y= x*w
    return y

x = np.arange(2,10,1)

plt.figure(figsize=(5,5))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.plot(x,f(x,i+1),'b')
    plt.xlim(0,10)
    plt.title(i+1)
    plt.ylim(0,100)
    plt.grid(True)    

plt.show()    
# %%

def f3(x0,x1):
    r= 2*x0**2+x1**2
    ans = r*np.exp(-r)
    return ans
xn=100
x0= np.linspace(-2,2,xn)
x1= np.linspace(-2,2,xn)
y= np.zeros((len(x0),len(x1)))

val_show(x0,'x0')
val_show(x1,'x1')
val_show(y,'y')

for i0 in range(xn):
    for i1 in range(xn):
        y[i1,i0]= f3(x0[i0], x1[i1])
        np.round(y,1)  
plt.figure(figsize=(5,5))
plt.gray()
plt.pcolor(y)
plt.colorbar()
plt.show()  
# %%

from mpl_toolkits.mplot3d import Axes3D
xx0, xx1 = np.meshgrid(x0,x1)

ax = plt.subplot(1,1,1,projection = '3d')
ax.plot_surface(xx0,xx1,y,rstride =1, cstride =1, alpha = 0.3
                ,color = 'blue', edgecolor ='black')
ax.set_zticks((0,0.2))
ax.view_init(45,-125)
# plt.figure(figsize=(6, 5))
# plt.gray()
# plt.pcolor(y)
# plt.colorbar()
plt.show()  
# %%
xn=50
x0= np.linspace(-2,2,xn)
x1= np.linspace(-2,2,xn)
y= np.zeros((len(x0),len(x1)))

for i0 in range(xn):
    for i1 in range(xn):
        y[i1,i0]= f3(x0[i0], x1[i1])
        np.round(y,1) 

xx0, xx1 = np.meshgrid(x0,x1)
plt.figure(figsize=(6,5))
cont =plt.contour(xx0,xx1,y,5,colors ='black')
cont.clabel(fmt = '%3.2f',fontsize=8)
plt.xlabel('$x_0$', fontsize =14)
plt.ylabel('$x_1$', fontsize =14)
plt.show()
# %%
