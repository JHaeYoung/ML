# %%
# 회귀 - regression
# 사람 키 ...150 ~ 180
# 음식 섭취 ..3  2  1

import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성

np.random.seed(1)  
X_min = 4
X_max = 30
X_n = 16
X = 5+25*np.random.rand(X_n)
Prm_C = [170,108,0.2]
T = Prm_C[0] - Prm_C[1]*np.exp(-Prm_C[2]*X)+4*np.random.rand(X_n)

print('T =',T)
plt.figure(figsize= (5,5))
#plt.plot(X,T,'red',linewidth=1,label ='age')
plt.scatter(X,T)
plt.xlim(0,30)
plt.ylim(100,200)
plt.grid(True)


# %%
import numpy as np
import matplotlib.pyplot as plt

import csv
samsung_stock = []
with open('samsung_stock.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        samsung_stock += row
T = [int(i) for i in samsung_stock]
X = [int(x) for x in range(len(samsung_stock))]
X_min = 1
X_max = len(X)

print('X=',X)
print('T=',T)
plt.figure(figsize=(5,5))
plt.scatter(X,T)
plt.xlim(X_min,X_max)
plt.grid(True)

def mse_line(x,t,w):
    y = w[0]*x+w[1]
    mse = np.mean((y-t)**2)
    return mse
def dmse_line(x,t,w):
    y = w[0]*x+w[1]
    d_w0 = 2*np.mean((y-t)*x)
    d_w1 = 2*np.mean(y-t)
    return d_w0, d_w1

#d_w = dmse_line(X,T,[12.5, 25000])

#d_w = np.round(d_w,1)

#경사 하강법
def fit_line_num(x,t):
    w_init=[12,25000]
    alpha = 0.001 #lre
    i_max = 100000
    eps = 0.1 #반복을 종료 기울기의 절대값의 한계
    w_i = np.zeros([i_max,2])
    w_i[0,:] = w_init
    for i in range(1, i_max):
        dmse = dmse_line(X,T,w_i[i-1])
        w_i[i,0] = w_i[i-1,0] - alpha*dmse[0]
        w_i[i,1] = w_i[i-1,1] - alpha*dmse[1]
        if(max(np.absolute(dmse))<eps):
            break
    w0 = w_i[i,0]
    w1 = w_i[i,1]
    w_i = w_i[:i,:]
    return w0, w1, dmse, w_i

def fit_line(x,t):
    mx = np.mean(x)
    mt = np.mean(t)
    mtx = np.mean(t*x)
    mxx = np.mean(x*x)
    w0 = (mtx-mt*mx)/(mxx-mx**2)
    w1 = mt-w0*mx
    return np.array([w0,w1])




#main
plt.figure(figsize=(4,4))
xn = 1225
w0_range = [0,1225]
w1_range = [20000,60000]
x0 = np.linspace(w0_range[0], w0_range[1], xn,dtype=int)
x1 = np.linspace(w1_range[0], w1_range[1], xn,dtype=int)

# x0 = np.round(x0).astype(int)
# x1 = np.round(x1).astype(int)  


xx0, xx1 = np.meshgrid(x0,x1)
J = np.zeros((len(x0),len(x1)))

for i0 in range(xn):
    for i1 in range(xn):            
        J[i1,i0] = mse_line(X,T, (x0[i0], x1[i1]))

#display
cont = plt.contour(xx0,xx1,J,30,colors='black',levels=[100,1000,10000,100000])
cont.clabel(fmt='%1.0f',fontsize=8)
plt.grid(True)

#경사 하강법 호출
W0, W1, dMSE, W_history = fit_line_num(X,T)
#결과
print(f'반복 횟수{W_history.shape[0]}')
#print(f'W={W0:.6f}, {W1:.6f}')
print(f'해석해 ----  w0={W0:.3f}, w1={W1:.3f}')
print(f'dMSE=[{dMSE[0]:.6f}, {dMSE[1]:.6f}]')
print(f'MSE=[{mse_line(X,T,[W0,W1]):.6f}]')
plt.plot(W_history[:,0],W_history[:,1], '.-',color='gray', markersize=10,markeredgecolor='cornflowerblue')
plt.grid(True)
plt.show()


W0, W1 = fit_line(X,T)
print(f'LMS ----  w0={W0:.3f}, w1={W1:.3f}')

# %%
