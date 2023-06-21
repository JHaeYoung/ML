# %%
import numpy as np

a= [1,2,3] 
#b= [4,5,6]

#print(a+b,type(a+b))
# %%
#n차열 dimension array
a= np.array([1,2,3])
b= np.array([4,5,6])

print(a,type(a))
print(a+b,type(a+b))

# %%
c = b # 얕은 복사... 주소값만 공유
print(c,type(c))

c = b.copy() # 깊은 복사 ... 모든 데이터 새로운 공간에 생성
print(c,type(c))


# %%
b[1] = 100
print('b=',b,type(b))
print('c=',c,type(c))
# 깊은 복사 후에 b의 값이 변경되도 c의 값은 변경 안됨

# %%

a = [x for x in range(10)] 
print('a= ',a,type(a))
# %%
a = np.arange(12,dtype=np.int64)
print(a,type(a),'shape =',a.shape)    


# %%
# transform shape as reshape(m,n)
a= a.reshape(2,5)
print('a=',a,type(a), 'shape =',a.shape)    

# %%
#linear algebra
# matrix 곱 = (mxn)dot(nxk) = mxk
a= a.reshape(3,4)
print('a=',a,type(a), 'shape =',a.shape) 
a= a.reshape(4,3)
print('a=',a,type(a), 'shape =',a.shape) 
a= a.reshape(2,3,2)
print('a=',a,type(a), 'shape =',a.shape) 

# %%
# zero matrix
a= np.zeros(10) 
print('a =', a, type(a),a.shape)
a= a.reshape(2,5)
print('a =', a, type(a),a.shape)

# %%
v= np.array([
    [1,1,1],
    [2,3,4]
])
w =np.array(
    [
        [1,1],
        [2,4],
        [5,2]
    ]
)

z= v.dot(w)
print('z=',z ,type(z), 'shape=', z.shape)   


# %%

dan = np.array(
    [
        [2,2,2,2,2,2,2,2,2],
        [3,3,3,3,3,3,3,3,3],  
        [4,4,4,4,4,4,4,4,4],  
        [5,5,5,5,5,5,5,5,5]  
    ]
)
step = np.array(
    [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9]
    ]
)
result = dan.dot(step)  
for x in range(2,6,1):
    sum = f'{x}단의 총합 = {result[x-2]}'
    print(sum)


# %%
dan = np.ones((4,9))
idx = 0
for x in range(2,6,1):
    dan[idx]= x
    idx+=1
step = np.arange(1,10)

G = dan*step
z = dan.dot(step) 
for x in range(2,6,1):
    for y in range(1,10,1):
        result = f'{x}X{y} = {G[x-2][y-1]}'
        print(result)      
    print('====='*8)    
    print(f'{x}단의 총합 = {z[x-2]}')
    print('====='*8) 



# %%
np.random.seed(1)   
x= np.arange(10)
y=np.random.rand(10)
