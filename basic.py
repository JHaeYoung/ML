# %%
# 변수 ....

#c/c++
# int ... type  변수  = 데이타
#code -> 해석 -> compiler ->타입 지정 ... 메모리 관리

#python

            # 변수 =  데이타
a=10 # type init
print(a, type(a))
a=0.123

# %%
# c/c++ struct ... array ... class
# python - list ... read/write/add/remove
a=10
b=20

v = [a,b]
v[0]#a
v[1]#b

print(v, type(v))
print(v[0], type(v[0]))
print(v[1], type(v[1]))

#tuple ... read only ... packed
# 데이타를 전송할 떄 오염이나 변경이 되지 않게 하기 위한 묶음
a=(10,20,30)
print(a,type(a))
print(a[0],type(a[0]))

#dict ... db/add/remove/search
value = 'value'
a= {'key' : value}
print(a,type(a))
a= {'key1' : value+'1'}
#a= {'key2' : value+'2'}
#a= {'key3' : value+'3'}
#a= {'key4' : value+'4'}
#a= {'key5' : value+'5'}
print(a,type(a))

a['key2'] = 'value2'
a['key3'] = [1,2,3,4,5]
print(a['key3'], type(a['key3']))

# %%

#Point** ... vector<vector<Point>> contours

#string
str1 = 'hello'
str2 = 'world'

msg = '%s %s'%(str1, str2) #tuple로 감싸서 전달
print(msg)
msg = f'{str1}  {str2}'
print(msg)

# %%
#if ... for
for x in range(1, 10+1,1):
    if x%2==0:
        msg = f'even {x}'
    else:
        msg = f'odd {x}'
    print(msg)

# %%
for j in range(1, 10):
    for i in range(2, 10):
        result = i * j
        print(f"{i} x {j} = {result}", end="  ")
    print()

# %%
