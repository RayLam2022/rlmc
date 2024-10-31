import time

import numpy as np

ii=200
jj=300

def forloop(ii=200,jj=200):
    collection=[]
    collection2=[]
    for i in range(ii):
        for j in range(jj):
            if i%2==0 and j%2!=0:

                collection.append([i+2,j*1.3])
                collection2.append([i+2,j*1.3])
            else:
                collection.append([i*2,j+1])
    return collection,collection2

s1=time.time()
c,c2=forloop(ii,jj)
c=np.array(c)
c=c.reshape((ii,jj,2))
c2=np.array(c2)
c2=c2.reshape((-1,2))
#print(c)
print(time.time()-s1)




def numpyloop(ii=200,jj=200):
    # constructing array
    array=np.mgrid[0:ii:1,0:jj:1]
    array=np.transpose(array,(1,2,0))
    
    # constructing condition mask
    mask1=array[:,:,0]%2==0 
    mask2=array[:,:,1]%2!=0

    mask_combined=mask1&mask2
    mask_inversed=~mask_combined
    #print(mask_combined)

    # calculating new values
    array=array.astype(np.float32)
    #print(array[:4,:4])
    array[:,:,0][mask_combined]=array[:,:,0][mask_combined]+2
    array[:,:,1][mask_combined]=array[:,:,1][mask_combined]*1.3

    array[:,:,0][mask_inversed]=array[:,:,0][mask_inversed]*2
    array[:,:,1][mask_inversed]=array[:,:,1][mask_inversed]+1
    #print('--------------------')
    #print(array)
    return array,mask_combined,mask_inversed
    

s2=time.time()
np_c,mask_combined,mask_inversed=numpyloop(ii,jj)
np_c2=np_c[mask_combined]
print(time.time()-s2)
print('*********************')

print(np_c.shape)
gap=np.around(np_c-c, decimals=2)
print(np.all(gap==0))


print(np_c2.shape)
gap2=np.around(np_c2-c2, decimals=2)
print(np.all(gap2==0))
