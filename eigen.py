import math
import numpy
f1 = [[1,2,3,4],[5,6,7,8]]
f2 = [[1,2],[3,4],[5,6],[7,8]]
print(len(f1), "x", len(f1[0]))
print(len(f2), "x", len(f2[0]))
print(numpy.array([1,2,3]))

n = int(input("n x n Matrix where n : "))
matx = []
for i in range(n):
    rowList = []
    for j in range(n):
        #rowList.append(dataList[rowCount * i + j])
        #alpha = ['a','b','c','d','e','f','h','i','j','k','l']
        rowList.append( int(input("element of %d x %d : " % (i+1, j+1) ) ) )
    matx.append(rowList)
print(matx)


eigs = []
if n==2:
    eig1 = (matx[0][0]+matx[1][1]+math.sqrt( (matx[0][0]+matx[1][1])**2-4*(matx[0][0]*matx[1][1]-matx[0][1]*matx[1][0]) ) )/2
    eig2 = (matx[0][0]+matx[1][1]-math.sqrt( (matx[0][0]+matx[1][1])**2-4*(matx[0][0]*matx[1][1]-matx[0][1]*matx[1][0]) ) )/2
    eigs.append( eig1 )
    eigs.append( eig2 )
print(eigs)
