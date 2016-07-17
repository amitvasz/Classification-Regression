import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):

    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    # IMPLEMENT THIS METHOD

    xcol = X.shape[1]
    #print X.shape[0]
    #print X.shape[1]
    #print y.shape[0]
    #print y.shape[1]
    a=np.unique(y)
    arow = a.shape[0]
    #print a.size
    means=np.zeros((xcol,arow))
    s=a.size
    for i in range(s):
        means[:,i]=np.mean(X[y.flatten()==a[i]],axis=0)

    #print means
    covmat=np.cov(X,rowvar=0)
    #print covmat
    return means,covmat



def qdaLearn(X,y):

    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    # IMPLEMENT THIS METHOD

    xcol = X.shape[1]
    a=np.unique(y)
    arow = a.shape[0]
    means=np.zeros((xcol,arow))
    covmats=[np.zeros((xcol,xcol))] * arow
    s=a.size
    for i in range(s):
        means[:,i]=np.mean(X[y.flatten()==a[i]],axis=0)
        covmats[i]=np.cov(X[y.flatten()==a[i]],rowvar=0)
    #print means
    #print covmats
    #print "in ldalearn"
    return means,covmats



def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    mcol=means.shape[1]
    xtrow=Xtest.shape[0]
    yt=ytest.flatten()
    d=det(covmat)
    #print "in ldatest"
    dist=np.zeros((xtrow,mcol))

    for i in range(mcol):
        sub=(Xtest - means[:,i])
        dotproduct=np.dot(sub,inv(covmat))
        sub2=np.sum(sub*dotproduct,1)
        up =np.exp(-1*sub2/2)
        pisqrt=sqrt(pi*2)
        squaring=(np.power(d,2))
        down=(pisqrt*squaring)
        dist[:,i]=up/down

    ypred=np.argmax(dist,1)
    ypred=ypred+1
    acc=100*np.mean(ypred==yt)

    #print "accuracy 1"
    #print acc
    #print ypred
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    # IMPLEMENT THIS METHOD

    mcol=means.shape[1]
    xtrow=Xtest.shape[0]
    yt=ytest.flatten()

    dist=np.zeros((xtrow,mcol))

    for i in range(mcol):
        invof = inv(covmats[i])
        d=det(covmats[i])
        sub=(Xtest - means[:,i])
        sub2=np.dot(sub,invof)
        sub3=np.sum(sub*sub2,1)
        sub4=(np.power(d,2))
        up =np.exp(-1*sub3/2)
        down=(sqrt(pi*2)*sub4)
        dist[:,i]=up/down

    ypred=np.argmax(dist,1)
    ypred=ypred+1
    acc=100*np.mean(ypred==yt)

    #print "accuracy 2"
    #print acc
    #print ypred
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD

    Xtrans = X.T
    XTX = np.dot(Xtrans,X)
    
    XTXinv = np.linalg.inv(XTX)
    #print XTXinv.shape
   
    XTY = np.dot(Xtrans,y)
    #print XTY.shape

    w = np.dot(XTXinv,XTY)
    #print w.shape
                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD

    #minimized function returns w

    #for w

    iden = np.identity(X.shape[1])
    
    lambdaiden = lambd*iden
    w1 = np.dot(X.T,X)
    w2 = w1 + lambdaiden
    w2 = np.linalg.inv(w2)
    w3 = np.dot(X.T,y)
    
    w = np.dot(w2, w3)
                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD

    #We have to return RMSE
    
    #print w.size
    
    rmse1 = np.dot(Xtest,w)
    
    rmse2 = ytest - rmse1
    
    rmse2 = np.square(rmse2)
    rmse2 = np.sum(rmse2)
    

    nval = Xtest.shape[0]

    rmse3 = rmse2/nval
    #print rmse3.shape

    rmse = np.sqrt(rmse3)
    #print rmse
	
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD

    #Jw is the error with Jw1 as first part of equation and Jw2 as the second part

    w = np.mat(w)
    
    #for Jw2
    wtrans = w.T
    wtw = np.dot(w,wtrans)
    #print wtrans.shape
    
    Jw2 = (lambd)*(wtw)
    Jw2 = Jw2/2
    #print Jw2.shape

    #for Jw1
    
    xw = np.dot(X,wtrans)
    
    yxw = (y - xw)
    
    yxwtrans = yxw.T

    Jw1 = np.dot(yxwtrans,yxw)
    
    Jw1 = Jw1/2
    #print Jw1

    #for Jw
    
    Jw = Jw1 + Jw2
    error = Jw
    #print "error is"
    #print Jw.shape

    #for error gradient Jwgrad = (XTX)W - XTy + LW

    Jwgrad1 = np.dot(X.T,X)
    
    Jwgrad1 = np.dot(Jwgrad1,w.T)

    Jwgrad2 = np.dot(X.T,y)
    
    Jwgrad3 = lambd*(w.T)

    Jwgrad = (Jwgrad1 - Jwgrad2) + Jwgrad3

    error_grad = np.array(Jwgrad).flatten()
    
    #print error.shape
    #print error_grad.shape                                         
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD

    #for higher powers
    m = x.shape[0]
    n = p+1
    
    res = np.ones((m,n))
    
    for i in range(m):
        for j in range(n):
            res[i][j] = np.power(x[i],j)

    Xd = res

    return Xd

# Main script

# Problem 1
# load the sample data  
                                                             
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
print 'Problem 1'
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print 'LDA Accuracy = ', ldaacc
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print 'QDA Accuracy = ', qdaacc

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()
plt.figure()
zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.show()

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
print 'Problem 2'
print'RMSE without intercept for test data ', mle
print'RMSE with intercept for test data ', mle_i


mle_tr = testOLERegression(w,X,y)

mle_i_tr = testOLERegression(w_i,X_i,y)

print'RMSE without intercept for training data ', mle_tr
print'RMSE with intercept for training data ', mle_i_tr

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3tr = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3tr[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
plt.figure()
m_rmse = np.min(rmses3)
min1 = np.min(rmses3tr)
print 'Problem 3'
print 'Minimum rmse value of ridge regression for test data: ', m_rmse #60.8920370937
print 'Minimum rmse value of ridge regress for training data: ', min1 #46.7670855937

#difference in weights of X_i
#print "weight prob2", w
#print "weight prob3", w_l
#print "difference", (w_l - w)
plt.plot(lambdas,rmses3)
plt.plot(lambdas,rmses3tr)
plt.legend(('Test Data','Train Data'))
plt.show()

# Problem 4
k = 201
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
rmses4train = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))

for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses4train[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
    #print "here"
print 'Problem 4'
plt.plot(lambdas,rmses4)
plt.plot(lambdas,rmses4train)
plt.legend(('Test Data','Train Data'))
#plt.xlabel('lambda')
#plt.ylabel('RMSE')

#plt.plot(lambdas,rmses4train)
plt.show()


# Problem 5
pmax = 7
#rmses4 = np.zeros((k,1))
lambda_opt = lambdas[np.argmin(rmses4)]
print 'Problem 5'
print 'lamba opt ', lambda_opt
rmses5 = np.zeros((pmax,2))
rmses5train = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    rmses5train[p,0] = testOLERegression(w_d1,Xd,y)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    rmses5train[p,1] = testOLERegression(w_d2,Xd,y)

plt.figure()
plt.plot(range(pmax),rmses5)
plt.xlabel('pmax')
plt.ylabel('RMSE')
plt.plot(range(pmax),rmses5train)
plt.legend(('Test data:No Regularization','Test data:Regularization','Train data:No Regularization','Train data:Regularization'))
#plt.legend(('test:No Regularization','test:Regularization'))
plt.show()
