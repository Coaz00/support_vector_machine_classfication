import numpy as np
import random
import matplotlib.pyplot as plt
import cvxopt

random.seed(10)

# loading data into predictor matrix and desired value vector
def load_data(filename):
    # reading data
    data = np.loadtxt(filename,delimiter=',',dtype=float)
    # first 5 columns are predictors, last column in true value

    X = data[:,:-1] # predictor matrix
    Y = data[:,-1].reshape(data.shape[0],1) # true value vector
    
    return X,Y

def split_train_test(X,Y,p=0.8):
    
    m,n = X.shape
    
    c = int(m*p)
    
    indices = [i for i in range(m)]
    random.shuffle(indices)
    
    X_train = X[indices[:c],:]
    Y_train = Y[indices[:c]]
    
    X_test = X[indices[c:],:]
    Y_test = Y[indices[c:]]
    
    return X_train, Y_train, X_test, Y_test

def standardization(X_train,X):
    meanX = np.mean(X_train, axis = 0)
    stdX = np.std(X_train, axis = 0)
    X = (X - meanX)/stdX
    
    return X

def calc_gamma(w,b,X,Y):
    gamma = np.zeros(len(Y))
    for i in range(len(gamma)):
        gamma[i] = Y[i] *(w.T@X.T[:,i]+b)
        
    return gamma

def find_support_vector_indices_primal(w,b,X,Y,tol):
    gamma = calc_gamma(w,b,X,Y)
    
    return gamma.reshape((-1,)) < (1 + tol)
    
def find_support_indices_dual(alpha,tol):
    return alpha > tol

def plot_primal(X,Y,b,w,ksi):
    
    X1 = X[Y.reshape(-1,) == 1,:]
    X_1 = X[Y.reshape(-1,) == -1,:]
    
    tol = 1e-6
    support_indices = find_support_vector_indices_primal(w,b,X,Y,tol)
    
    support_vectors = X[support_indices,:]
    support_ksi = ksi[support_indices]
    
    plt.figure()
    plt.scatter(X1[:,0],X1[:,1])
    plt.scatter(X_1[:,0],X_1[:,1])
    plt.scatter(support_vectors[:,0],
                support_vectors[:,1],
                c='yellow',s = 5,marker="^")

    for i in range(len(support_ksi)):
        plt.annotate(f"{support_ksi[i]:1.2f}",(support_vectors[i,0],support_vectors[i,1]),size=8)

    plt.plot([-2,2],[2*w[0]/w[1]-b/w[1],-2*w[0]/w[1]-b/w[1]],c='black')
    plt.xlim([-2,2])
    plt.show()
    
def plot_dual(X,Y,alpha,b,kernel_type,params):
    X1 = X[Y.reshape(-1,) == 1,:]
    X_1 = X[Y.reshape(-1,) == -1,:]
    
    xs = np.arange(-2,2,0.01)
    ys = np.arange(-2,2,0.01)
    
    N = len(xs)
    
    xy = np.zeros((N*N,2))
    
    for i in range(N):
        for j in range(N):
            xy[i*N+j,:] = np.array([xs[i],ys[j]])
           
    tol = 1e-6
    support_indices = find_support_indices_dual(alpha, tol)
    
    support_alpha = alpha[support_indices]
    support_vectors = X[support_indices,:]
    support_output = Y[support_indices]
    
    h = np.sum(support_alpha*support_output.reshape((-1,))*kernel(X,support_vectors,kernel_type,params),axis=1) + b
    ksi = np.clip(1 - Y.reshape((-1,))*h,0,None)
    support_ksi = ksi[support_indices]
    
    h = np.sum(support_alpha*support_output.reshape((-1,))*kernel(xy,support_vectors,kernel_type,params),axis=1) + b
    h = h.reshape((N,N))
       
    plt.figure()
    plt.scatter(X1[:,0],X1[:,1])
    plt.scatter(X_1[:,0],X_1[:,1])
    plt.scatter(support_vectors[:,0],
                support_vectors[:,1],
                c='yellow',s = 5,marker="^")
    plt.contour(xs, ys, h.T, levels=[0])
    
    for i in range(len(support_ksi)):
        plt.annotate(f"{support_ksi[i]:1.2f}",(support_vectors[i,0],support_vectors[i,1]),size=8)
    
    return h
    
def predict_primal(w,b,X):
    return np.sign(w@X.T+b).reshape(-1,1)

def calc_hinge_loss(w,b,X,Y):
    gamma = calc_gamma(w,b,X,Y)
    return np.sum(np.clip(1-gamma,0,None))

def solve_primal_problem(X,Y,C):
    m, n = X.shape
    
    # Equation to minimize
    P = np.zeros((m+n+1,m+n+1))
    for i in range(n):
        P[i+1,i+1] = 1
        
    q = C*np.ones(m+n+1)
    q[:n+1] = 0
    
    # Constraints
    G = np.zeros((2*m,m+n+1))
    
    G[:m,0] = Y.reshape((-1,))
    G[:m,1:n+1] = X*Y
    G[:m,n+1:] = np.eye(m)
    G[m:,n+1:] = np.eye(m)
    
    G = -G
    
    h = np.zeros(2*m)
    h[:m] = -1
    
    # numpy->cvxopt
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    
    sol = np.array(cvxopt.solvers.qp(P=P,q=q,G=G,h=h,options={'show_progress':False})['x']).reshape(-1,)
    
    b = sol[0]
    w = sol[1:n+1]
    ksi = sol[n+1:]
    
    return b, w, ksi

def k_fold_validation_primal(X,Y,k,Cs):
    c = X.shape[0]//k
    
    folds_X = []
    folds_Y = []
    for i in range(k):
        folds_X.append(X[i*c:(i+1)*c,:])
        folds_Y.append(Y[i*c:(i+1)*c,:])
    
    rest = X.shape[0] % k
    if rest != 0:
        folds_X[-1] = np.concatenate((folds_X[-1],X[-rest:,:]),axis = 0)
        folds_Y[-1] = np.concatenate((folds_Y[-1],Y[-rest:,:]),axis = 0)
    

    hinge_losses = []
    for C in Cs:

        hinge_loss = 0
        for i in range(k):
            X_val = folds_X[i]
            Y_val = folds_Y[i]
            
            first = True
            for j in range(k):
                if j != i:
                    if first:
                        X_train = folds_X[j]
                        Y_train = folds_Y[j]
                        
                        first = False
                    else:
                        X_train = np.concatenate((X_train,folds_X[j]),axis = 0)
                        Y_train = np.concatenate((Y_train,folds_Y[j]),axis = 0)
            
            b, w, ksi = solve_primal_problem(X_train, Y_train, C)
            

            hinge_loss += calc_hinge_loss(w,b,X_val,Y_val)
            
        hinge_losses.append(hinge_loss/k)

    plt.plot(Cs,hinge_losses)
    plt.xlabel("$C$")
    plt.ylabel("Sarka gubici")
    plt.xscale('log')
    return Cs[np.argmin(hinge_losses)]

def linear_kernel(X1,X2):
    return X1@X2.T

def polynomial_kernel(X1,X2,c,d):
    return (X1@X2.T+c)**d

def gaussian_kernel(X1,X2,sigma):
    return np.exp(-(X1-X2)@(X1-X2).T/(2*sigma**2))

def kernel(x1,x2,kernel_type, params):
    
    if kernel_type == 'linear':
        return linear_kernel(x1,x2)
    
    elif kernel_type == 'poly':
        if len(params) != 2:
            print("Error: Polynomial kernel needs 2 parameters")
        else:
            c = params[0]
            d = params[1]
            return polynomial_kernel(x1, x2, c, d)
    
    elif kernel_type == 'gaussian':
        if len(params) != 1:
            print("Error: Gaussian kernel needs 1 parameter")
        else:
            sigma = params[0]
            return gaussian_kernel(x1, x2, sigma)
        
    else:
        print("Error: Invalid kernel name")

def dual_bias(support_alpha,support_vectors,support_output,tol,C,kernel_type,params):
    margin_indices = np.where(support_alpha < C)[0]
    
    return support_output[margin_indices[0]] - np.sum(support_alpha*support_output.reshape((-1,))*kernel(support_vectors,support_vectors[margin_indices[0],:],kernel_type,params))

def solve_dual_problem(X,Y,C,kernel_type,params):
    m,n = X.shape
    
    # Equation to maximize
    P = np.outer(Y,Y)*kernel(X,X,kernel_type,params)
    
    q = np.ones(m)
    q = -q
    
    # Constraints
    
    # Equality
    A = Y.astype(np.float64).T
    b = np.zeros(1)
    
    # Inequality
    G = np.vstack((np.eye(m),-np.eye(m)))
    h = np.zeros(2*m)
    h[:m] = C
    
    #numpy->cvxopt
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
        
    alpha = np.array(cvxopt.solvers.qp(P=P,q=q,G=G,h=h,A=A,b=b,options={'show_progress':False})['x']).reshape(-1,)
    
    tol = 1e-6
    support_indices = find_support_indices_dual(alpha, tol)
    
    support_alpha = alpha[support_indices]
    support_vectors = X[support_indices,:]
    support_output = Y[support_indices]
    
    b = dual_bias(support_alpha,support_vectors,support_output,tol,C,kernel_type,params)
    
    return alpha,b

def k_fold_validation_dual(X,Y,k,Cs,kernel_type,params):
    c = X.shape[0]//k
    
    folds_X = []
    folds_Y = []
    for i in range(k):
        folds_X.append(X[i*c:(i+1)*c,:])
        folds_Y.append(Y[i*c:(i+1)*c,:])
    
    rest = X.shape[0] % k
    if rest != 0:
        folds_X[-1] = np.concatenate((folds_X[-1],X[-rest:,:]),axis = 0)
        folds_Y[-1] = np.concatenate((folds_Y[-1],Y[-rest:,:]),axis = 0)
    
    hinge_losses = []
    for C in Cs:
        hinge_loss = 0
        for i in range(k):
            X_val = folds_X[i]
            Y_val = folds_Y[i]
            
            first = True
            for j in range(k):
                if j != i:
                    if first:
                        X_train = folds_X[j]
                        Y_train = folds_Y[j]
                        
                        first = False
                    else:
                        X_train = np.concatenate((X_train,folds_X[j]),axis = 0)
                        Y_train = np.concatenate((Y_train,folds_Y[j]),axis = 0)
            
            alpha, b = solve_dual_problem(X_train, Y_train, C,kernel_type,params)
            
            tol = 1e-6
            support_indices = find_support_indices_dual(alpha, tol)
            
            support_alpha = alpha[support_indices]
            support_vectors = X_train[support_indices,:]
            support_output = Y_train[support_indices]
            
            b = dual_bias(support_alpha,support_vectors,support_output,tol,C,kernel_type,params)
            
            h = np.sum(support_alpha*support_output.reshape((-1,))*kernel(X_val,support_vectors,kernel_type,params),axis=1) + b
            ksi = np.clip(1 - Y_val.reshape((-1,))*h,0,None)
            hinge_loss += np.sum(ksi)
            
        hinge_losses.append(hinge_loss/k)
        
    plt.plot(Cs,hinge_losses)
    plt.xlabel("$C$")
    plt.ylabel("Sarka gubici")
    plt.xscale('log')
    return Cs[np.argmin(hinge_losses)]

def predict_dual(alpha,b,X_train,Y_train,X,kernel_type,params):
    tol = 1e-6
    support_indices = find_support_indices_dual(alpha, tol)
    
    support_alpha = alpha[support_indices]
    support_vectors = X_train[support_indices,:]
    support_output = Y_train[support_indices]
    
    h = np.sum(support_alpha*support_output.reshape((-1,))*kernel(X,support_vectors,kernel_type,params),axis=1) + b
    
    return np.sign(h)
    
def print_accuracy(w,b,X_train,Y_train,X_test,Y_test):
    Y_pred_train = predict_primal(w, b, X_train)
    print(f"Accuracy on Train: {np.sum(Y_pred_train == Y_train)/len(Y_train)*100}")
    
    Y_pred_test = predict_primal(w, b, X_test)
    print(f"Accuracy on Test: {np.sum(Y_pred_test == Y_test)/len(Y_test)*100}")

      
#%% Preprocessing
X, Y = load_data('svmData.csv')
X_train, Y_train, X_test, Y_test = split_train_test(X,Y)

X_test = standardization(X_train, X_test)
X_train = standardization(X_train,X_train)

#%% Primal problem
C_opt = k_fold_validation_primal(X_train, Y_train, k=6, Cs=[0.1,1,10,100,1000,10000])
b_primal, w_primal, ksi_primal = solve_primal_problem(X_train, Y_train, C_opt)
plot_primal(X_train,Y_train,b_primal,w_primal,ksi_primal)

print("PRIMAL")
print_accuracy(w_primal, b_primal, X_train, Y_train, X_test, Y_test)

#%% Dual problem

kernel_type = 'poly'
params = [1,6]

k = 6
Cs = [0.001,0.01,0.1,1,10,100,1000,10000]
C_opt = k_fold_validation_dual(X_train, Y_train, k, Cs,kernel_type,params)
alpha_dual, b_dual = solve_dual_problem(X_train,Y_train,C_opt,kernel_type,params)

print("---------------------------")

print("DUAL")
Y_pred_train = predict_dual(alpha_dual,b_dual,X_train,Y_train,X_train,kernel_type,params)
print(f'Accuracy on Train: {np.sum(Y_pred_train == Y_train.reshape((-1,)))/len(Y_train)*100}')

Y_pred_test = predict_dual(alpha_dual,b_dual,X_train,Y_train,X_test,kernel_type,params)
print(f'Accuracy on Test: {np.sum(Y_pred_test == Y_test.reshape((-1,)))/len(Y_test)*100}')

h = plot_dual(X_train,Y_train,alpha_dual,b_dual,kernel_type,params)


