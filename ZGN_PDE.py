import numpy as np
import itertools
import operator

##################################################################################
# Functions for differentiation
##################################################################################
def SpectralDiff(u, d, axis=0, order=1):
    """
    Takes spectral derivative

    Input:
    u = data to be differentiated
    d = Grid spacing.  Assumes uniform spacing
    axis = differentiation axis 
    order = differentiation order
    """

    q=np.fft.fft(u,axis=axis)
    n=u.shape[axis]
    dims=np.ones(len(u.shape),dtype=int)
    dims[axis]=n
    freqs=np.zeros(n,dtype=np.complex128)
    positives=np.arange(int(n/2+1))
    negatives=np.setdiff1d(np.arange(n),positives)
    freqs[:int(n/2+1)]=(positives)*2*np.pi/(n*d)
    freqs[int(n/2+1):]=(negatives-n)*2*np.pi/(n*d)

    return np.fft.ifft(np.reshape(1j*freqs,dims)**order*q,axis=axis)

def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3

    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """

    n = u.size
    ux = np.zeros(n, dtype=np.complex64)

    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)

        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux

    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2

        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux

    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3

        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux

    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)


def PolyDiff(u, x, deg = 3, diff = 1, width = 5):

    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        points = np.arange(j - width, j + width)

        # Fit to a Chebyshev polynomial
        # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])

    return du


##################################################################################
# Functions specific to PDE-FIND
##################################################################################

def build_Theta(data, derivatives, derivatives_description, P, data_description = None):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
        derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        derivatives_description: description of what derivatives have been passed in
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """

    n,d = data.shape
    m, d2 = derivatives.shape
    if n != m: raise Exception('dimension error')
    if data_description is not None:
        if len(data_description) != d: raise Exception('data descrption error')

    Theta = np.ones((n,0), dtype=np.complex64)
    descr = []
    
    # Create a list of all multiindices for d variables up to degree P
    indices=()
    for i in range(0,d):
        indices=indices+(np.arange(P+1),)

    multiindices=[]
    for x in itertools.product(*indices):
        current=np.array(x)
        if(np.sum(x)<=P):
            multiindices.append(current)
    multiindices=np.array(multiindices)

    for D in range(derivatives.shape[1]):
        for multiindex in multiindices:
            new_column=derivatives[:,D]*np.prod(data**multiindex,axis=1)
            Theta = np.hstack([Theta, new_column[:,np.newaxis]])
            if data_description is None: descr.append(str(multiindex) + derivatives_description[D])
            else:
                function_description = ''
                for j in range(d):
                    if multiindex[j] != 0:
                        if multiindex[j] == 1:
                            function_description = function_description + data_description[j]
                        else:
                            function_description = function_description + data_description[j] + '^' + str(multiindex[j])
                descr.append(function_description + derivatives_description[D])

    return Theta, descr


def build_Theta2(features, P, feature_descriptions = None):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables
    
    input:
        features: features to combine
        P: max power of polynomial functions
        feature_descriptions: names of features

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """

    n,d = features.shape

    if feature_descriptions is not None:
        if len(feature_descriptions) != d: raise Exception('features descrption error')

    Theta = np.ones((n,0), dtype=np.complex64)
    descr = []
    
    # Create a list of all multiindices for d variables up to degree P
    indices=()
    for i in range(0,d):
        indices=indices+(np.arange(P+1),)

    multiindices=[]
    for x in itertools.product(*indices):
        current=np.array(x)
        if(np.sum(x)<=P):
            multiindices.append(current)
    multiindices=np.array(multiindices)

    for multiindex in multiindices:
        new_column=np.prod(features**multiindex,axis=1)
        Theta = np.hstack([Theta, new_column[:,np.newaxis]])
        if feature_descriptions is None: descr.append(str(multiindex))
        else:
            function_description = ''
            for j in range(d):
                if multiindex[j] != 0:
                    if multiindex[j] == 1:
                        function_description = function_description + feature_descriptions[j]
                    else:
                        function_description = function_description + feature_descriptions[j] + '^' + str(multiindex[j])
            descr.append(function_description)

    return Theta, descr


def print_pde(w, rhs_description, ut = 'u_t'):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    print(pde)


##################################################################################
# Functions for sparse regression.
##################################################################################

def STRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg=1.0/(np.linalg.norm(X0,normalize,axis=0))[:,np.newaxis]
        X=Mreg.T*X0
    else: X = X0

    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=None)[0]
    else: w = np.linalg.lstsq(X,y,rcond=None)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]

    # Threshold and continue
    for j in range(maxit):
        if print_results: print("iter, terms: %i %i"%(j,num_relevant))
        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        # new_biginds = [i for i in range(d) if i not in smallinds]
        new_biginds = np.setdiff1d(np.arange(len(w)),smallinds)

        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                if print_results: print("Tolerance too high - all coefficients set below tolerance")
                return w
            else: break
        biginds = new_biginds

        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y), rcond=None)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]

    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
