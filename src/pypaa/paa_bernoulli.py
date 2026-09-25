import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

class BernoulliAA():

    def __init__(self,
                 n_archetypes: int = 3,
                 niter: int = 100,
                 random_seed: int = 42,
                 ):
        self.n_archetypes = n_archetypes
        self.n_iterations = niter
        self.random_seed = random_seed

        self.bool_is_fitted_ = False


    def fit(self, X):

        self.nfeats, self.nsamples = X.shape

        # -- initialise matrices
        Winit, Hinit = self.initialise_matrices()

        # -- normalization 
        eps = 1e-10
        Xmat = X.copy().astype(float)
        Xmat[Xmat <= eps] = eps
        Xmat[Xmat >= 1 - eps] = 1 - eps

        Xmat_1m = 1 - Xmat

        # -- fit the model
        W, H, cost_array = self.optimise_nll(Xmat, Xmat_1m, Winit, Hinit)

        # -- final normalised W, H
        #Wnorm = normalize(W, norm='l1', axis=0)
        #Hnorm = normalize(H, norm='l1', axis=0)
        
        A = Xmat @ W
        #A = Xmat @ Wnorm
    
        self.bool_is_fitted_ = True
        self.A = A

        return W, H, A, cost_array
        # return Wnorm, Hnorm, A, cost_array


    def initialise_matrices(self):
        """
        Initialise matrices
        """
        W = np.random.uniform(size=(self.nsamples,self.n_archetypes))
        H = np.random.uniform(size=(self.n_archetypes,self.nsamples))        

        return W, H


    def compute_bernoulli_cost(self,
                               A: FloatArray,
                               H: FloatArray,
                               Xmat: FloatArray, ) -> float:
        """
        Negative Bernoulli log likelihood
        """
        reconstruction = A @ H

        nll = np.sum( -Xmat * np.log(reconstruction)) + np.sum(-(1-Xmat) * np.log(1-reconstruction))

        return nll




    def optimise_nll(self,
                     Xmat: FloatArray,
                     Xmat_1m: FloatArray,
                     W: FloatArray,
                     H: FloatArray,):

        # normalize W, H
        Wnorm = normalize(W,norm='l1',axis=0)
        Hnorm = normalize(H,norm='l1',axis=0)

        # -- initialise cost array
        cost_array = np.empty(self.n_iterations+1)

        cost_array[0] = self.compute_bernoulli_cost(Xmat @ Wnorm, Hnorm, Xmat)

        for iter in range(self.n_iterations):

            # Update H
            H = H * (
                ( (Xmat @ Wnorm).T @ ( Xmat / (Xmat @ Wnorm @ Hnorm) ) )
                + ( (Xmat_1m @ Wnorm).T @ ( Xmat_1m / (Xmat_1m @ Wnorm @ Hnorm) ) )
            ) / self.nfeats

            Hnorm = normalize(H, norm='l1', axis=0)
            
            # Update W
            term1 = (Xmat @ Wnorm).T @ (Xmat / (Xmat @ Wnorm @ Hnorm) ) @ Hnorm.T
            term2 = (Xmat_1m @ Wnorm).T @ (Xmat_1m / (Xmat_1m @ Wnorm @ Hnorm) ) @ Hnorm.T

            normalization = np.diag(term1) + np.diag(term2)

            numerator = ( 
                Xmat.T @ ( Xmat / (Xmat @ Wnorm @ Hnorm) ) @ Hnorm.T
                + Xmat_1m.T @ ( Xmat_1m / (Xmat_1m @ Wnorm @ Hnorm) ) @ Hnorm.T
            )

            W = W * numerator / normalization[None, :] 
            Wnorm = normalize(W, norm='l1', axis=0)

            cost_array[iter+1] = self.compute_bernoulli_cost(Xmat @ Wnorm, Hnorm, Xmat)

        return Wnorm, Hnorm, cost_array
