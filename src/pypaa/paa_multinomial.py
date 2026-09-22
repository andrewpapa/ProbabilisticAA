import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]

class MultinomialAA():

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
        Xcounts = X.copy().astype(float)
        Xmat = X.astype(float).copy()
        Xmat = normalize(Xmat, norm='l1', axis=0)

        # -- set penalty
        eps = 1e-12

        # -- fit the model
        W, H, cost_array = self.optimise_nll(Xmat, Xcounts, Winit, Hinit, eps)
        
        A = Xmat @ W
    
        self.bool_is_fitted_ = True
        self.A = A

        return W, H, A, cost_array


    def initialise_matrices(self):
        """
        Initialise matrices
        """
        W = np.random.uniform(size=(self.nsamples,self.n_archetypes))
        W = normalize(W,norm='l1',axis=0)

        H = np.random.uniform(size=(self.n_archetypes,self.nsamples))
        H = normalize(H,norm='l1',axis=0)

        return W, H


    def compute_multinomial_cost(self,
                                 A: FloatArray,
                                 H: FloatArray,
                                 X: FloatArray,
                                 eps: float = 1e-16,
    ) -> float:
        """
        Negative Multinomial log likelihood
        """
        reconstruction = A @ H + eps

        return np.sum( -X * np.log(reconstruction))


    def optimise_nll(self,
                     Xmat: FloatArray,
                     Xcounts: FloatArray,
                     W: FloatArray,
                     H: FloatArray,
                     eps: float):

        cost_array = np.empty(self.n_iterations+1)
        cost_array[0] = self.compute_multinomial_cost(Xmat @ W, H, Xcounts, eps) ## HERE

        for iter in range(self.n_iterations):

            # -- expectation
            temp = Xcounts / (Xmat @ W @ H)
            WNew = (eps + Xmat.T @ temp @ H.T) * W
            HNew = (eps + W.T @ Xmat.T @ temp) * H

            # -- maximisation
            W = normalize(W, norm='l1', axis=0)
            H = normalize(H, norm='l1', axis=0)
            # W = WNew / np.sum(WNew, axis=0, keepdims=True)
            # H = HNew / np.sum(HNew, axis=0, keepdims=True)

            # -- compute cost
            cost_array[iter+1] = self.compute_multinomial_cost( Xmat @ W, H, Xcounts, eps)


        return W, H, cost_array
