import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]

class PoissonAA():

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
        Xscale = np.max(X, axis=1, keepdims=True)
        Xmat = X.copy().astype(float) / Xscale
        Xmat = Xmat + 1e-12 # -- offset for stability
        print(Xmat.shape)

        # -- set penalty
        penalty = 20 * Xmat.max()

        # -- fit the model
        W, H, cost_array = self.optimise_nll(Xmat, Winit, Hinit, penalty)
        
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


    def compute_poisson_cost(self,
        A: FloatArray,
        H: FloatArray,
        X: FloatArray
    ) -> float:
        """
        Negative Poisson log likelihood
        """
        reconstruction = A @ H

        nll = np.sum( - X * np.log(reconstruction) + reconstruction )

        return nll


    def optimise_nll(self,
                     Xmat: FloatArray,
                     W: FloatArray,
                     H: FloatArray,
                     penalty: float):
        """
        Function to optimise the negative log likelihood
        """

        cost_array = np.empty(self.n_iterations+1)
        cost_array[0] = self.compute_poisson_cost( Xmat @ W, H, Xmat)

        for iter in range(self.n_iterations):

            # -- update H
            gradPosH = ( np.sum(Xmat, axis=0) @ W )[:, None] + penalty

            gradNegH = W.T @ Xmat.T \
                     @ ( Xmat / (Xmat @ W @ H) ) \
                     + penalty / np.sum(H, axis=0)[None, :]

            H = H * gradNegH / gradPosH

            # -- update W
            gradPosW = ( np.sum(Xmat.T, axis=1)[:, None] @ np.sum(H.T, axis=0)[None, :] ) + penalty

            gradNegW = ( Xmat.T @ ( Xmat / (Xmat @ W @ H) ) \
                            @ H.T + penalty / np.sum(W, axis=0)[None, :] )

            W = W * gradNegW / gradPosW

            # -- compute cost
            cost_array[iter+1] = self.compute_poisson_cost( Xmat @ W, H, Xmat)

        return W, H, cost_array
