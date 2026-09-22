import numpy as np
import pandas as pd
from scipy.special import gammaln, psi
from sklearn.preprocessing import normalize
from scipy.optimize import nnls
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

class NormalAA():

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
        Xmat = X.astype(float).copy()
        Xmat = normalize(Xmat, norm='l1', axis=0)

        # -- set penalty
        penalty = 1000

        # -- fit the model
        W, H, cost_array = self.optimise_ll(Xmat,Winit,Hinit,penalty)
        
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


    def compute_gaussian_cost(
        self,
        W: FloatArray,
        H: FloatArray,
        Xmat: FloatArray,
        A: FloatArray,
        alpha_0: float,
        beta_0: float,
    ) -> float:
        
        # Posterior parameters
        alpha_1, beta_1, alpha_2, beta_2 =  self.compute_alphas_betas(W, H, Xmat, A, alpha_0,beta_0)

        val = (
            -alpha_1
            -alpha_2
            + Xmat.size * self.log_gamma(alpha_1, beta_1) / 2.0
            + A.size * self.log_gamma(alpha_2, beta_2) / 2.0
            + self.ent_gamma(alpha_1, beta_1)
            + self.ent_gamma(alpha_2, beta_2)
        )

        return float(val)


    def ent_gamma(self, alpha: float, beta: float) -> float:
        return (
            -np.log(beta)
            + alpha
            + gammaln(alpha)
            + (1.0 - alpha) * psi(alpha)
        )


    def log_gamma(self, alpha: float, beta: float) -> float:
        return psi(alpha) - np.log(beta)
    

    def compute_alphas_betas(
        self,
        W: FloatArray,
        H: FloatArray,
        Xmat: FloatArray,
        A: FloatArray,
        alpha_0: float,
        beta_0: float) -> FloatArray:

        alpha_1 = alpha_0 + Xmat.size / 2.0
        beta_1 = beta_0 + np.linalg.norm( Xmat - A @ H, ord="fro", ) ** 2 / 2.0 

        alpha_2 = alpha_0 + A.size / 2.0
        beta_2 = beta_0 + np.linalg.norm( A - Xmat @ W, ord="fro", ) ** 2 / 2.0 

        return np.array([alpha_1, beta_1, alpha_2, beta_2])


    def update_factors(
        self,
        Xmat: FloatArray,
        A: FloatArray,
        W: FloatArray,
        H: FloatArray,
        penalty: float,
    ) -> tuple[FloatArray, FloatArray]:

        n_sam, n_lat = W.shape

        # Update W
        C = np.vstack([ Xmat, penalty * np.ones((1, n_sam)), ])

        for count_col in range(n_lat):
            d = np.concatenate([ A[:, count_col], [penalty], ])

            W[:, count_col], _ = nnls(C, d)

        # Update H
        C = np.vstack([ A, penalty * np.ones((1, n_lat)), ])

        for count_col in range(n_sam):
            d = np.concatenate([ Xmat[:, count_col], [penalty], ])

            H[:, count_col], _ = nnls(C, d)

        return W, H


    def optimise_ll(self,
                    Xmat: FloatArray,
                    W: FloatArray,
                    H: FloatArray,
                    penalty: float):

        alpha_0 = 1.
        beta_0 = 1.

        A = Xmat @ W

        # -- initialise cost array
        cost_array = np.empty(self.n_iterations+1)
        cost_array[0] = self.compute_gaussian_cost(W,H,Xmat,A,alpha_0, beta_0)

        for iter in range(self.n_iterations):

            W, H = self.update_factors(Xmat,A,W,H,penalty,)

            T = self.compute_gaussian_cost(W,H,Xmat,A,alpha_0, beta_0)
            
            alpha_1, alpha_2, beta_1, beta_2 =  self.compute_alphas_betas(W,H,Xmat,A,alpha_0,beta_0)
            
            regS = (alpha_2 / beta_2) / (alpha_1 / beta_1)

            temp = np.linalg.solve(H @ H.T + regS * np.eye(self.n_archetypes), 
                                            H + regS * W.T,).T
            A = Xmat @ temp
                
            cost_array[iter+1] = self.compute_gaussian_cost(W,H,Xmat,A,alpha_0, beta_0)
            if cost_array[iter+1] < T:
                print('Error: cost decreased')

        return W, H, cost_array
