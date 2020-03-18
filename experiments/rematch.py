# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import numpy as np
from mpi4py import MPI
import argparse
from sklearn.preprocessing import normalize

from localsimilaritykernel import LocalSimilarityKernel

def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high

class REMatchKernel(LocalSimilarityKernel):
    """Used to compute a global similarity of structures based on the
    regularized-entropy match (REMatch) kernel of local atomic environments in
    the structure. More precisely, returns the similarity kernel K as:

    .. math::
        \DeclareMathOperator*{\\argmax}{argmax}
        K(A, B) &= \mathrm{Tr} \mathbf{P}^\\alpha \mathbf{C}(A, B)

        \mathbf{P}^\\alpha &= \\argmax_{\mathbf{P} \in \mathcal{U}(N, N)} \sum_{ij} P_{ij} (1-C_{ij} +\\alpha \ln P_{ij})

    where the similarity between local atomic environments :math:`C_{ij}` has
    been calculated with the pairwise metric (e.g. linear, gaussian) defined by
    the parameters given in the constructor.

    For reference, see:

    "Comparing molecules and solids across structural and alchemical
    space", Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti,
    Phys.  Chem. Chem. Phys. 18, 13754 (2016),
    https://doi.org/10.1039/c6cp00415f
    """
    def __init__(self, alpha=0.1, threshold=1e-6, metric="linear", gamma=None, degree=3, coef0=1, kernel_params=None, normalize_kernel=True):
        """
        Args:
            alpha(float): Parameter controlling the entropic penalty. Values
                close to zero approach the best-match solution and values
                towards infinity approach the average kernel.
            threshold(float): Convergence threshold used in the
                Sinkhorn-algorithm.
            metric(string or callable): The pairwise metric used for
                calculating the local similarity. Accepts any of the sklearn
                pairwise metric strings (e.g. "linear", "rbf", "laplacian",
                "polynomial") or a custom callable. A callable should accept
                two arguments and the keyword arguments passed to this object
                as kernel_params, and should return a floating point number.
            gamma(float): Gamma parameter for the RBF, laplacian, polynomial,
                exponential chi2 and sigmoid kernels. Interpretation of the
                default value is left to the kernel; see the documentation for
                sklearn.metrics.pairwise. Ignored by other kernels.
            degree(float): Degree of the polynomial kernel. Ignored by other
                kernels.
            coef0(float): Zero coefficient for polynomial and sigmoid kernels.
                Ignored by other kernels.
            kernel_params(mapping of string to any): Additional parameters
                (keyword arguments) for kernel function passed as callable
                object.
            normalize_kernel(boolean): Whether to normalize the final global
                similarity kernel. The normalization is achieved by dividing each
                kernel element :math:`K_{ij}` with the factor
                :math:`\sqrt{K_{ii}K_{jj}}`
        """
        self.alpha = alpha
        self.threshold = threshold
        super().__init__(metric, gamma, degree, coef0, kernel_params, normalize_kernel)

    def get_global_similarity(self, localkernel):
        """
        Computes the REMatch similarity between two structures A and B.

        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: REMatch similarity between the structures A and B.
        """
        n, m = localkernel.shape
        if self.metric=='linear':
           localkernel = localkernel**2/(n*m)
           return(np.sum(localkernel))
          
          
        K = np.exp(-(1 - localkernel) / self.alpha)

        # initialisation
        u = np.ones((n,)) / n
        v = np.ones((m,)) / m

        en = np.ones((n,)) / float(n)
        em = np.ones((m,)) / float(m)

        # converge balancing vectors u and v
        itercount = 0
        error = 1
        while (error > self.threshold):
            uprev = u
            vprev = v
            v = np.divide(em, np.dot(K.T, u))
            u = np.divide(en, np.dot(K, v))

            # determine error every now and then
            if itercount % 5:
                error = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
            itercount += 1

        # using Tr(X.T Y) = Sum[ij](Xij * Yij)
        # P.T * C
        # P_ij = u_i * v_j * K_ij
        pity = np.multiply( np.multiply(K, u.reshape((-1, 1))), v)

        glosim = np.sum( np.multiply( pity, localkernel))

        return glosim

def main(src_text, src, dir, kern, n_best, out_txt):
     mpi_comm = MPI.COMM_WORLD
     mpi_rank = mpi_comm.Get_rank()
     mpi_size = mpi_comm.Get_size()

     src = np.load(src, allow_pickle=True)
     #src = src[0:6]
     src = normalize(src)
     src = [src] 
     h_list = np.load(dir+'h_arrays_full.npy',allow_pickle=True)

     dat_size = len(h_list)

     my_border_low, my_border_high = return_borders(mpi_rank, dat_size, mpi_size)

     h_list = h_list[my_border_low:my_border_high]
     h_list = [normalize(i) for i in h_list]

     my_len = len(h_list)
      
     if kern=='rematch':
        rem = REMatchKernel(alpha=1e-3, threshold=1e-5, metric='polynomial', degree=2, gamma=1, coef0=0, normalize_kernel=True)
     elif kern=='average':
          rem = REMatchKernel(metric='linear')

     K = np.empty((my_len,1), dtype=np.float64)
 
     sendcounts = np.array(mpi_comm.gather(my_len,root=0))
     
     K += rem.create(h_list, src) #Generate rematch similarities

     if mpi_rank==0:
        K_full = np.empty((dat_size,1),dtype=np.float64)
        #print("memory usage(bytes): {}".format(K.nbytes+K_full.nbytes))
     else:
        K_full = None
     
     #Gather rows of similarity vector
     mpi_comm.Gatherv(sendbuf=K,recvbuf = (K_full, sendcounts),root=0)
     
     if mpi_rank==0:
        print('Kernel used: '+kern)
  
        K = K_full

        #filter NaNs
        n_nans = np.count_nonzero(np.isnan(K))

        max_indices = np.argpartition(K.flatten(),-n_best-n_nans)[-n_best-n_nans:-n_nans]        
                
        print('Similarities for {} most similar reactions: {}'.format(len(max_indices),K[max_indices]))
        #src_file = open(src_text,'r')
        #lines = src_file.readlines()

        #print(lines[max_indices])
        for index in max_indices:
            cmd='sed "'+str(index)+'q;d" '+src_text+' >> '+out_txt
            os.system(cmd)   
        
        cmd = 'sed -i "s/ //g" '+out_txt
        os.system(cmd)

        print(K)
        print('Max similarity: {} which is index {}'.format(np.nanmax(K), np.nanargmax(K)))
        print('Min similarity: {}, which is index {}'.format(np.nanmin(K), np.nanargmin(K)))
        print('No. of NaNs: {}'.format(np.count_nonzero(np.isnan(K))))
     mpi_comm.Barrier()
     MPI.Finalize()

if __name__ == "__main__":

     parser = argparse.ArgumentParser()
     parser.add_argument('-src_txt', type=str, 
                         help='Path to .txt file of training set reactions.')
     parser.add_argument('-out_txt', type=str, 
                         help='Path to .txt file which will contain the n_best detokenized reactions.')
     parser.add_argument('-src', type=str, 
                         help='Path to .npy file of source hidden states to compare with the training set.')
     parser.add_argument('-dir', type=str, 
                         help='Directory where the hidden state numpy arrays are stored.')
     parser.add_argument('-kernel', type=str, 
                         help='Type of kernel used to calculate the similarity between hidden state vectors')
     parser.add_argument('-n_best', type=int, default=5, 
                         help='Number of most similar training reactions to return.')
     args = parser.parse_args()

     main(args.src_txt, args.src, args.dir, args.kernel, args.n_best, args.out_txt)
