import numpy as np
import pandas as pd
import pdb
# These are input data
# c, c_prim and d are just parameters which should make our meet the conditions
# A = np.array([[c, -2, -10, 1],
#         [-2, c, 1, 5],
#         [-10, 1, c, -2],
#         [1, 5, -2, c]])


def get_eigenvalues(c, c_prim, d, t):
    A = np.array([[c, 1, 1, 1],
            [1, c, 1, 1],
            [1, 1, c, 1],
            [1, 1, 1, c]])
    n_qmodes = 4


    I = np.eye(2 * n_qmodes)

    # This is X matrix needed for the eq. (1)
    X_top = np.hstack((np.zeros((n_qmodes, n_qmodes)), np.eye(n_qmodes)))
    X_bot = np.hstack((np.eye(n_qmodes), np.zeros((n_qmodes, n_qmodes))))
    X = np.vstack((X_top, X_bot))

    zeros = np.zeros((n_qmodes,n_qmodes))

    # This is  (0, A // A 0) matrix + a constant on a diagonal
    # AFAIR this is invention from Przemek, to make a matrix which is 
    # symmetric, and has the dimension of 2M instead of M
    A_prim = np.vstack((np.hstack((zeros, A)), np.hstack((A, zeros)))) + np.eye(2 * n_qmodes) * c_prim

    # We then apply eq. (3). 
    # We apply d here to make sure some conditions are met.
    cov_matrix = t * (np.linalg.inv(I - X@(d * A_prim)) - I/2)

    # And this is your part of the code for checking if everything is right
    hbar = 2
    O = np.vstack([np.hstack([np.zeros([4, 4]), np.identity(4)]),
                           np.hstack([-np.identity(4), np.zeros([4, 4])])])
    success = 0
    if np.all(np.linalg.eigvals(np.abs(1j*O@(cov_matrix)))/hbar - 0.5 >= 0):
        success = 1
        # print("SUCCESS!", c, c_prim, d, t)
    return np.linalg.eigvals(np.abs(1j*O@(cov_matrix)))/hbar - 0.5, success
    
def main():
    filename = "results_2.csv"
    c = 10
    c_prim = 1
    d = 10
    t = 10
    file = open(filename, 'w+')
    all_eigvals = []
    resolution = 50
    for t in np.linspace(1,100,resolution):
        for d in np.linspace(-10,10,resolution):
            print(t, d, end="\r")
            all_eigvals = []
            for c in np.linspace(-10,10,resolution):
                for c_prim in np.linspace(-10,10,resolution):
                    params = [t,d,c,c_prim]
                    eigenvalues, success = np.real(get_eigenvalues(c, c_prim, d, t))
                    results = list(eigenvalues)
                    final_results = params + results + [success]
                    all_eigvals.append(final_results)
            all_eigvals = np.real(np.array(all_eigvals))
            np.savetxt(file, all_eigvals, fmt='%.2f', delimiter=",")
                    # all_eigvals.append(np.real(eigenvalues))
                    # results = pd.DataFrame(all_eigvals)
                
            # results.round(3).to_csv(filename, index=False,header=False,)



if __name__ == '__main__':
    main()