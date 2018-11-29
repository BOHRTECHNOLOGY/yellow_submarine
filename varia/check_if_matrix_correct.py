import numpy as np
import pdb

for c in np.linspace(-10,10,100):#[0,1,2,3,4]:

    A1 = np.array([[c, -2, -10, 1],
        [-2, c, 1, 5],
        [-10, 1, c, -2],
        [1, 5, -2, c]])
    
    A2 = np.array([[c, 1, 1, 1],
        [1, c, 1, 1],
        [1, 1, c, 1],
        [1, 1, 1, c]])
    
    A3 = np.array([[c,3,0,-5],
              [3,c,-5,8],
              [0,-5,c,0],
              [-5,8,0,c]])
    for A in [A1, A2, A3]:
        for d in np.linspace(0.01, 1, 100):#[0.05, 0.1,0.2,0.8, 1]:
            for c_prim in np.linspace(-10,10,100):#[0,1,2,3,4]:
                print(c,d,c_prim,end="\r")
                c = 1
                d = 0.05
                c_prim = 1
                
                
                def is_semi_pos_def(M):
                    return np.all(np.linalg.eigvals(M) >= 0)
                
                def is_symplectic_semi_pos_def(M):
                    hbar = 2
                    O = np.vstack([np.hstack([np.zeros([4, 4]), np.identity(4)]),
                                           np.hstack([-np.identity(4), np.zeros([4, 4])])])
                    # pdb.set_trace()
                    return np.all(np.linalg.eigvals(np.abs(1j*O@(M)))/hbar - 0.5 >= 0)

                n_qmodes = 4
                
                I = np.eye(2 * n_qmodes)
                X_top = np.hstack((np.zeros((n_qmodes, n_qmodes)), np.eye(n_qmodes)))
                X_bot = np.hstack((np.eye(n_qmodes), np.zeros((n_qmodes, n_qmodes))))
                X = np.vstack((X_top, X_bot))
                
                zeros = np.zeros((n_qmodes,n_qmodes))
                A_prim = np.vstack((np.hstack((zeros, A)), np.hstack((A, zeros)))) + np.eye(2 * n_qmodes) * c_prim
                try:
                    cov_matrix = np.linalg.inv(I - X@(d * A_prim)) - I/2
                    cond_1 = is_symplectic_semi_pos_def(cov_matrix)
                    cond_2 = is_semi_pos_def(cov_matrix)
                    if cond_1 and cond_2:
                        print(A, c, d, c_prim)

                except Exception as e:
                    # print(A, c, d, c_prim)
                    print(e)
                
                
                
                pdb.set_trace()
                    