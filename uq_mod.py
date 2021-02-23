import time
import os
import numpy as np


def safe_mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def dot_product(X, Y):
    return np.dot(X, Y)


def apply_lbfgs_loop(q, S, Y, precond0, LBFGS_mem):
    kk = LBFGS_mem
    rh = np.zeros(kk)
    al = np.zeros(kk)
    for ii in range(kk):
        rh[ii] = 1 / dot_product(Y[:, ii], S[:, ii])
        al[ii] = rh[ii] * dot_product(S[:, ii], q)
        q = q - al[ii] * Y[:, ii]

    r = q * precond0

    for ii in range(kk-1, -1, -1):
        be = rh[ii] * dot_product(Y[:, ii], r)
        r = r + S[:, ii] * (al[ii] - be)

    return r


def load_sample_precond_kernel():
    return np.load("./example/LBFGS/precond0.npy")


def load_sample_S_Y(m, LBFGS_mem):
    # NOTE: S and Y are store in reverse order.
    # S[:, 0] are the last iteration and S[-1] are the first one.
    S = np.memmap('example/LBFGS/S', mode='r', dtype='float32', shape=(m, LBFGS_mem))
    Y = np.memmap('example/LBFGS/Y', mode='r', dtype='float32', shape=(m, LBFGS_mem))
    return S, Y


def load_example_matrix(m, LBFGS_mem):
    print("Load input data from example (From Qianchen Liu)")
    p = load_sample_precond_kernel()
    S, Y = load_sample_S_Y(m, LBFGS_mem)
    return p, S, Y


def load_random_matrix(m, LBFGS_mem):
    print("Load input data from random matrix")
    return np.random.randn(m).astype('f'), \
        np.random.randn(m, LBFGS_mem).astype('f'), \
        np.random.randn(m, LBFGS_mem).astype('f')


def random_probing(m, LBFGS_mem, LBFGS_mem_used):
    """
    Most memory intensive function, save to save S & Y matrix
    and R & r matrix

    In the future, could be optimized by computing
    one column of R and r at one time.
    """
    R = np.random.randn(m, LBFGS_mem_used).astype('f')
    r = np.zeros((m, LBFGS_mem_used)).astype('f')

    if m == 135000:
        precond0, S, Y = load_example_matrix(m, LBFGS_mem)
    else:
        precond0, S, Y = load_random_matrix(m, LBFGS_mem)

    print("\tprecond 0: ", precond0.shape, precond0.dtype)
    print("\tS: ", S.shape, S.dtype, " | Y: ", Y.shape, Y.dtype)

    for ii in range(LBFGS_mem_used):
        if (ii+1) % 10 == 0 or ii == LBFGS_mem_used - 1:
            print(f"\t[{ii+1}/{LBFGS_mem_used}] Apply l-bfgs "
                  "loop for random variable")
        r[:, ii] = apply_lbfgs_loop(R[:, ii], S, Y, precond0, LBFGS_mem_used)
        r[:, ii] = r[:, ii] - R[:, ii] * precond0
    return R, r


def save_uq_map(UQ_map, m, outputdir):
    safe_mkdir('./UQ_map')

    file_vp = os.path.join(outputdir, 'proc000000_vp.bin')
    print(f"Save Vp uq map: {file_vp}")
    UQ_map[0:int(m/2)].real.astype('float32').tofile(file_vp)

    file_vs = os.path.join(outputdir, 'proc000000_vs.bin')
    print(f"Save Vs uq map: {file_vs}")
    UQ_map[int(m/2):].real.astype('float32').tofile(file_vs)


def svd_part(Zt, Wt, outputdir):
    # Linear Solver
    B = np.linalg.solve(Zt, Wt)
    Bt = 0.5 * (B + B.transpose())

    # SVD
    print("\tBt: ", Bt.shape, Bt.dtype)
    U, S, _ = np.linalg.svd(Bt, full_matrices=False)

    # Eigenvalues
    norm_factor = 4.957558e+04
    S = S / norm_factor
    S_t = np.diag(S)

    outputfn = os.path.join(outputdir, "Eigen_values.txt")
    print(f"\tEigen values saved to: {outputfn}")
    np.savetxt(outputfn, S)

    return U, S_t


def construct_uq_map(U_new, U_t):
    # Construct uncertainty map
    m = U_new.shape[0]
    LBFGS_mem_used = U_new.shape[1]

    UQ_map = np.zeros(m, dtype='f')

    #for ii in range(m):
    #    UQ_map[ii] = np.inner(U_new[ii, :], U_t[ii, :])

    for j in range(LBFGS_mem_used):
        UQ_map += U_new[:, j] * U_t[:, j]

    UQ_map = np.sqrt(UQ_map)

    return UQ_map


def print_timestamp(tag, t0):
    dt = time.time() - t0
    print("*" * 20 + "\n{} [ t = {:.2f} sec ]\n".format(tag, dt) + "*" * 20)


def uq_analysis(m, LBFGS_mem, LBFGS_mem_used, outputdir):
    t0 = time.time()

    # ###############################
    # Random Probing
    print_timestamp("Random probing", t0)
    # dim of R and r: (m, LBFGS_mem_used)
    R, r = random_probing(m, LBFGS_mem, LBFGS_mem_used)
    print("\tR : ", R.shape, R.dtype, " | r: ", r.shape, r.dtype)

    # ###############################
    # QR
    # dimension of Q (m, n) (n is LBFGS_mem_used)
    Q, _ = np.linalg.qr(r)
    print("\tQ: ", Q.shape, Q.dtype)

    # dimension of Zt and Wt (n, n)
    Zt = np.dot(R.transpose(), Q)
    Wt = np.dot(r.transpose(), Q)

    # free memory
    R = None
    r = None
    time.sleep(5)

    print_timestamp("SVD", t0)
    U, S_t = svd_part(Zt, Wt, outputdir)

    print_timestamp("UQ-map", t0)
    # U_new: (m, n) | U_t: (m, n)
    U_new = np.dot(Q, U)
    U_t = np.dot(U_new, S_t)
    print("\tU_new: ", U_new.shape, U_new.dtype, " | U_t: ", U_t.shape, U_t.dtype)

    # free memory
    Q = None
    time.sleep(5)

    UQ_map = construct_uq_map(U_new, U_t)
    print("\tUQ_map: ", UQ_map.shape, UQ_map.dtype)

    # ###########################################
    # save the UQ_map to disk
    print_timestamp("Save UQ map", t0)
    save_uq_map(UQ_map, m, outputdir)

    print_timestamp("Done", t0)


def main():
    m = 135000  # model size
    LBFGS_mem = 100
    LBFGS_mem_used = 49
    outputdir = "./output.example"

    #m = 1 * 10 ** 8
    #m = 503808000 * 4
    #LBFGS_mem = 5
    #LBFGS_mem_used = 5

    safe_mkdir(outputdir)

    uq_analysis(m, LBFGS_mem, LBFGS_mem_used, outputdir)


if __name__ == "__main__":
    main()
