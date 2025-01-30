#!/usr/bin/python
import numpy as np
import cvxpy as cvx
from scipy.optimize import fsolve
from qcqp import *


def reorder(indices, mode):
    """Reorders the elements. Used for fortan ordering.
    """
    indices = list(indices)
    element = indices.pop(mode)
    return ([element] + indices[::-1])


def tens2vec(tensor):
    """
    Vectoizes a tensor according to the fortran ordering.
    """
    vec_indices = list(range(tensor.ndim - 1, -1, -1))
    return np.transpose(tensor, vec_indices).flatten()


def vec2tens(vec, shape):
    """
    Folds a vector back to a tensor according to the fortran ordering.
    """
    tens = vec.reshape(shape[::-1])
    return np.transpose(tens,list(range(len(shape) - 1, -1, -1)))

def mat2tens(unfolded, shape, mode):
    """Returns the folded tensor of shape `shape` from the `mode`-mode unfolding `unfolded`.
    """
    unfolded_indices = reorder(range(len(shape)), mode)
    original_shape = [shape[i] for i in unfolded_indices]
    unfolded = unfolded.reshape(original_shape)

    folded_indices = list(range(len(shape) - 1, 0, -1))
    folded_indices.insert(mode, 0)
    return np.transpose(unfolded, folded_indices)


def tmprod(tensor, mat, mode):
    """
    Computes the mode-n product of a tensor and a matrix.

    Input: Tensor an n-dimensional tensor
           mat a matrix
    Output: The resulting tensor matrix product.
    """
    if (1 in mat.shape):
        out_shape = list(tensor.shape)
        out_shape[mode] = 1
        result = np.zeros(out_shape)
        # Iterate over each mode-n slice and perform dot product with the vector
        for idx in range(tensor.shape[mode]):
            result = result + np.take(tensor, idx, mode) * mat[idx]
        return result
    else:
        out_n = np.matmul(mat, tens2mat(tensor, mode))
        out_shape = list(tensor.shape)
        out_shape[mode] = mat.shape[0]
        return mat2tens(out_n, out_shape, mode)


def tens2mat(tensor, mode):
    """
    Contracts a tens according to the n-th mode.

    Input: tens of size
           mode is the axis at which the tensor will be contracted
    Output: the tensor matrix product where the ith dimension is replaced by the row dimension of the matrix
    """

    d = tensor.shape
    nd = len(tensor.shape)
    assert mode < nd, "The mode should be less than the dimension of the tensor"

    row_d = d[mode]
    return np.transpose(tensor, reorder(range(tensor.ndim), mode)).reshape((row_d, -1))


def constraint_n_mode_sing_val(m,n):
    """
    Create a block matrix Q that enforces the condition off diags S S^T = 0.
    This will enforce orthogonality of the rows of matrix S.

    Args:
    - m: number of rows of matrix S
    - n: number of columns of matrix S

    Returns:
    - Q: The matrix that represents the D in a quadratic form.
    """
    allQs = []
    # We want to enforce that the rows of S are orthogonal
    # Fill Q with identity blocks to enforce zero inner products for Fortran-order off-diagonals
    for i in range(m):
        Q = np.zeros((int(m * n), int(m * n)))
        for j in range(n):
            # For Fortran ordering, skip by m (row size) for correct block positions
            Q[m * j + i, m * j + i] = 1
        allQs.append(Q)
    return allQs


def get_perm_vec(mode_i, mode_j, tensor_shape):

    """
    Construct the permutation matrix P such that A = PB, where A and B
    are the column-wise vectorizations of the tensor unfolded in mode_i and mode_j, respectively.
    """
    # Calculate the size of the tensor
    size = np.prod(tensor_shape)

    # Initialize the permutation matrix P
    P = np.zeros((size, size), dtype=int)

    # Generate all possible indices for the N-dimensional tensor
    indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1, order='F').T

    for idx in indices:
        # Calculate index_A for unfolding in mode_i
        idx_A = list(idx)
        i = idx_A.pop(mode_i)
        idx_A = [i] + idx_A
        shape_A = [tensor_shape[mode_i]] + list(tensor_shape[:mode_i]) + list(tensor_shape[mode_i + 1:])

        index_A = np.ravel_multi_index(idx_A, shape_A, order='F')

        # Calculate index_B for unfolding in mode_j
        idx_B = list(idx)
        j = idx_B.pop(mode_j)
        idx_B = [j] + idx_B
        shape_B = [tensor_shape[mode_j]] + list(tensor_shape[:mode_j]) + list(tensor_shape[mode_j + 1:])

        index_B = np.ravel_multi_index(idx_B, shape_B, order='F')

        # Set the corresponding entry in P
        P[index_A.astype('int'), index_B.astype('int')] = 1

    return P


def constraint_zero_off_diagonals(m, n):
    """
    Create a block matrix Q that enforces the condition off-diag(S @ S.T) = 0.
    This will enforce orthogonality of the rows of matrix S.

    Args:
    - m: number of rows of matrix S
    - n: number of columns of matrix S

    Returns:
    - Q: The matrix that represents the constraint in a quadratic form.
    """
    allQs = []
    # We want to enforce that the rows of S are orthogonal
    # Fill Q with identity blocks to enforce zero inner products for Fortran-order off-diagonals
    for i in range(m):
        # Create a zero matrix of size (m*n, m*n)
        for j in range(i+1, m):
            Q = np.zeros((m * n, m * n))
            for k in range(n):
                Q[k * m + j, k * m + i] = 1
                Q[k * m + i, k * m + j] = 1
            allQs.append(Q)
    return allQs


def generate_core_tensor(rank, lambdas):
    """
    Generate a core tensor with specified n-mode singular values.

    Parameters:
        shape (tuple): Shape of the tensor (I1, I2, ..., IN).
        lambdas (list of lists): Desired squared singular values for each mode.
            Each element in the list corresponds to the singular values for one mode.

    Returns:
        np.ndarray: Core tensor satisfying the specified n-mode singular values.
    """
    eps = 10**-3
    N = len(rank)
    sum_n_mode_power_same = all(np.sum(lambdas[n])-np.sum(lambdas[0]) < eps for n in range(N))
    if not sum_n_mode_power_same:
        print("The sum of the squared n_mode singular values should be the same")
        return None
    q_non_zero_off_diags = []
    q_n_mode_singular_vals = []
    for n in range(N):
        q_non_zero_off_diags.append(constraint_zero_off_diagonals(rank[n], int(np.prod(rank) / rank[n])))
        q_n_mode_singular_vals.append(constraint_n_mode_sing_val(rank[n], int(np.prod(rank) / rank[n])))

    s = np.prod(rank)
    x = cvx.Variable(s)
    cons_N = [cvx.quad_form(x, np.matmul(
        np.matmul(np.linalg.inv(get_perm_vec(0, n_mode, rank)).T, Q),np.linalg.inv(get_perm_vec(0, n_mode, rank)))) == 0
              for n_mode, Q_i in enumerate(q_non_zero_off_diags)for Q in Q_i] + \
             [(cvx.quad_form(x, np.matmul(np.matmul(np.linalg.inv(get_perm_vec(0, n_mode, rank)).T, QQ),
        np.linalg.inv(get_perm_vec(0, n_mode, rank))))) == lambdas[n_mode][count] for n_mode, Q0_S in
            enumerate(q_n_mode_singular_vals) for count, QQ in enumerate(Q0_S)]


    obj = cvx.Minimize(0)
    prob = cvx.Problem(obj, cons_N)
    qcqp = QCQP(prob)
    qcqp.suggest(SDR)
    f_cd, v_cd = qcqp.improve(ADMM, tol=10**-5, num_iter=10**len(rank))
    print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
    print(x.value)
    tens_s = vec2tens(np.array(x.value), rank)
    return tens_s


def equally_separated_sv(rank, separation, power):
    """
    Creates an equally separated singular values starting from min_sv to the max_sv in each dimension.
    """
    sv = []
    for k in range(len(rank)):
        def func(x):
            tmp = [(x[0] + n*separation) ** 2 for n in range(rank[k])]
            return sum(tmp) - power

        delta_n = fsolve(func, np.array((1)))
        sv.append([(delta_n[0] + n*separation) for n in range(rank[k])])

    return sv

