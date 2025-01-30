from generate_core_svs import *

size = (4, 5, 6, 7)
rank = (3, 4, 3, 6)
N = len(rank)
#  The separation between the mode-n singular values
separation = 1
#  The total power of the tensor, i.e. squared sum of each mode-n singular values for each mode
total_power = 100
#  Function that generates the singular values that are separted by separation, while keeping the power fixed. The
#  minimum and the maximum singular value changes accordingly.
sv = equally_separated_sv(rank, separation, total_power)
print('The requested singular values are ' + str(sv))
sv_squared = []
for i in range(N):
    sv_squared.append(np.array(sv[i]) ** 2)
#
core = generate_core_tensor(rank, sv_squared)
lambdas_est = []
for i in range(N):
    lambdas_est.append(np.sqrt(np.diag(np.matmul(tens2mat(core, i), tens2mat(core, i).T))))
print('The estimated singular values are ' + str(lambdas_est))

err = np.array(sv) - np.array(lambdas_est)
print("The squared error between the estimated and true singular values is " +
      str(np.sum([np.sum(i)**2 for i in zip(*err)])))

