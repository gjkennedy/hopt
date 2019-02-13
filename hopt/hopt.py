'''
This code uses an interior point method to perform gradient-based
optimization with Hessian-vector products. Inequality constraints and
bound constraints can be imposed on the optimization problem. An
approximate Hessian is formulated using a standard damped L-BFGS
update scheme. The KKT system is solved using either a quasi-Newton or
an inexact Newton scheme. The goal is to achieve robustness and a
superlinear asymptotic rate of convergence.

One of the principal difficulties with these types of schemes is
globalization. We use a line search method with a backtracking method
that satisifies the sufficient decrease conditions and possibly the
strong Wolfe conditions. Ensuring a descent direction with the full
Newton solve is not possible. Instead, we maintain a positive definite
Hessian approximation and resort to the quasi-Newton method if
negative curvature is detected (a non-monotone approach that is more
lenient would be better.)  Standard methods, like inertia tracking,
are not possible since we cannot compute the full Hessian. All line
searches utilize a technique that attempts to satisfy the strong Wolfe
conditions while retaining a fractional-decrease to the boundary
rule. If the strong Wolfe conditions cannot be satisfied, the
sufficient decrease conditions are imposed.

The perturbed KKT conditions
----------------------------

The optimization problem is formulated as follows:

min  f(x)
w.r.t.  x
st.  lb <= x <= ub
and  c(x) >= 0

The perturbed KKT conditions for this problem are:

g(x) + A^{T}(x)*z - zl + zu = 0
c(x) - s = 0
S*z - mu*e = 0
(X - Xl)*zl - mu*e = 0
(Xu - X)*zu - mu*e = 0

where s are slack variables and z, zl, and zu are Lagrange multipliers
for the general, lower bound and upper bound constraints,
respectively. In addition, c(x) are the general constraints, A(x) is
the constraint Jacobian. 

We solve the KKT system using a combination of full- and quasi- Newton
steps as follows:

K_k*p_k = - r_k

where K_k is a linearization of the KKT system, p_k is the step
direction and r_k are the residuals of the perturbed KKT residuals. At
each step, we use either a Hessian estimate based on the compact
L-BFGS representation or an inexact solution to the above system
obtained with a Krylov method. When using the L-BFGS formula alone, we
obtain an exact solution to the above equations using the
Sherman-Morrison-Woodbury formula. This is possible due to the compact
L-BFGS representation.

Once the KKT residuals are reduced sufficiently, we switch from a
quasi-Newton method to a full Newton-method with (nearly) exact
Lagrangian Hessian-vector products. At this point the Newton system is
solved inexactly in the following sense:

||r_k + K_k*p_k|| <= eta_k*||r_k||

where eta_k is the forcing term in an inexact Newton method. Here
we select eta_k as follows:

eta_{k} = eta_{0}*min(1, sqrt(||r_{k}||/||r_{0}||))

where r_{0} is the KKT residual from the beginning of the most-recent
barrier update and r_{k} is the KKT residual at the k-th update since
the barrier update. (Strictly speaking this will interfere with the
superlinear rate of convergence.... so there! Note that if we instead
use r_{0} and r_{k} based on the initial/current residual, we restore
the superlinear rate of convergence. In practice, we don't use
sufficiently large Krylov subspaces which also interferes with
superlinear convergence anyways.)

KKT solution technique
----------------------

At each iteration, we obtain an approximate solution of the 
following equations:

[  B   -A^{T}  0  -I         I        ][ px  ]
[  A    0     -I   0         0        ][ pz  ] = -R
[  0    S      Z   0         0        ][ ps  ]
[  Zl   0      0   (X - Xl)  0        ][ pzl ]
[ -Zu   0      0   0         (Xu - X) ][ pzu ]

where B is either a quasi-Newton Hessian approximation or a Hessian
operator with access provided via Hessian-vector products. We solve
this positive semi-definite system using FGMRES since the
preconditioner may be flexible.

In the case of the inexact Newton step computations, we utilize a
preconditioner formed either specifically for this purpose or an
L-BFGS approximation of B. Both these formula are assumed to take
the form:

B = B0 - Z*M*Z^{T}

For efficiency the dimension of the matrix M should be much less than
the dimension of the matrix B. Furthermore, B0 is a diagonal matrix
that is a scalar multiple of the identity matrix for the case of the
BFGS scheme.

Damped L-BFGS update:
---------------------

The update to the BFGS formula is computed using a damped update. This
is directly from Nocedal and Wright. The BFGS implementation is based
on the compact representation presented originally by Byrd, Nocedal
and Schnabel.

SR-1 update:
------------

If the output from the Hessian vector product is y = H*s,
then the update to B^{-1} is as follows:

p = (s - B^{-1}*y)
alpha = p^{T}*y

B^{-1} <- B^{-1} + p*p^{T}/alpha

Line searching and all that jazz:
---------------------------------

Once we have a new step (px, pz, ps), we use a line search based on
the exact merit function:

m(x, s) = f(x) + rho*||(c(x) - s)||_{2}

where rho is a vector of penalty parameters chosen to ensure a
sufficiently negative descent direction. 
'''

'''
Use dummy mpi class if mpi4py is not present
'''
try:
    from mpi4py import MPI
except:
    class mpi_dummy:
        def __init__(self):
            self.COMM_WORLD = self
            self.rank = 0
            self.size = 1
            return
        def bcast(self, *args, **kwargs):
            return
        def Bcast(self, *args, **kwargs):
            return
        def Wtime(self):
            return 0.0

    MPI = mpi_dummy()

import numpy as np
import scipy
from scipy import sparse, linalg
import scipy.sparse.linalg as sci_alg

class DenseBFGS:
    def __init__(self, n):
        '''
        Create a dense BFGS matrix
        '''

        self.initialized = False
        self.B = np.zeros((n, n))

        return

    def update(self, s, y):
        '''
        Perform the BFGS update on a dense matrix
        '''

        # Compute alpha = s^{T}*y
        alpha = np.dot(s, y)

        # Initialize the matrix
        if not self.initialized:
            self.B[:] = 0.0
            gamma = np.dot(y, y)
            d = gamma/alpha

            if d > 0.0:
                np.fill_diagonal(self.B, gamma/alpha)
            else:
                np.fill_diagonal(self.B, 1.0)

            self.initialized = True

        # Compute beta = s^{T}*B*s
        t = np.dot(self.B, s)
        beta = np.dot(t, s)

        # Perform a damped update
        if alpha <= 0.2*beta:
            theta = 0.8*beta/(beta - alpha)
            r = theta*y + (1.0 - theta)*t

            alpha = np.dot(s, r)
            self.B += -np.outer(t, t)/beta + np.outer(r, r)/alpha

            return

        self.B += -np.outer(t, t)/beta + np.outer(y, y)/alpha

        return

    def mult(self, x, y):
        '''
        Perform a matrix-vector product
        '''

        y[:] = np.dot(self.B, x)

        return 

class QuasiNewton:
    def __init__(self, n, m_max, qn_type='BFGS'):
        '''
        This object implements limited-memory BFGS and SR-1 update
        schemes using the compact representations presented in Byrd,
        Nocedal and Schnabel, 1992. This allows the computation of:

        y <- B*x

        where B is an approximate Hessian.

        input:
        n:  the size of the Hessian approximation
        m:  the number of updates to store

        ----------------------------------------------------------
        The BFGS update can be written in the following form:

        B_k = B_0 - [B_0*S_k   Y_k]* M^{-1} * [ S_k^{T}*B_0 ]
        .                                     [ Y_k^{T}     ]

        where the matrix M is given as follows:
        
        M = [ S_k^{T}*B_0*S_k    L_k ]
        .   [ L_k^{T}           -D_k ]

        ----------------------------------------------------------
        The SR-1 matrix can be written in the following form:

        B_k = B_0 - (Y - B0*S)*M^{-1}*(Y - B0*S)^{T}
        
        where the matrix M takes the following form in this case:

        M = [ S^{T}*B0*S - D - L - L^{T} ]

        In this code, we restrict B_0 such that it is a diagonal matrix.
        '''

        self.n = n
        self.m_max = m_max
        self.m = 0

        self.qn_type = qn_type
        self.curvature_skip_update = False

        self.B0 = np.zeros(n)

        # You must call update before using anything
        self.M = None
        self.M_inv = None

        # Allocate the update vectors
        self.S = np.zeros((n, m_max))
        self.Y = np.zeros((n, m_max))

        # Set the log file pointer 
        self.log = None

        return

    def check_hessian(self):
        '''
        Check that the results are what we expect...
        '''
        y = np.zeros(self.n)

        for i in xrange(self.m-1, -1, -1):
            t = self.S[:, i]
            self.mult(t, y)

            print '[%d] Secant Hessian check %15.4e'%(
                self.m, np.sqrt(np.dot(y - self.Y[:, i],
                                       y - self.Y[:, i])))
        return

    def mult(self, x, y):
        '''
        Compute B*x -> y
        '''

        y[:] = self.B0*x
        
        # If no vectors have been added, there's nothing more
        # to do... 
        if self.M == None:
            return

        if self.qn_type == 'BFGS':
            # Allocate a temporary array
            m = self.m
            r = np.zeros(2*m)
            
            # Compute S_{k}^{T}*B_0
            for i in xrange(m):
                r[i] = np.dot(self.B0*self.S[:, i], x) 
                r[m + i] = np.dot(self.Y[:, i], x)

            # Compute r <- M^{-1}*r
            r = scipy.linalg.lu_solve(self.M_inv, r)

            # Compute y <- B0*x - B0*S*r[:m] - Y*r[m:]
            y[:] -= self.B0*np.dot(self.S[:, :m], r[:m])
            y[:] -= np.dot(self.Y[:, :m], r[m:])
        else: 
            # Allocate a temporary array
            m = self.m
            r = np.zeros(m)
            
            # Compute S_{k}^{T}*B_0
            for i in xrange(m):
                r[i] = np.dot(self.Y[:, i] - self.B0*self.S[:, i], x)

            # Compute r <- M^{-1}*r
            r = scipy.linalg.lu_solve(self.M_inv, r)

            # Compute y <- B0*x - B0*S*r[:m] - Y*r[m:]
            for i in xrange(m):
                y[:] -= (self.Y[:, i] - self.B0*self.S[:, i])*r[i]

        return

    def update(self, s, y, mu, c, ctmp, A, Atmp):
        '''
        Update the matrices involved in the BFGS/SR-1 formula using
        the s/y vectors provided.

        This updates the matrices required for the compact
        representation of the BFGS update.

        The BFGS matrix has the form:
        -----------------------------

        B_k = B_0 - [B_0*S_k   Y_k]* M^{-1} * [ S_k^{T}*B_0 ]
        .                                     [ Y_k^{T}     ]

        The matrix M is given as follows:
        
        M = [ S_k^{T}*B_0*S_k    L_k ]
        .   [ L_k^{T}           -D_k ]

        The SR-1 matrix has the following form:
        ---------------------------------------

        B_k = B_0 - (Y - B0*S)*M^{-1}*(Y - B0*S)^{T}
        
        where the matrix M takes the following form in this case:

        M = [ S^{T}*B0*S - D - L - L^{T} ]
        '''

        r = np.zeros(y.shape)

        # Set the diagonal entries if this is the first
        # time through
        if self.m == 0:
            alpha = np.dot(y, s)
            gamma = np.dot(y, y)

            # Set the diagonal entry
            d = gamma/alpha
            if d >= 0.0:
                self.B0[:] = d
            else:
                self.B0[:] = 1.0

        if self.qn_type == 'BFGS':
            # Compute alpha = s^{T}*y
            alpha = np.dot(s, y)
            gamma = np.dot(y, y)

            # Create a temporary array
            self.mult(s, r)
            
            # Compute r^{T}*s
            beta = np.dot(r, s)

            if self.curvature_skip_update:
                if alpha <= 1e-8*gamma:
                    if self.log: self.log.write('skipping L-BFGS update\n')
                    return
                else:
                    r[:] = y[:]
            else:
                # Add the vectors to the L-BFGS update depending
                # on the value of sk^{T}*yk
                # Modify the damped BFGS update criterion for if 
                # c(xk) and c(xk+1) are feasible
                '''
                if np.all(c >= 0.0) and np.all(ctmp >= 0.0):
                    inv_c = np.zeros(len(c))
                    inv_ctmp = np.zeros(len(ctmp))
                    for index in xrange(len(c)):
                        inv_c[index] = 1.0/c[index]
                        inv_ctmp[index] = 1.0/ctmp[index]
                    mod_a = mu*(np.dot(inv_ctmp,Atmp)-np.dot(inv_c,A))
                        
                    alpha_mod = np.dot(mod_a, s)
                    
                else:
                    alpha_mod = 0.0
                '''
                if (alpha) <= 0.2*beta:
                    if self.log: self.log.write('damped L-BFGS update\n')

                    theta = 0.8*beta/(beta - alpha)
                    r[:] = theta*y + (1.0 - theta)*r
                else:
                    r[:] = y[:]
        else:
            r[:] = y[:]

        # Set the diagonal entries if this is the first
        # time through
        gamma = np.dot(r, r)
        alpha = np.dot(r, s)
        self.B0[:] = gamma/alpha

        # Copy the vectors into the storage
        if self.m < self.m_max:
            self.S[:, self.m] = s[:]
            self.Y[:, self.m] = r[:]
            self.m += 1
        else:
            for i in xrange(self.m-1):
                self.S[:, i] = self.S[:, i+1]
                self.Y[:, i] = self.Y[:, i+1]
                        
            self.S[:, self.m-1] = s[:]
            self.Y[:, self.m-1] = r[:]

        # Copy this over to make life simpler
        m = self.m

        # Compute L = s_{i}^{T}*y_{j} for i > j
        L = np.zeros((m, m))
        for i in xrange(m):
            for j in xrange(i):
                L[i, j] = np.dot(self.S[:, i], self.Y[:, j])

        if self.qn_type == 'BFGS':
            # Allocate a new matrix of size m x m
            self.M = np.zeros((2*m, 2*m))

            # Compute the M matrix
            for i in xrange(m):
                for j in xrange(m):
                    self.M[i, j] = np.dot(self.S[:, i], 
                                          self.B0*self.S[:, j])
                
            # Fill in the off-diagonal matrices
            self.M[:m, m:] = L
            self.M[m:, :m] = L.T

            # Set the elements in the diagonal matrix
            for i in xrange(m):
                self.M[m+i, m+i] = -np.dot(self.S[:, i], self.Y[:, i])

            # Compute the LU-factorization of M
            self.M_inv = scipy.linalg.lu_factor(self.M)
        else:
            # Allocate a new matrix of size m x m
            self.M = np.zeros((m, m))

            # Compute the M matrix
            for i in xrange(m):
                for j in xrange(m):
                    self.M[i, j] = np.dot(self.S[:, i], 
                                          self.B0*self.S[:, j])

            # Add the contributions to the diagonal matrix
            self.M[:, :] -= L
            self.M[:, :] -= L.T

            for i in xrange(m):
                self.M[i, i] -= np.dot(self.S[:, i], self.Y[:, i])

            # Compute the LU-factorization of M
            self.M_inv = scipy.linalg.lu_factor(self.M)

        return

    def get_size(self):
        '''
        Get the size of the subspace
        '''

        if self.qn_type == 'BFGS':
            return 2*self.m
        else:
            return self.m

        return

    def get_vector(self, i):
        '''
        Get the i-th vector 
        '''

        if self.qn_type == 'BFGS':
            if i < 0 or i >= 2*self.m:
                raise ValueError('Index out of range') 
            if i < self.m:
                return self.B0*self.S[:, i]
            else:
                return self.Y[:, i-self.m]
        else:
            if i < 0 or i >= self.m:
                raise ValueError('Index out of range') 
            return (self.Y[:, i] - self.B0*self.S[:, i])

        return

class KKTMatrix:
    def __init__(self, x, s, z, zl, zu, 
                 b0, lb, ub, A):
        '''
        Form the information required for the KKT matrix
        '''

        # Set number of design variables/constraints
        self.n = x.shape[0]
        self.m = s.shape[0]

        # Keep pointers to the original data
        self.x = x
        self.lb = lb
        self.ub = ub
        self.s = s
        self.z = z
        self.zl = zl
        self.zu = zu
        self.A = A
        
        # Compute the diagonal matrix c 
        self.c = b0 + zl/(x - lb) + zu/(ub - x)

        # Compute and factor the matrix D = Z^{-1}S + A*C^{-1}*A^{T} 
        self.D = np.zeros((self.m, self.m))

        for i in xrange(self.m):
            self.D[i, i] = s[i]/z[i]

        for i in xrange(self.m):
            for j in xrange(self.m):
                self.D[i, j] += np.dot(self.A[i, :], self.A[j, :]/self.c)

        # Factor the matrix D
        self.D_factor = scipy.linalg.lu_factor(self.D)

        return

    def solve(self, b):
        '''
        Solve the KKT system
        '''

        # Slice up the array for easier access
        bx = b[0:3*self.n:3]
        bl = b[1:3*self.n:3]
        bu = b[2:3*self.n:3]  
        bc = b[3*self.n:3*self.n+self.m]
        bs = b[3*self.n+self.m:]

        # Get the right-hand-side of the first equation
        d = (bx + bl/(self.x - self.lb) - bu/(self.ub - self.x))
        
        # Compute the right-hand-side for the Lagrange multipliers
        rz = (bc + bs/self.z - np.dot(self.A, d/self.c))

        # Compute the step in the Lagrange multipliers
        pz = scipy.linalg.lu_solve(self.D_factor, rz)

        # Compute the step in the slack variables
        ps = (bs - self.s*pz)/self.z

        # Compute the step in the design variables
        px = (d + np.dot(self.A.T, pz))/self.c

        # Compute the step in the bound Lagrange multipliers
        pzl = (bl - self.zl*px)/(self.x - self.lb)
        pzu = (bu + self.zu*px)/(self.ub - self.x)

        # Now create the output array and assign the values
        x = np.zeros(b.shape)

        x[0:3*self.n:3] = px
        x[1:3*self.n:3] = pzl
        x[2:3*self.n:3] = pzu
        x[3*self.n:3*self.n+self.m] = pz
        x[3*self.n+self.m:] = ps

        return x

class HOpt:
    def __init__(self, opt, x, lb, ub, n, m,
                 comm=MPI.COMM_WORLD,
                 max_qn_subspace=20, qn_type='BFGS', 
                 max_major_iters=150, inverse_scaling=False):
        '''
        Initialize the optimization class
        
        The optimization is only performed on the root processor,
        comm.rank == 0, but all objective/constraint, gradient,
        Hessian and file-writing calls are made on all processors
        through the use of a wait loop. As a result, the internal
        values are only correct on the root processor.
        '''

        # Assign the optimization instance
        self.opt = opt
        
        # Set the size of the design space to use
        self.n = n # The number of design variables
        self.m = m # The number of constraints

        # Set the flag for inverse scaling
        self.inverse_scaling = inverse_scaling

        # Copy the communicator
        self.comm = comm

        # Set the log file pointer to None
        self.log = None

        if self.inverse_scaling:
            # Check that the lower and upper bounds are strictly positive
            for i in xrange(self.n):
                if lb[i] <= 0.0 or ub[i] <= 0.0:
                    print 'Inverse scaling not permitted for lb or ub <= 0.0'

            # Set the inverse arrays for scaling
            self.x_scale = 100.0*np.array(lb)
            self.x = self.x_scale/np.array(x)
            self.lb = self.x_scale/np.array(ub)
            self.ub = self.x_scale/np.array(lb)
        else:
            # Set the arrays for the primal variables + upper/lower
            # bounds
            self.x_scale = np.ones(x.shape)
            self.x = np.array(x)
            self.lb = np.array(lb)
            self.ub = np.array(ub)
        
        # Allocate an array for the dual variables
        self.z = np.ones(self.m)
        self.zl = np.ones(self.n)
        self.zu = np.ones(self.n)
        
        # Allocate arrays for the slack variables
        self.s = np.ones(self.m)

        # Check that we have a feasible point w.r.t. bounds - this is
        # required since we use a strict interior point method for the
        # bounds
        for i in xrange(self.n):
            if self.lb[i] >= self.ub[i]:
                raise ValueError('Lower and upper bounds inconsistent')
            if self.x[i] <= self.lb[i]:
                self.x[i] = 0.5*(self.lb[i] + self.ub[i])
            if self.x[i] >= self.ub[i]:
                self.x[i] = 0.5*(self.lb[i] + self.ub[i])

        # Use an initial starting point strategy
        self.init_starting_point = True

        # Set the initial residual norm
        self.init_res_norm = 1.0

        # Set the maximum scaling value for the residual
        self.max_res_scaling = 100.0

        # Set the constraint scaling
        self.constraint_scale = 1.0

        # Set the frequency with which files shall be output! 
        self.write_output_frequency = 10
        
        # Set the complementarity
        self.mu = 0.1
        self.barrier_strategy = 'monotone'

        # Set the relative tolerance for the barrier problem
        self.monotone_rtol = 0.1
        self.monotone_barrier_fraction = 0.25
        self.monotone_barrier_power = 1.5 

        # Set the fractional reduction of the barrier problem - used
        # for the fractional barrier strategy
        self.fractional_barrier_param = 0.75

        # Use a line search - or take the full step each time
        # This is more useful for debugging
        self.use_line_search = True
        self.line_search_type = 'Armijo'

        # Use the additional penalty parameter increment
        # due to the positive curvature information
        self.use_curvature_penalty = True

        # Set the minimum fraction to the boundary
        self.min_fraction_to_boundary = 0.95

        # Set some initial parameters that should be set elsewhere
        # in a more refined version of this code
        self.max_major_iters = max_major_iters
        self.max_line_iters = 5
        self.abs_opt_tol = 1e-6
        self.rel_opt_tol = 1e-12

        # Set the maximum relative tolerance for the inexact
        # Krylov solves
        self.rtol_max = 0.1

        # Set the intial value of rho - the penalty parameter in the
        # augmented Lagrangian merit function 
        self.rho = 0.0

        # Perform updates using the BFGS formula
        self.bfgs_update = True
        self.use_bfgs_pc = True
        self.use_hvec = True

        # Use the problem-specific preconditioner
        self.use_ks_pc = False

        # Check if the optimization class defines hvec_prod
        if getattr(self.opt, "hvec_prod", None) is None:
            self.use_hvec = False

        # Set the type of factorization to use
        self.kkt_matrix_factor_type = 'default'

        # Set the minimum step length
        self.min_step_length = 1e-3

        # Set the matrix diagonal factor
        self.mat_diag_factor = 0.0

        # Set the FGMRES subspace size
        self.fgmres_subspace_size = 10

        # The steepness of the descent direction as a fraction of the
        # constraint violation/penalty parameter. i.e.
        # argphi'(alpha=0) <= - penalty_descent_fraction*c^{T}(x)*c(x)
        self.penalty_descent_fraction = 0.1

        # Tolerance below which to switch to FGMRES.
        # rtol: switch when the barrier problem is solved this far
        # atol: switch when the optimization problem is solved this far
        self.fgmres_switch_rtol = 5e-1
        self.fgmres_switch_atol = 1e-3

        # Keep track of different values
        self.major_iterations = 0
        self.function_evals = 0
        self.gradient_evals = 0
        self.hessian_vec_prods = 0

        # Set up the full BFGS approximation
        self.qn = QuasiNewton(self.n, max_qn_subspace, 
                              qn_type=qn_type)

        # Set the matrix to None - for re-initialization later
        self.Be = None
        self.Be_factor = None
        
        return

    def print_vars(self, file_name=None):
        '''
        Print the values of the primal and dual variables
        '''

        if self.comm.rank != 0:
            return

        fp = None
        if file_name:
            fp = open(file_name, 'w')

        s = 'Objective: %15.6e\n\n'%(self.obj)        
        s += 'Constraint slacks and dual variables\n'
        for i in xrange(self.m):
            s += 'c[%4d]: %15.6e    s[%4d]: %15.6e   z[%4d]: %15.6e\n'%(
                i, self.c[i], i, self.s[i], i, self.z[i])

        if fp: fp.write(s)
        else: print s
            
        s = 'Primal and upper/lower dual variables\n'
        for i in xrange(self.n):
            s += 'x[%4d]: %15.6e   zl[%4d]: %15.6e  zu[%4d]: %15.6e\n'%(
                i, self.x[i], i, self.zl[i], i, self.zu[i])
        
        if fp:
            s = '\n' + s
            fp.write(s)
            fp.close()
        else:
            print s

        return

    def compute_kkt_res(self, res, x, z, s, zl, zu, 
                        g, c, A, mu):
        '''
        Compute the (negative of the) KKT residual for the reduced
        system

        res = -[ g - A^T*z - zl + zu ]
        .      [ c(x) - s            ]
        .      [ S*z - mu*e          ]
        .      [ (x - xl)*zl - mu*e  ] 
        .      [ (ub - x)*zu - mu*e  ]

        Note that these residuals do not actually depend on the values
        of the slack variables s.

        input:
        x:  the values of the design variables
        g:  the derivative of the objective
        c:  the constraint values              
        A:  the constraint Jacobian
        z:  the dual variables

        in/out:
        res: the KKS residuals
        '''
        
        # Record the number of variables and constraints
        n = self.n
        m = self.m
        
        # Evaluate the KKT system 
        res[0:3*n:3] = -(g - np.dot(A.T, z) - zl + zu)

        # Add the bound constraints on x
        res[1:3*n:3] = -((x - self.lb)*zl - mu)
        res[2:3*n:3] = -((self.ub - x)*zu - mu)
        
        # Add the residuals due to the inequalities
        res[3*n:3*n+m]     = -(c - s)
        res[3*n+m:3*n+2*m] = -(s*z - mu)

        return

    def set_up_mat_pc(self, x, z, s, zl, zu, A, use_ks_pc=False):
        '''
        Initialize data required for the matrix and preconditioner

        Compute the factorization of the matrix:
        
        K*p = [  B0  -A^{T}  0  -I         I        ][ px  ]
        .     [  A    0     -I   0         0        ][ pz  ] = res
        .     [  0    S      Z   0         0        ][ ps  ]
        .     [  Zl   0      0   (X - Xl)  0        ][ pzl ]
        .     [ -Zu   0      0   0         (Xu - X) ][ pzu ]
        '''

        # Retrieve the number of variables and constraints
        n = self.n
        m = self.m

        if use_ks_pc:
            d0, self.M, self.Z = self.opt.set_up_pc(x, -z)
        else:
            d0 = self.qn.B0

        if self.kkt_matrix_factor_type == 'bordered':
            # Instead of using a full factorization of the matrix, use
            # a bordered factorization that uses a reduction to the
            # lagrange multipliers: z. This will be successful if the
            # number of constraints is small...
            kkt_mat = KKTMatrix(self.x, self.s, self.z,
                                self.zl, self.zu, d0 + self.mat_diag_factor,
                                self.lb, self.ub, self.A)

            self.Be_factor = kkt_mat.solve
        else:
            if self.Be is None:
                # Zero the entire matrix
                self.Be = sparse.lil_matrix((3*self.n + 2*self.m,
                                             3*self.n + 2*self.m))
                
            # Set the componets of the first equation
            # ---------------------------------------
            for i in xrange(n):
                # Set the diagonal 
                self.Be[3*i, 3*i] = (self.mat_diag_factor + d0[i])

                # Set the contributions from the Lagrange multipliers
                self.Be[3*i, 3*i+1] = -1.0
                self.Be[3*i, 3*i+2] = 1.0

                # Add the contribution from the bound constraints
                self.Be[3*i+1, 3*i] = zl[i]
                self.Be[3*i+1, 3*i+1] = (x[i] - self.lb[i])

                self.Be[3*i+2, 3*i] = -zu[i]
                self.Be[3*i+2, 3*i+2] = (self.ub[i] - x[i])

            # Set the constraint Jacobians
            self.Be[0:3*n:3, 3*n:3*n+m] = -self.A.T

            # Add the values from the linearization of c - s = 0
            # --------------------------------------------------
            self.Be[3*n:3*n+m, 0:3*n:3] = self.A

            for i in xrange(m):
                self.Be[3*n+i, 3*n+m+i] = -1.0

            # Set the values for the final equation: s*z - mu = 0
            # ---------------------------------------------------
            for i in xrange(m):
                # Add the contribution from the linearization of the
                # complementarity constraint
                self.Be[3*n+m+i, 3*n+i]   = s[i]
                self.Be[3*n+m+i, 3*n+m+i] = z[i]

            # Factor the system of equations - this can take a long time.
            tf = MPI.Wtime()
            if self.kkt_matrix_factor_type == 'splu':
                Be = self.Be.tocsc()
                self.Be_factor = sci_alg.splu(Be).solve
            else: 
                # Default factorization: umfpack if installed, otherwise,
                # SuperLU. This depends on the scipy installation.
                Be = self.Be.tocsc()
                self.Be_factor = sci_alg.dsolve.factorized(Be)

            tf = MPI.Wtime() - tf
            if self.log: 
                self.log.write('KKT matrix factorization time: %10.2f s\n'%(tf))

        # Now that we have a factorization of the full KKT matrix with
        # the diagonal from the Hessian matrix, compute the remainder
        # of the information required to solve a system with the full
        # KKT system...
        if use_ks_pc:
            # Get the size of the subspace
            size = self.Z.shape[1]

            # Compute the factor Ce - the Schur complement
            self.Ce = np.zeros((size, size))
            self.Ce[:] = -self.M[:]
        
            # Create a temporary array for the computation
            r = np.zeros(3*self.n + 2*self.m)

            # Compute the Ce factor
            for j in xrange(size):
                # Compute Be^{-1}*z_i
                r[:] = 0.0
                r[0:3*n:3] = self.Z[:, j]
                r[:] = self.Be_factor(r)

                for i in xrange(size):
                    cij = np.dot(self.Z[:n, i], r[0:3*n:3])
                    self.Ce[i, j] += cij

            # Factor the matrix (W0^{T}*Be^{-1}*W0 - M)
            self.Ce_factor = scipy.linalg.lu_factor(self.Ce)
        elif self.qn.get_size() > 0:
            size = self.qn.get_size()

            # Compute the factor Ce - the Schur complement
            self.Ce = np.zeros((size, size))
            self.Ce[:] = -self.qn.M[:]
        
            # Create a temporary array for the computation
            r = np.zeros(3*self.n + 2*self.m)

            # Compute the Ce factor
            for j in xrange(size):
                # Compute Be^{-1}*z_i
                r[:] = 0.0
                r[0:3*n:3] = self.qn.get_vector(j)
                r[:] = self.Be_factor(r)

                for i in xrange(size):
                    t = self.qn.get_vector(i)
                    cij = np.dot(t, r[0:3*n:3])
                    self.Ce[i, j] += cij

            # Factor the matrix (W0^{T}*Be^{-1}*W0 - M)
            self.Ce_factor = scipy.linalg.lu_factor(self.Ce)

        return

    def mult(self, x, y):
        '''
        Compute the matrix-vector product:

        K*p = [  B0  -A^{T}  0  -I         I        ][ px  ]
        .     [  A    0     -I   0         0        ][ pz  ] = y
        .     [  0    S      Z   0         0        ][ ps  ]
        .     [  Zl   0      0   (X - Xl)  0        ][ pzl ]
        .     [ -Zu   0      0   0         (Xu - X) ][ pzu ]

        input:
        x:  the input perturbation
        y:  the matrix-vector product with x: y <- K*x
        '''

        # Get the number of design variables and constraints
        n = self.n
        m = self.m

        # Compute the iterates in the x, z and s directions
        px = x[0:3*n:3]
        pzl = x[1:3*n:3]
        pzu = x[2:3*n:3]

        pz = x[3*n:3*n+m]
        ps = x[3*n+m:]
        
        # Compute the Hessian-vector product
        Hw = self.hvec(self.x, self.z, px,
                       g=self.gobj, A=self.A)

        # Assign the results to the output array
        y[0:3*n:3] = (Hw + self.mat_diag_factor*px 
                      - np.dot(self.A.T, pz) - pzl + pzu)
        y[1:3*n:3] = self.zl*px + (self.x - self.lb)*pzl
        y[2:3*n:3] = -self.zu*px + (self.ub - self.x)*pzu

        y[3*n:3*n+m] = np.dot(self.A, px) - ps
        y[3*n+m:] = self.z*ps + self.s*pz

        return

    def apply_pc(self, x, y):
        '''
        Apply the plain old preconditioner
        '''

        y[:] = self.Be_factor(x)

        return

    def apply_bfgs_pc(self, x, y):
        '''
        Apply the preconditioner based on the current Hessian estimate

        Given B = B0 - Z*M^{-1}*Z, and the Be_factor matrix that
        stores the KKT matrix above, compute:

        y <- Be^{-1}*x
        y <- y - K^{-1}*Z*Ce^{-1}*Z^{T}*y

        where Ce = Z^{T}*Be^{-1}*Z - M^{-1}
        '''

        y[:] = self.Be_factor(x)

        if self.qn.get_size() > 0:
            size = self.qn.get_size()

            # Allocate a temporary vector
            yv = np.zeros(y.shape)
            r = np.zeros(size)

            for i in xrange(size):
                t = self.qn.get_vector(i)
                r[i] = np.dot(t, y[0:3*self.n:3])

            # Compute r <- Ce^{-1}*r
            r[:] = scipy.linalg.lu_solve(self.Ce_factor, r)

            for i in xrange(size):
                t = self.qn.get_vector(i)
                yv[0:3*self.n:3] += t*r[i]

            # Compute yv <- Be^{-1}*yv
            y[:] -= self.Be_factor(yv)

        return

    def apply_ks_pc(self, x, y):
        '''
        Apply the specialized preconditioner for the KS function
        '''

        y[:] = self.Be_factor.solve(x)

        size = self.Z.shape[1]

        # Allocate a temporary vector
        yv = np.zeros(y.shape)
        r = np.zeros(size)

        for i in xrange(size):
            r[i] = np.dot(self.Z[:, i], y[0:3*self.n:3])

        # Compute r <- Ce^{-1}*r
        r[:] = scipy.linalg.lu_solve(self.Ce_factor, r)

        for i in xrange(size):
            yv[0:3*self.n:3] += r[i]*self.Z[:, i]

        # Compute yv <- Be^{-1}*yv
        y[:] -= self.Be_factor(yv)

        return

    def compute_max_step(self, px, pz, ps, pzl, pzu, tau=0.995):
        '''
        Compute the maximum step lengths in x, z and s with the
        specified fraction to the boundary tau

        input:
        x, z, s:    the current values of x, z, and s
        px, pz, ps: the steps in x, z, and s        
        pzl, pzu:   the steps in the Lagrange multipliers
        
        returns:
        max_z, max_s: maximum step lengths in x, z, s to maintain
        feasibility

        The lower bounds on s/z are enforced as follows:
        
        alpha = -tau*z/pz for pz < 0.0

        The lower/upper bounds on x are enforced as follows:
        
        alpha =  tau*(ub - x)/px   px > 0
        alpha = -tau*(x - lb)/px   px < 0
        '''

        # Check the distance of the design vars to the boundary
        max_x = 1.0
        x_str = 'n'
        for i in xrange(self.n):
            if px[i] < 0.0:
                m = -tau*(self.x[i] - self.lb[i])/px[i]
                if m < max_x:
                    max_x = m
                    x_str = 'l%d'%(i)
            if px[i] > 0.0:
                m = tau*(self.ub[i] - self.x[i])/px[i]
                if m < max_x:
                    max_x = m
                    x_str = 'u%d'%(i)

        # Check the slacks
        for i in xrange(self.m):
            if ps[i] < 0.0:
                m = -tau*self.s[i]/ps[i]
                if m < max_x:
                    max_x = m
                    x_str = 's%d'%(i)

        # Check the Lagrange multiplier distance to the barrier
        max_z = 1.0
        z_str = 'n'
        for i in xrange(self.m):
            if pz[i] < 0.0:
                m = -tau*self.z[i]/pz[i]
                if m < max_z:
                    max_z = m
                    z_str = 'z%d'%(i)

        for i in xrange(self.n):
            if pzl[i] < 0.0:
                m = -tau*self.zl[i]/pzl[i]
                if m < max_z:
                    max_z = m
                    z_str = 'L%d'%(i)
            if pzu[i] < 0.0:
                m = -tau*self.zu[i]/pzu[i]
                if m < max_z:
                    max_z = m
                    z_str = 'U%d'%(i)

        return max_x, max_z, x_str, z_str

    def objcon(self, x):
        '''
        Evaluate the objective and constraints and apply the required
        scaling factor. Test for a failed point.
        '''
        
        if self.comm.rank == 0 and self.comm.size > 1:
            # If this is the root processor, broadcast to every
            # other processor
            mode = 0
            self.comm.bcast(mode, root=0)
            self.comm.bcast(x, root=0)

        self.function_evals += 1

        if self.inverse_scaling:
            t = self.x_scale/x
            obj, con, fail = self.opt.obj_con(t)
        else:
            # Evaluate the objective and constraint values
            obj, con, fail = self.opt.obj_con(x)
            con *= self.constraint_scale

        return obj, con

    def gobjcon(self, x):
        '''
        Evaluate the derivative of the constraints and the constraint
        gradients.
        '''

        if self.comm.rank == 0 and self.comm.size > 1:
            # Broadcast the mode and the values of x to the 
            # other processors
            mode = 1
            self.comm.bcast(mode, root=0)
            self.comm.bcast(x, root=0)

        self.gradient_evals += 1
        
        if self.inverse_scaling:
            t = self.x_scale/x
            self.g_n, self.A_n, fail = self.opt.gobj_con(t)

            g = -(self.x_scale/x**2)*self.g_n
            A = -(self.x_scale/x**2)*self.A_n
        else:
            g, A, fail = self.opt.gobj_con(x)
            A *= self.constraint_scale

        return g, A

    def hvec(self, x, z, w, g=None, A=None):
        '''
        Compute the hessian-vector product.

        *If* scaling is applied, adjust the result to reflect the
         scaled variables.
        '''

        if self.comm.rank == 0 and self.comm.size > 1:
            # Broadcast the mode and the values of x to the 
            # other processors
            mode = 2
            self.comm.bcast(mode, root=0)
            self.comm.bcast((x, z, w, g, A), root=0)

        self.hessian_vec_prods += 1

        if self.inverse_scaling:
            t = self.x_scale/x
            Hw, fail = self.opt.hvec_prod(t, -z, w*(self.x_scale/x**2))
            Hw *= (self.x_scale/x**2)

            Hw -= 2*(w/x)*(g - np.dot(A.T, z))
        else:
            Hw, fail = self.opt.hvec_prod(x, -self.constraint_scale*z, w)

        return Hw

    def write_output_files(self, x):
        '''
        Write the output files
        '''
        if self.comm.rank == 0 and self.comm.size > 1:
            mode = 3
            self.comm.bcast(mode, root=0)
            self.comm.bcast(x, root=0)

        if hasattr(self.opt, 'write_output_files'):
            self.opt.write_output_files(x)
        
        return

    def wait_loop(self):
        '''
        Wait for calls to be made on the root processor.

        This must be called from processors with comm.rank != 0
        '''
        
        while True:
            # Perform a wait loop. Based on the 'mode' argument,
            # make calls to the appropriate functions
            mode = self.comm.bcast(root=0)
            
            # Exit completely if we're not on the root processor.
            if mode == -1:
                return

            # Recieve the appropriate arguments
            if mode == 0:
                x = self.comm.bcast(root=0)
                self.objcon(x)
            elif mode == 1:
                x = self.comm.bcast(root=0)
                self.gobjcon(x)
            elif mode == 2:
                (x, z, w, g, A) = self.comm.bcast(root=0)
                self.hvec(x, z, w, g=g, A=A)
            elif mode == 3:
                x = self.comm.bcast(root=0)
                self.write_output_files(x)

        return

    def optimize(self, file_name=None, var_file=None, 
                 log_file=None):
        '''
        Perform the optimization with a back-tracking line search
        '''
        
        if self.comm.rank != 0:
            # Call the wait loop on
            self.wait_loop()

            # Receive the broadcast from the root processor
            self.comm.Bcast(self.x, root=0)
            self.comm.Bcast(self.z, root=0)
            self.comm.Bcast(self.s, root=0)
            self.comm.Bcast(self.zl, root=0)
            self.comm.Bcast(self.zu, root=0)

            return

        # Open the log file
        if log_file is not None:
            if isinstance(log_file, str): 
                try:
                    self.log = open(log_file, 'w')
                except:
                    print 'Could not create log file: %s'%(log_file)
                    self.log = None
            elif isinstance(log_file, file) and not log_file.closed:
                self.log = log_file

            if self.qn:
                self.qn.log = self.log
                
        # Allocate the right-hand-side
        res = np.zeros(3*self.n + 2*self.m)

        # Set the incremental residual norm
        incr_res_norm = 1.0

        # Initialize the objective, constraints and
        # obj/con gradients at the initial point
        self.obj, self.c = self.objcon(self.x)
        self.gobj, self.A = self.gobjcon(self.x)
        
        if self.init_starting_point:            
            # Estimate the Lagrange multipliers by finding the
            # minimum norm solution to the problem:
            # A^{T}*z = g - zl + zu
            rhs = np.dot(self.A, self.gobj - self.zl + self.zu)

            # Solve the normal equations
            C = np.dot(self.A, self.A.T)
            z = np.linalg.solve(C, rhs)

            # If the least squares multipliers lie on the interval [0,
            # 1e3] use them, otherwise keep the pre-assigned values
            for i in xrange(self.m):
                if z[i] >= 0.0 and z[i] < 1e3:
                    self.z[i] = z[i]

        # Keep track of the internal condition 
        converged = False
        
        # Set the new barrier flag to true initially
        new_barrier = True

        # Keep track of the last step length for monitoring purposes
        alpha = 1.0

        # Initialize the file pointer
        fp = None
        if file_name:
            fp = open(file_name, 'w')
            fp.write('variables = iteration, alpha, obj, opt, ')
            fp.write('infeas, res, barrier, comp\n')
        
        # Create temporary solution vectors
        y = np.zeros(res.shape)
        y_affine = np.zeros(res.shape)

        for i in xrange(self.max_major_iters):
            # Write the output files
            if i % self.write_output_frequency == 0:
                self.write_output_files(self.x)
                self.print_vars(file_name=var_file)

            # Compute the average complementarity
            comp = (np.sum(self.z*self.s) + 
                    np.sum(self.zl*(self.x - self.lb)) +
                    np.sum(self.zu*(self.ub - self.x)))/(
                self.m + 2*self.n)
            
            # Safe guard the primal-dual multipliers by the primal
            # multiplier estimates. This shouldn't really be required.
            for k in xrange(self.n):
                zp = self.mu/(self.x[k] - self.lb[k])
                self.zl[k] = max(min(self.zl[k], 1e4*zp), zp/1e4)

                zp = self.mu/(self.ub[k] - self.x[k])
                self.zu[k] = max(min(self.zu[k], 1e4*zp), zp/1e4)

            # Compute the residual of the KKT system with the
            # current barrier parameter
            self.compute_kkt_res(res, self.x, self.z, 
                                 self.s, self.zl, self.zu, 
                                 self.gobj, self.c, self.A, self.mu)

            # Set up the preconditioner class
            self.set_up_mat_pc(self.x, self.z, self.s, 
                               self.zl, self.zu, self.A)

            # Compute the stopping criterion
            sd = max(self.max_res_scaling,
                     (np.sum(np.abs(self.z)) + 
                      np.sum(np.abs(self.zl)) + 
                      np.sum(np.abs(self.zu)))/(self.m + 2*self.n))
            sd /= self.max_res_scaling

            opt_norm = np.max(np.abs(res[0:3*self.n:3]))/sd
            con_norm = np.max(np.abs(self.c - self.s))
            comp_norm = max(np.max(np.abs(res[1:3*self.n:3])),
                            np.max(np.abs(res[2:3*self.n:3])),
                            np.max(np.abs(res[3*self.n+self.m:])))/sd

            res_norm = max(opt_norm, con_norm, comp_norm)

            if i == 0:
                # If we're using the starting point strategy, use the
                # initial norm of the residual
                if self.init_starting_point:
                    self.init_res_norm = res_norm

                # Set the true value of the initial incremental
                # residual
                incr_res_norm = res_norm

            if self.barrier_strategy == 'monotone':
                # This is the monotone approach of Fiacco and McCormick
                new_barrier = (res_norm < 10.0*self.mu)

                if new_barrier:
                    # Record the value of the old barrier function
                    mu_old = self.mu

                    # Compute the new barrier parameter: It is either:
                    # 1. A fixed fraction of the old value
                    # 2. A function mu**exp for some exp > 1.0
                    # Point 2 ensures superlinear convergence (eventually)
                    self.mu = min(self.monotone_barrier_fraction*self.mu, 
                                  self.mu**self.monotone_barrier_power)

                    # Adjust the residual for the new barrier parameter
                    res[1:3*self.n:3] -= (mu_old - self.mu)
                    res[2:3*self.n:3] -= (mu_old - self.mu)
                    res[3*self.n+self.m:] -= (mu_old - self.mu)

                    # Adjust the residual norm - exclude the only 
                    # part that has changed
                    res_norm = max(opt_norm, con_norm)

                    # Set the new residual norm
                    incr_res_norm = res_norm

                    # Reset the penalty parameter to zero
                    self.rho = 0.0

                    # Set the new barrier parameter to true
                    new_barrier = True

            elif self.barrier_strategy == 'fractional':
                mu_old = self.mu
                sigma = self.fractional_barrier_param
                self.mu = sigma*comp

                # Adjust the residual for the new barrier parameter
                res[1:3*self.n:3] -= (mu_old - self.mu)
                res[2:3*self.n:3] -= (mu_old - self.mu)
                res[3*self.n+self.m:] -= (mu_old - self.mu)

                # Adjust the residual norm
                res_norm = np.sqrt(np.dot(res, res))
            elif self.barrier_strategy == 'LOQO':
                pass

            elif self.barrier_strategy == 'Mehrotra':
                # Compute the affine residual
                res[self.n+self.m:] -= self.mu

                # Apply the 'preconditioner' to the affine residual
                self.apply_bfgs_pc(res, y_affine)

                # Extract the affine direction
                px = y_affine[0:3*self.n:3]
                pzl = y_affine[1:3*self.n:3]
                pzu = y_affine[2:3*self.n:3]
                pz = y_affine[3*self.n:3*self.n+self.m]
                ps = y_affine[3*self.n+2*self.m:]

                # Compute the step lengths for the full step towards
                # the boundary
                a_x, a_z, x_s, z_s = self.compute_max_step(px, pz, ps, 
                                                           pzl, pzu, tau=1.0)

                # Compute the complementarity of the full affine step
                comp_affine = (
                    np.sum((self.z + a_z*pz)*(self.s + a_x*ps)) + 
                    np.sum((self.zl + a_z*pzl)*(self.x + a_x*px - self.lb)) +
                    np.sum((self.zu + a_z*pzu)*(self.ub - self.x - a_x*px)))/(
                    self.m + 2*self.n)

                # Compute the new barrier parameter based on the
                # new scaling direction
                sigma = (comp_affine/comp)**3

                # Compute the new value of the barrier parameter
                self.mu = sigma*comp

                # Adjust the residual for the new barrier parameter
                res[self.n+self.m:] += self.mu

                # Adjust the residual norm
                res_norm = np.sqrt(np.dot(res, res))

            # Print opt/infeas to the log file
            s = '\nMajor[%3d]: |opt|: %10.3e  |c-s|: %10.3e  '%(
                i, opt_norm, con_norm)
            s += '|d|: %10.3e  mu: %8.2e\n'%(comp_norm, self.mu)
            if self.log: self.log.write(s)

            if fp:
                # Write the current state of things to the file
                s = '%3d %15.8e %15.8e %15.8e '%(
                    self.major_iterations, alpha, self.obj, opt_norm)
                s += '%15.8e %15.8e %15.8e %15.8e\n'%(
                    con_norm, res_norm, self.mu, comp)
                fp.write(s)

            if i % self.write_output_frequency == 0:
                if fp: fp.flush()
                if self.log: self.log.flush()

            # Increment the major iteration parameter
            self.major_iterations += 1

            # Compute the relative solution tolerance 
            if new_barrier:
                rtol = self.rtol_max
            else:
                r = res_norm/incr_res_norm
                rtol = self.rtol_max*min(1.0, r)

            # Perform the convergence test
            if (res_norm < self.rel_opt_tol*self.init_res_norm or
                res_norm < self.abs_opt_tol):
                converged = True
                break

            # Record the current gradient 
            gtmp = np.array(self.gobj)
            Atmp = np.array(self.A)
            sk = np.array(self.x)

            if (self.use_hvec and
                (res_norm/incr_res_norm < self.fgmres_switch_rtol or
                 res_norm/self.init_res_norm < self.fgmres_switch_atol)):
                # Set the preconditioner function handle
                pc = self.apply_pc

                if self.use_bfgs_pc:
                    pc = self.apply_bfgs_pc
                elif self.use_ks_pc:
                    # Set up the preconditioner class
                    self.set_up_mat_pc(self.x, self.z, self.s, 
                                       self.zl, self.zu, self.A,
                                       use_ks_pc=True)
                    pc = self.apply_ks_pc

                # Approximately solve with FGMRES
                y = self.fgmres(res, m=self.fgmres_subspace_size, 
                                rtol=rtol, pc=pc)
            else:
                # Just apply the regular quasi-Newton preconditioner
                self.apply_bfgs_pc(res, y)

            # Reset the barrier parameter
            if new_barrier:
                new_barrier = False

            # Extract the line search direction
            px = y[0:3*self.n:3]
            pzl = y[1:3*self.n:3]
            pzu = y[2:3*self.n:3]
            pz = y[3*self.n:3*self.n+self.m]
            ps = y[3*self.n+self.m:]

            # Compute the step to the boundary
            tau = max(self.min_fraction_to_boundary, 1.0-self.mu)

            # Evaluate the maximum steps in x, z, and the slacks
            a_x, a_z, x_str, z_str = self.compute_max_step(px, pz, ps, 
                                                           pzl, pzu, tau=tau)

            # Bound the difference in the step lengths
            if a_x >= a_z:
                a_x = max(min(a_x, 1e2*a_z), a_z/1e2)
            else: # a_z >= a_x
                a_z = max(min(a_z, 1e2*a_x), a_x/1e2)

            # Check the complementarity of the full step - if it increases,
            # use equal sized steps
            comp_new = (
                np.sum((self.z + a_z*pz)*(self.s + a_x*ps)) + 
                np.sum((self.zl + a_z*pzl)*(self.x + a_x*px - self.lb)) +
                np.sum((self.zu + a_z*pzu)*(self.ub - self.x - a_x*px)))/(
                self.m + 2*self.n)
            
            if comp_new > comp:
                a_x = min(a_x, a_z)
                a_z = a_x

            # Scale the directions by the step lengths to make things
            # easier for the line search
            px *= a_x
            ps *= a_x
                
            # Scale the Lagrange multiplier steps
            pz *= a_z
            pzl *= a_z
            pzu *= a_z

            # Log the step lengths
            s = 'Step lengths:   x: %10.3e      z: %10.3e |px|: %10.3e %s %s\n'%(
                a_x, a_z, np.sqrt(np.dot(px, px)), x_str, z_str)
            if self.log: self.log.write(s)

            if self.use_line_search:
                # Evaluate the required penalty parameter for the
                # sufficient decrease conditions
                rho_hat = self.get_penalty_param(a_x, px, ps, self.gobj, self.c, 
                                                 self.x, self.z, self.s, 
                                                 self.zl, self.zu)
                
                # Set the penalty parameter to the smallest
                # non-negative value
                if rho_hat > self.rho:
                    self.rho = rho_hat
                else:
                    # Damp the value of the penalty parameter
                    self.rho = max(rho_hat, 0.5*self.rho)

                # Compute the merit function at the current point
                m_0 = self.eval_merit_func(self.obj, self.c, 
                                           self.x, self.z, self.s,
                                           self.zl, self.zu)
                
                # Compute the derivative of the merit function at the
                # current point
                dm_0 = self.eval_merit_deriv(px, pz, ps, pzl, pzu,
                                             self.gobj, self.c, self.A, 
                                             self.x, self.z, self.s, 
                                             self.zl, self.zu)

                # Compute an update based on a line search
                alpha = self.linesearch(m_0, dm_0, px, pz, ps, pzl, pzu,
                                        line_search_type=self.line_search_type)

                # Scale alpha by the initial scaling
                alpha *= a_x
            else:
                # Take the full step without a line search
                self.x += px
                self.z += pz
                self.s += ps
                self.zl += pzl
                self.zu += pzu

                # Alpha is just the step scaling parameter in this case
                alpha = a_x

                # Evaluate the objective/constraints and their derivatives
                self.obj, self.c = self.objcon(self.x)
                self.gobj, self.A = self.gobjcon(self.x)

            if self.bfgs_update:
                # Compute the update in the Hessian
                yk = ((self.gobj - np.dot(self.A.T, self.z)) - 
                      (gtmp - np.dot(Atmp.T, self.z)))
                sk = self.x - sk           
                
                # Add the vectors to the quasi-Newton update
                # Create variables for update
                xtmp = self.x - px
                objtmp, ctmp = self.objcon(xtmp) 
                
                self.qn.update(sk, yk,self.mu,self.c,ctmp,
                               self.A, Atmp)

        if fp:
            fp.close()

        # Write the final point out to a file
        self.write_output_files(self.x)
        self.print_vars(file_name=var_file)

        # Close the log file and set the pointer to None
        if self.log is not None:
            # Only close the file if we created it
            if isinstance(log_file, str):
                self.log.close()
            else:
                self.log.flush()

            self.log = None
            if self.qn:
                self.qn.log = None

        # Broadcast the termination flag as well as the results to all
        # processors for consistency
        if self.comm.size > 1:
            self.comm.bcast(-1, root=0)

            # Broad cast the components of the solution vector
            # Note that only the root processor gets this far...
            self.comm.Bcast(self.x, root=0)
            self.comm.Bcast(self.z, root=0)
            self.comm.Bcast(self.s, root=0)
            self.comm.Bcast(self.zl, root=0)
            self.comm.Bcast(self.zu, root=0)

        return
    
    def get_penalty_param(self, a_x, px, ps, g, c, 
                          x, z, s, zl, zu, perb=1e-10):
        '''
        Evaluate the penalty parameter required to make the given
        direction a descent direction.

        input:
        px, ps:  the step directions in x/s
        g:       the objective gradient
        c:       the constraints
        x:       the design variables
        s:       the slack variables

        The quadratic model of the merit function is given as follows:

        Q(px, ps) = 
        g^{T}*px - mu*S^{-1}*ps + 
        0.5*(px^{T}*H*px + mu*ps^{T}*S^{-2}*ps)
        '''

        # Compute the infeasibility
        infeas = np.sqrt(np.dot(c - s, c - s))

        # If the infeasibility is very small, set it to a small
        # parameter
        if infeas < perb:
            infeas = perb

        # Compute the denominator
        d = (np.dot(g, px) 
             - self.mu*(np.sum(ps/s) + 
                        np.sum(px/(x - self.lb)) + 
                        np.sum(-px/(self.ub - x))))

        pf = self.penalty_descent_fraction

        # Compute rho_hat using just the first-order condition
        rho_hat = d/((1.0 - pf)*a_x*infeas)

        if self.use_curvature_penalty:
            # Compute gamma = 0.5*px^{T}*H*px 
            r = np.zeros(px.shape)
            self.qn.mult(px, r)
            gamma = 0.5*(np.dot(r, px) + 
                         self.mu*(np.sum(ps**2/s**2) + 
                                  np.sum(px**2/(x - self.lb)**2) +
                                  np.sum(px**2/(self.ub - x)**2)))
            
            pf = self.penalty_descent_fraction

            # If the curvature contribution is positive, add it
            if gamma > 0.0:
                rho_hat = (d + gamma)/((1.0 - pf)*a_x*infeas)

                # If the penalty is too high, use the smaller value
                # from the first-order model only
                if rho_hat > 1e2:
                    rho_hat = d/((1.0 - pf)*a_x*infeas)
        
        return rho_hat

    def eval_merit_func(self, f, c, 
                        x, z, s, zl, zu):
        '''
        Evaluate the merit function using the data provided.
        
        intput:
        f:      the objective value
        c:      the constraints
        x:      the design variables
        z:      the Lagrange multipliers
        s:      the slack variables
        zl, zu: the lower/upper Lagrange multipliers

        returns: the value of the merit function

        Note that the merit function depends on the penalty parameter
        self.rho and the upper/lower bounds. Therefore changes to
        these member values will impact the value of the merit
        function.
        '''

        # Compute the infeasibility
        infeas = np.sqrt(np.dot(c - s, c - s))

        # Compute the merit function at the current point
        merit = (f - self.mu*(np.sum(np.log(s)) + 
                              np.sum(np.log(x - self.lb)) + 
                              np.sum(np.log(self.ub - x))) 
                 + self.rho*infeas)

        return merit

    def eval_merit_deriv(self, px, pz, ps, pzl, pzu,
                         g, c, A, x, z, s, zl, zu):
        '''
        Evaluate the derivative of the merit function:

        input:
        px, pz, ps:  the search directions along x, z and s
        pzl, pzu:    the search directions along zl, zu
        g:           the objective gradient
        c:           the constraints
        A:           the constraint Jacobian
        x:           the design variables
        z:           the dual variables
        s:           the slack variables
        zl, zu:      the variable bound multipliers

        Evaluate the derivative of the merit function and return the
        derivative and the portion of the derivative that is
        proportional to the penalty parameter self.rho. This is
        required if the directional derivative is not negative, but
        can be made so by a small increment in rho.
        '''

        # Compute the infeasibility
        infeas = np.sqrt(np.dot(c - s, c - s))

        # Compute the derivative of the merit function along the
        # px/pz/ps directions
        pmerit = (np.dot(g, px) 
                  - self.mu*(np.sum(ps/s) + 
                             np.sum(px/(x - self.lb)) + 
                             np.sum(-px/(self.ub - x))))
        if infeas > 0.0:
            pmerit += self.rho*np.dot(c - s, 
                                         np.dot(A, px) - ps)/infeas

        return pmerit

    def cubic_interp(self, x0, m0, dm0,
                     x1, m1, dm1):
        '''
        Return an x in the interval (x0, x1) that minimizes a cubic
        interpolant between two points with both function and
        derivative values.

        This method does not assume that x0 > x1. If the solution is
        not in the interval, the function returns the mid-point.
        '''

        # Compute d1
        d1 = dm0 + dm1 - 3*(m0 - m1)/(x0 - x1)

        # Check that the square root will be real in the
        # expression for d2
        if (d1**2 - dm0*dm1) < 0.0:
            if self.log: self.log.write('Cubic interpolation fail\n')
            return 0.5*(x0 + x1)

        # Compute d2
        d2 = np.sign(x1 - x0)*np.sqrt(d1**2 - dm0*dm1)

        # Evaluate the new interpolation point
        x = x1 - (x1 - x0)*(dm1 + d2 - d1)/(dm1 - dm0 + 2*d2)

        # If the new point is outside the interval, return
        # the mid point
        if x1 > x0 and (x > x1 or x < x0):
            return 0.5*(x0 + x1)
        elif x0 > x1 and (x > x0 or x < x1):
            return 0.5*(x0 + x1)

        return x

    def line_zoom(self, m_0, dm_0, 
                  px, pz, ps, pzl, pzu,
                  alph_lo, m_lo, dm_lo, f_lo, c_lo, g_lo, A_lo, 
                  alph_hi, m_hi, dm_hi, f_hi, c_hi, g_hi, A_hi,
                  c1=1e-3, c2=0.9):
        '''
        The zoom function in the line search. This can be called once
        an interval containing a point satisfying the strong Wolfe
        conditions has been located.

        The arguments to the zoom function define an interval on which
        the strong Wolfe conditions are satisfied. The alph_lo and
        alph_hi are NOT the low/high values of alpha, but refer
        instead to the low/high merit function values. The following
        conditions are required on input to this function and are
        maintained throughout:

        a) The interval bounded by alph_lo, alpha_hi contains step
        lengths that satisfy the strong Wolfe conditions

        b) alph_lo satisfies the sufficient decrease conditions and
        gives the lowest value of the merit function
        
        c) alph_hi satisifies m'(alph_lo)*(alph_hi - alph_lo) < 0
        '''

        eps = 1e-12 # A small tolerance

        for j in xrange(self.max_line_iters):
            if np.fabs(alph_hi - alph_lo) < eps:
                break

            # Interpolate between alph_lo, alph_hi to obtain a set
            # on the interval between (alph_lo, alph_hi)
            alph_j = self.cubic_interp(alph_lo, m_lo, dm_lo,
                                       alph_hi, m_hi, dm_hi)

            # Compute the line step update
            x_j = self.x + alph_j*px
            z_j = self.z + alph_j*pz
            s_j = self.s + alph_j*ps

            zl_j = self.zl + alph_j*pzl
            zu_j = self.zu + alph_j*pzu

            # Evaluate the objective and constraint
            f_j, c_j = self.objcon(x_j)
            g_j, A_j = self.gobjcon(x_j)

            # Evaluate the merit function for this line step
            m_j = self.eval_merit_func(f_j, c_j, 
                                       x_j, z_j, s_j, zl_j, zu_j)
            dm_j = self.eval_merit_deriv(px, pz, ps, pzl, pzu,
                                         g_j, c_j, A_j, 
                                         x_j, z_j, s_j, zl_j, zu_j)

            s = 'Zoom[%2d]:       a: %10.7f   m(a): %10.3e   dm: %10.3e\n'%(
                    j, alph_j, m_j, dm_j)
            if self.log: self.log.write(s)

            # Check if the sufficient decrease condition is
            # not satisfied, if so we assign alph_j -> alph_hi
            # and copy over the required function/constraint
            # values and their gradients
            if m_j > m_0 + c1*alph_j*dm_0 or m_j >= m_lo:
                alph_hi = alph_j
                m_hi = m_j
                dm_hi = dm_j

                f_hi = f_j
                g_hi = g_j
                c_hi = c_j
                A_hi = A_j
            else:
                if np.fabs(dm_j) <= -c2*dm_0:
                    # Then the strong Wolfe conditions are satisfied
                    # and we cna
                    self.x += alph_j*px
                    self.z += alph_j*pz
                    self.s += alph_j*ps

                    # Add the steps for the multipliers
                    self.zl += alph_j*pzl
                    self.zu += alph_j*pzu
                
                    # Assign the values of the objective/constraints
                    # and their gradients to the new values
                    self.obj = f_j
                    self.gobj = g_j
                    self.c = c_j
                    self.A = A_j

                    # Success
                    s = 'Line search satisfying strong Wolfe conditions'
                    if self.log: self.log.write(s)

                    return alph_j

                # The strong Wolfe conditions are not satisfied at
                # alph_j, but we do have that m(alph_j) < m(alph_lo)
                # so we need to decide whether to keep alph_lo or
                # alph_hi as the new alhp_hi. This decision is based
                # on keeping m'(alph_j)*(alph_hi - alph_lo) < 0. A
                # switch is only required if the condition is violated.
                if dm_j*(alph_hi - alph_lo) >= 0.0:
                    alph_hi = alph_lo
                    m_hi = m_lo
                    dm_hi = dm_lo
                    
                    f_hi = f_lo
                    g_hi = g_lo
                    c_hi = c_lo
                    A_hi = A_lo

                # Copy the values from alph_j to alph_lo
                alph_lo = alph_j
                m_lo = m_j
                dm_lo = dm_j

                f_lo = f_j
                g_lo = g_j
                c_lo = c_j
                A_lo = A_j      

        if alph_lo < 1e-3:
            alph_j = 1e-3
            alph_lo = 1e-3
            
            # Compute the line step update
            x_j = self.x + alph_j*px
            z_j = self.z + alph_j*pz
            s_j = self.s + alph_j*ps

            zl_j = self.zl + alph_j*pzl
            zu_j = self.zu + alph_j*pzu

            # Evaluate the objective and constraint
            f_j, c_j = self.objcon(x_j)
            g_j, A_j = self.gobjcon(x_j)

        s = 'Wolfe line search failed; too many iterations. a: %10.3e\n'%(
            alph_lo)
        if self.log: self.log.write(s)
                
        # We failed within the resource limitations..
        # Here, we just satisfy the sufficient decrease condition
        self.x += alph_lo*px
        self.z += alph_lo*pz
        self.s += alph_lo*ps

        # Add the steps for the multipliers
        self.zl += alph_lo*pzl
        self.zu += alph_lo*pzu
                
        # Assign the values of the objective/constraints
        # and their gradients to the new values
        self.obj = f_lo
        self.gobj = g_lo
        self.c = c_lo
        self.A = A_lo

        return alph_lo

    def linesearch(self, m_0, dm_0, px, pz, ps, pzl, pzu,
                   alpha_max=1.0, c1=1e-4, c2=0.95, 
                   line_search_type='Wolfe', second_order_corr=True):
        '''
        Perform a line search along the direction given by the input
        px, pz, ps, pzl, pzu.

        This function uses either a strong-Wolfe type line search
        method or a Armijo-type

        input:
        px, pz, ps: the perturbations in the x, z, and s variables
        
        rho >= ((gobj + A^{T}*lambda)^{T}*y[:n] + c(x)^{T}*y[n:])/c^{T}*c
        '''

        if self.rho == 0.0:
            s = 'Line search:  rho: %10s   m(0): %10.3e   dm: %10.3e\n'%(
                ' ', m_0, dm_0)
        else:
            s = 'Line search:  rho: %10.3e   m(0): %10.3e   dm: %10.3e\n'%(
                self.rho, m_0, dm_0)
        if self.log: self.log.write(s)

        if line_search_type == 'Armijo':
            # Perform the line search - in this case an Armijo backtracking
            # line search 
            alph_j = alpha_max

            # Keep track of whether a new point is assigned
            new_point = False

            for j in xrange(self.max_line_iters):
                # Increment the variables and evaluate the function at
                # the next point
                x_j = self.x + alph_j*px
                z_j = self.z + alph_j*pz
                s_j = self.s + alph_j*ps
                zl_j = self.zl + alph_j*pzl
                zu_j = self.zu + alph_j*pzu

                f_j, c_j = self.objcon(x_j)

                # Evaluate the merit function for this line step
                m_j = self.eval_merit_func(f_j, c_j, 
                                           x_j, z_j, s_j, zl_j, zu_j)

                s = 'Armijo[%2d]:     a: %10.7f   m(a): %10.3e\n'%(
                    j, alph_j, m_j)
                if self.log: self.log.write(s)

                # Check to see if the infeasibility increase accounted for
                # the line search failure
                soc_flag = False
                if j == 0 and second_order_corr:
                    # Compute the infeasibility
                    infeas_0 = np.sqrt(np.dot(self.c - self.s,
                                                    self.c - self.s))
                    infeas_j = np.sqrt(np.dot(c_j - s_j, 
                                                    c_j - s_j))

                    # If the infeasibility increased, but the barrier function
                    # (not including the exact penalty) decreased, perform a 
                    # second-order correction
                    if (infeas_j > infeas_0 and
                        (m_j - self.rho*infeas_j) < (m_0 - self.rho*infeas_0)):
                        soc_flag = True

                # Check the sufficient decrease condition
                if m_j < m_0 + c1*alph_j*dm_0:
                    # Update the variables
                    self.x += alph_j*px
                    self.z += alph_j*pz
                    self.s += alph_j*ps
                    self.zl += alph_j*pzl
                    self.zu += alph_j*pzu

                    if self.log: 
                        self.log.write('Sufficient decrease conditions satisfied\n')
                
                    # Assign the values of the objective/constraints
                    self.obj = f_j
                    self.c = c_j
                    self.gobj, self.A = self.gobjcon(self.x)

                    # Set the new point flag to true
                    new_point = True
                    
                    break
                elif soc_flag:
                    # Store the original value of the merit function
                    m_tmp = m_j

                    min_norm_soc = True

                    if min_norm_soc:
                        # Find the minimum norm solution of the following system:
                        # [ A, -I ][ dx ] = - [c - s] 
                        #          [ ds ] 
                        Ahat = np.zeros((self.m, self.n + self.m))
                        Ahat[:, :self.n] = self.A[:]
                        for i in xrange(self.m):
                            Ahat[i, i+self.n] = -1.0
                            
                        # Solve the normal equations
                        C = np.dot(Ahat, Ahat.T)
                        dc = np.linalg.solve(C, c_j - s_j)
                    
                        # Compute the step length
                        d = -np.dot(Ahat.T, dc)
                        dx = px + d[0:self.n]
                        ds = ps + d[self.n:]
                        dz = pz
                        dzl = pzl
                        dzu = pzu
                    else:
                        # Compute the second order correction
                        res = np.zeros(3*self.n + 2*self.m)
                    
                        # Compute the KKT conditions again, but with the second
                        # order correction
                        self.compute_kkt_res(res, x_j, z_j, 
                                             s_j, zl_j, zu_j, 
                                             self.gobj, c_j, self.A, self.mu)
                        d = np.zeros(res.shape)
                        self.apply_bfgs_pc(res, d)

                        # Extract the SOC direction
                        dx = d[:self.n] + px
                        dz = d[self.n:self.n+self.m] + pz
                        ds = d[self.n+self.m:self.n+2*self.m] + ps
                        dzl = d[self.n+2*self.m:2*self.n+2*self.m] + pzl
                        dzu = d[2*self.n+2*self.m:3*self.n+2*self.m] + pzu

                    # Evaluate the maximum steps in x, z, and the slacks
                    a_x, a_z, x_str, z_str = self.compute_max_step(dx, dz, ds,
                                                                   dzl, dzu)
                
                    # Log the step lengths
                    s = 'SOC step:       x: %10.3e      z: %10.3e %s %s\n'%(
                        a_x, a_z, x_str, z_str)
                    if self.log: self.log.write(s)

                    # Scale the direction by the step-to-boundary rule
                    dx *= a_x
                    ds *= a_x
                    dz *= a_z
                    dzl *= a_z
                    dzu *= a_z

                    # Compute the new point
                    x_j = self.x + alph_j*dx
                    z_j = self.z + alph_j*dz
                    s_j = self.s + alph_j*ds
                    zl_j = self.zl + alph_j*dzl
                    zu_j = self.zu + alph_j*dzu

                    f_j, c_j = self.objcon(x_j)

                    # Evaluate the merit function for this line step
                    m_j = self.eval_merit_func(f_j, c_j, 
                                               x_j, z_j, s_j, zl_j, zu_j)

                    # Check if the new point satisfies the sufficient 
                    # decrease conditions
                    if m_j < m_0 + c1*alph_j*dm_0:
                        # Update the variables
                        self.x += alph_j*dx
                        self.z += alph_j*dz
                        self.s += alph_j*ds
                        self.zl += alph_j*dzl
                        self.zu += alph_j*dzu

                        if self.log:
                            self.log.write('SOC: Sufficient decrease conditions satisfied\n')
                
                        # Assign the values of the objective/constraints
                        self.obj = f_j
                        self.c = c_j
                        self.gobj, self.A = self.gobjcon(self.x)
                        
                        # Set the new point flag to true
                        new_point = True

                        break
                    else:
                        if self.log: self.log.write('SOC unsuccessful\n')
                        m_j = m_tmp

                # Perform an interpolation to determine the next
                # point we should examine. This picks the step
                # length on the interval alph_j+ in [0, alpa_j].
                # This formula is from Nocedal and Wright, pg  58
                alph_j = -0.5*dm_0*alph_j**2/(m_j - m_0 - dm_0*alph_j)
                # alph_j *= 0.5

                # Stop the line search if the step is smaller than
                # the minimum length
                if alph_j < self.min_step_length:
                    s = 'Armijo line search minimum step length failure '
                    s += 'alph_j = %.4e\n'%(alph_j)
                    if self.log: self.log.write(s)
                    alph_j = self.min_step_length
                    break

            if not new_point:
                if self.log: 
                    self.log.write('Armijo line search failed\n')
                
                # Add steps for the variables
                self.x += alph_j*px
                self.z += alph_j*pz
                self.s += alph_j*ps
                
                # Add the steps for the multipliers
                self.zl += alph_j*pzl
                self.zu += alph_j*pzu

                # Assign the values of the objective/constraints
                self.obj = f_j
                self.c = c_j
                self.gobj, self.A = self.gobjcon(self.x)
        else:
            # This is a butchered form of the line search presented in
            # Nocedal and Wright. The difference is that we check the
            # alpha = alpha_max at the first iteration. If we can
            # satisfy the strong Wolfe conditions on that interval, we
            # do so, otherwise we only satisfy a sufficient decrease
            # condition... that's the reason for the damped BFGS update.
            alph_j = alpha_max

            x_j = self.x + alph_j*px
            z_j = self.z + alph_j*pz
            s_j = self.s + alph_j*ps

            zl_j = self.zl + alph_j*pzl
            zu_j = self.zu + alph_j*pzu
            
            # Evaluate the objective and constraints and their gradients
            f_j, c_j = self.objcon(x_j)
            g_j, A_j = self.gobjcon(x_j)

            # Evaluate the merit function for this line step
            m_j = self.eval_merit_func(f_j, c_j, 
                                       x_j, z_j, s_j, zl_j, zu_j)
            dm_j = self.eval_merit_deriv(px, pz, ps, pzl, pzu,
                                         g_j, c_j, A_j, 
                                         x_j, z_j, s_j, zl_j, zu_j)

            s = 'Wolfe[%2d]:      a: %10.7f   m(a): %10.3e   dm: %10.3e\n'%(
                0, alph_j, m_j, dm_j)
            if self.log: self.log.write(s)

            # The sufficient decrease conditions are violated, but
            # we can satisfy the strong Wolfe conditions on this interval
            if (m_j > m_0 + c1*alph_j*dm_0):
                if self.log:
                    self.log.write('Zoom function activated\n')

                # Call the zoom function on the specified interval
                a = self.line_zoom(m_0, dm_0, px, pz, ps, pzl, pzu,
                                   0.0, m_0, dm_0, self.obj, self.c, self.gobj, self.A,
                                   alph_j, m_j, dm_j, f_j, c_j, g_j, A_j,
                                   c1=c1, c2=c2)
                return a

            # The strong Wolf conditions are satisfied outright, we're done!
            if np.fabs(dm_j) <= -c2*dm_0:
                if self.log:
                    self.log.write('Strong Wolfe conditions satisfied\n')
                # Update to the latest value
                self.x += alph_j*px
                self.z += alph_j*pz
                self.s += alph_j*ps
                
                # Add the steps for the multipliers
                self.zl += alph_j*pzl
                self.zu += alph_j*pzu

                # Assign the values of the objective/constraints
                # and their gradients to the new values
                self.obj = f_j
                self.gobj = g_j
                self.c = c_j
                self.A = A_j
                
                return alph_j

            # The derivative of the merit function is positive, so we
            # can satisfy the strong Wolfe conditions on the given
            # interval. Note the switching of the arguments to the
            # zoom function.
            if dm_j >= 0:
                if self.log: self.log.write('Zoom function activated\n')
                # Call the zoom function on the specified interval
                a = self.line_zoom(m_0, dm_0, px, pz, ps, pzl, pzu,
                                   alph_j, m_j, dm_j, f_j, c_j, g_j, A_j,
                                   0.0, m_0, dm_0, self.obj, self.c, self.gobj, self.A,
                                   c1=c1, c2=c2)
                return a

            if self.log:
                self.log.write('Only sufficient decrease condition satisfied\n')

            # The sufficient decrease condition is satisfied, but the
            # curvature condition is not...
            self.x += alph_j*px
            self.z += alph_j*pz
            self.s += alph_j*ps

            # Add the steps for the multipliers
            self.zl += alph_j*pzl
            self.zu += alph_j*pzu
                
            # Assign the values of the objective/constraints
            # and their gradients to the new values
            self.obj = f_j
            self.gobj = g_j
            self.c = c_j
            self.A = A_j

        return alph_j

    def fgmres(self, rhs, pc=None, m=10, rtol=1e-4, atol=1e-30, 
               print_flag=False):
        '''
        Solve a linear system using flexible GMRES 

        The use of FGMRES is required since we use a flexible
        preconditioner. The use of the flexible preconditioner
        destroys the short-term Lanczos recurrence formula that is the
        basis of Krylov subspace methods for symmetric
        systems. Therefore, we use GMRES will full orthogonalizaiton.
        In addition, FGMRES is required when approximate
        Hessian-vector products are employed. Here we compute
        Hessian-vector products that include finite-difference
        matrix-vector products which introduce numerical errors.

        input:
        rhs:  the right-hand-side (np array)
        m:    the size of the preconditioner subspace
        rtol: the relative residual tolerance
        atol: the absolute residual tolerance        
        '''

        # Compute the size of the right-hand-side
        n = len(rhs)

        # Allocate the working arrays
        W = np.zeros((n, m+1))
        Z = np.zeros((n, m))

        # Allocate the Hessenberg - this allocates a full matrix
        H = np.zeros((m+1, m))

        # Allocate small arrays of size m
        res = np.zeros(m+1)
        Qsin = np.zeros(m)
        Qcos = np.zeros(m)

        # Perform the initialization: copy over rhs to W[0] and
        # normalize the result - store the entry in res[0]
        W[:, 0] = rhs[:]
        res[0] = np.sqrt(np.dot(W[:, 0], W[:, 0]))
        W[:, 0] /= res[0]

        # Store the initial residual norm
        rhs_norm = res[0]
        if self.log: 
            self.log.write('FGMRES[%2d]: %10.3e\n'%(0, res[0]))

        # Keep track of how many iterations are actually required
        niters = 0

        # Perform the matrix-vector products
        for i in xrange(m):
            # Apply the preconditioner and then compute the
            # matrix-vector product
            if pc is None:
                Z[:, i] = W[:, i]
            else:
                pc(W[:, i], Z[:, i])

            self.mult(Z[:, i], W[:, i+1])

            # Perform modified Gram-Schmidt orthogonalization
            for j in xrange(i+1):
                H[j, i] = np.dot(W[:, i+1], W[:, j])
                W[:, i+1] -= H[j, i]*W[:, j]

            # Compute the norm of the orthogonalized vector and 
            # normalize it
            H[i+1, i] = np.sqrt(np.dot(W[:, i+1], W[:, i+1]))
            W[:, i+1] /= H[i+1, i]

            # Apply the Givens rotations
            for j in xrange(i):
                h1 = H[j, i]
                h2 = H[j+1, i]
                H[j, i]   =  h1*Qcos[j] + h2*Qsin[j]
                H[j+1, i] = -h1*Qsin[j] + h2*Qcos[j]
                
            # Compute the contribution to the Givens rotation
            # for the current entry
            h1 = H[i, i]
            h2 = H[i+1, i]
            sq = np.sqrt(h1*h1 + h2*h2)
            Qcos[i] = h1/sq
            Qsin[i] = h2/sq
      
            # Apply the newest Givens rotation to the last entry
            H[i, i]   =  h1*Qcos[i] + h2*Qsin[i]
            H[i+1, i] = -h1*Qsin[i] + h2*Qcos[i]
      
            # Update the residual
            h1 = res[i]
            res[i]   =  h1*Qcos[i]
            res[i+1] = -h1*Qsin[i]
 
            # Update the iteration count
            niters += 1
            if self.log:
                self.log.write('FGMRES[%2d]: %10.3e\n'%(niters, abs(res[i+1])))

            # Perform the convergence check
            if (np.fabs(res[i+1]) < atol or 
                np.fabs(res[i+1]) < rtol*rhs_norm):
                break

        # Compute the linear combination
        for i in xrange(niters-1, -1, -1):
            for j in xrange(i+1, niters):
                res[i] -= H[i, j]*res[j]
            res[i] /= H[i, i]

        # Form the linear combination
        x = np.zeros(rhs.shape)
        for i in xrange(niters):
            x[:] += res[i]*Z[:, i]

        return x

    def test_derivatives(self, dh=1e-7):
        '''
        Test the derivative implementation
        '''
        
        # Call the wait loop on all processors except the root
        if self.comm.rank != 0:
            self.wait_loop()
            return

        # Extract the primal and dual variables
        x = self.x
        z = self.z

        # Set the perturbation vector
        w = np.random.uniform(size=len(x))
        
        obj, con = self.objcon(x)
        gobj, gcon = self.gobjcon(x)
        if self.use_hvec:
            Hw = self.hvec(x, z, w, g=gobj, A=gcon)

        obj2, con2 = self.objcon(x + dh*w)
        gobj2, gcon2 = self.gobjcon(x + dh*w)
        
        # Compute the fd products
        print ' '
        print 'Objective gradient test'
        pobj = np.dot(gobj, w)
        fd_obj = (obj2 - obj)/dh
        
        print 'FD: %20.10e An: %20.10e Err: %15.4e'%(
                fd_obj, pobj, np.fabs((fd_obj - pobj)/pobj))

        print ' '
        print 'Constraint gradient test'
        pcon = np.dot(gcon, w)
        fd_con = (con2 - con)/dh
        
        for i in xrange(len(con)):
            print 'Con[%3d] FD: %20.10e An: %20.10e Err: %15.4e'%(
                i, fd_con[i], pcon[i], np.fabs((fd_con[i] - pcon[i])/pcon[i]))

        # Test for the FD
        if self.use_hvec:
            print ' '
            print 'Hessian-vector product test'
            Hw_fd = (gobj2 - gobj)/dh
            for i in xrange(len(con)):
                Hw_fd -= z[i]*(gcon2[i, :] - gcon[i, :])/dh

            for i in xrange(len(Hw)):
                print 'Hw[%3d] FD: %20.10e An: %20.10e Err: %15.4e'%(
                    i, Hw_fd[i], Hw[i], np.fabs((Hw_fd[i] - Hw[i])/Hw[i]))

        # Broadcast the return
        if self.comm.size > 1:
            self.comm.bcast(-1, root=0)

        return

    def test_kkt_matrix(self, dh=1e-7):
        '''
        Test the KKT matrix for consistency with the implementation
        '''

        if not self.use_hvec:
            return
        
        # Call the wait loop on all processors except the root
        if self.comm.rank != 0:
            self.wait_loop()
            return

        # Record two vectors that will be the perturbation on 
        # the KKT system
        r1 = np.zeros(self.n + self.m)
        r2 = np.zeros(self.n + self.m)
        w = np.random.uniform(size=(self.n + self.m))
        out = np.zeros(self.n + self.m)

        # Evaluate the KKT system
        self.obj, self.c = self.objcon(self.x)
        self.gobj, self.A = self.gobjcon(self.x)
        self.compute_kkt_res(r1, self.x, self.gobj, 
                             self.c, self.A, self.z, self.mu)
        
        # Compute the matrix-vector product
        self.set_up_mat_pc(self.x, self.z, self.s, self.A)
        self.mult(w, out)

        self.x += dh*w[:self.n]
        self.z -= dh*w[self.n:]

        # Evaluate the KKT system
        obj, c = self.objcon(self.x)
        gobj, A = self.gobjcon(self.x)
        self.compute_kkt_res(r2, self.x, gobj, c, A, self.z, self.mu)

        # Note that the negative is here since we compute the negative
        # of the KKT residuals in compute_kkt_res
        r2 = -(r2 - r1)/dh

        print ' '
        print 'KKT matrix-vector product check'
        for i in xrange(self.n + self.m):
            print 'Hvec[%3d] FD: %20.10e An: %20.10e Err: %15.4e'%(
                i, r2[i], out[i], np.fabs((r2[i] - out[i])/r2[i]))

        return
