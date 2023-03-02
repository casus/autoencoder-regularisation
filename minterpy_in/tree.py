# -*- coding:utf-8 -*-
import numpy as np

from minterpy.global_settings import INT_DTYPE
#from minterpy.newt_eval_fast import newt_eval
from minterpy.utils import leja_ordered_values, gamma_lp


# TODO refactor code. more clever way? creating tree element directly?
def insert_new_element(tree, element_id):
    # TODO raise ValueError if already present to detect bugs
    if element_id not in tree:
        tree[element_id] = TreeElement()
    return tree


class TreeElement:
    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['split', 'parent', 'child', 'length', 'pro_number', 'project', 'depth']

    def __init__(self, split=None, parent=None, child=None, length=2, pro_number=None, project=None, depth=0):

        # default arguments must not be mutable
        if split is None:
            split = [-1, -1]
        if parent is None:
            parent = [-1, -1]
        if child is None:
            child = [-1, -1]
        if pro_number is None:
            pro_number = []
        if project is None:
            project = []

        # TODO copy really required?
        self.split = np.array(split).copy()
        self.parent = np.array(parent).copy()
        self.child = np.array(child).copy()
        self.length = length
        self.pro_number = pro_number
        self.project = project
        self.depth = depth


class MultiIndicesTree:
    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['m', 'n', 'lp_degree', 'tree', 'grid_points', 'grid_values', 'gamma', 'N']

    def __init__(self, m: int, n: int, lp_degree: float = 2.0):
        # assert m > 1, "dimension m must be larger than 1"  # FIXME dimension 1 should be allowed
        # TODO redundancy: Interpolator and Transformer classes store the same parameters
        self.m = m
        self.n = n
        self.lp_degree = lp_degree

        # multi indices, shape: (m,N)
        # gamma = gamma_lp = np.stack([np.tile(np.arange(0,m+1), m+1), np.repeat(np.arange(0,m+1), m+1)])
        # TODO assert the completeness of gamma. otherwise e.g. gradient computation fails!
        self.gamma: np.ndarray = gamma_lp(self.m, self.n, np.zeros((self.m, 1)), np.zeros((self.m, 1)), self.lp_degree)
        # TODO specify already during construction!
        self.gamma = self.gamma.astype(INT_DTYPE)
        # TODO lowercase variable names
        m, N = self.gamma.shape
        assert m == self.m, 'multi index dimensions do not match'
        self.N: int = N

        leja_values = self._get_leja_vals()
        # TODO required to store like this? only signs are shifted! redundancy.
        self.grid_values: np.ndarray = np.zeros((self.m, self.n + 1))
        for i in range(self.m):
            self.grid_values[i] = (-1) ** (i + 1) * leja_values

        # Chebyshev grid
        self.grid_points: np.ndarray = self.__gen_points_lp(self.m, self.N, self.grid_values, self.gamma)
        m, N = self.grid_points.shape
        assert m == self.m, 'grid point dimensions do not match'
        assert N == self.N, 'grid point dimensions do not match'

        # TODO remove double nesting: interpolator.tree.tree
        self.tree = None
        self.build_tree()

    def _get_leja_vals(self):
        leja_values = leja_ordered_values(self.n)
        return leja_values

    def build_tree(self):

        # TODO efficient tree data structure
        tree = {}
        insert_new_element(tree, (1, 1))
        tree[(1, 1)].depth = 0

        # TODO two steps necessary?
        tree = self.__gen_tree_lp(self.m, self.gamma, tree, 1, 1)
        tree = self.__pro_lp(self.m, self.n, self.lp_degree, self.gamma, tree, 1, 1)
        self.tree = tree

    def __gen_points_lp(self, m, N, Points, Gamma):
        PP = np.zeros((m, N))
        for i in range(N):
            for j in range(m):
                # TODO remove int casting. exponents/multi indices should be int already
                PP[j, i] = Points[j, int(Gamma[j, i])]
        return PP

    def __gen_tree_lp(self, m, Gamma, tree, I, J):
        """
        Generates tree with index properties...
        child [i,j] is J of children of [N0,N1] splitting
        parent [i,1] or [i,2] gives parent J and 1,2 specifies of [N0,N1]
        splitting
        """
        kkkk, N = Gamma.shape
        depth = tree[(1, 1)].depth
        for i in range(N):
            if abs(Gamma[m - 1, i] - Gamma[m - 1, 0]) == 1:
                N0 = i
                N1 = N - N0
                break
            N0 = N
            N1 = 0
        tree = insert_new_element(tree, (I, J))
        # FIXME Local variable 'N0' might be referenced before assignment
        tree[(I, J)].split = np.array([N0, N1]).copy()

        if I > depth:
            tree = insert_new_element(tree, (I, 1))
            tree[(I, 1)].length = 0
            tree[(1, 1)].depth = I

        k = tree[(I, 1)].length + 1

        if m > 2 and N0 > 1:
            tree = insert_new_element(tree, (I, 1))
            tree = insert_new_element(tree, (I, J))
            tree = insert_new_element(tree, (I + 1, k))

            tree[(I, 1)].length = k
            tree[(I, J)].child[0] = k
            tree[(I + 1, k)].parent = np.array([J, 1]).copy()

            Gamma0 = Gamma[:, 0:N0]
            tree = self.__gen_tree_lp(m - 1, Gamma0, tree, I + 1, k)
            k = k + 1
        elif m == 2 and N0 > 1:
            tree = insert_new_element(tree, (I, 1))
            tree = insert_new_element(tree, (I, J))
            tree = insert_new_element(tree, (I + 1, k))

            tree[(I, 1)].length = k
            tree[(I, J)].child[0] = k
            tree[(I + 1, k)].parent = np.array([J, 1]).copy()

            tree[(I + 1, k)].split = np.array([N0, 0]).copy()
            tree[(I + 1, k)].child = np.array([0, 0]).copy()
            k = k + 1
        else:
            tree[(I, J)].child[0] = 0

        if N1 > 1:
            tree = insert_new_element(tree, (I, 1))
            tree = insert_new_element(tree, (I, J))
            tree = insert_new_element(tree, (I + 1, k))

            tree[(I, 1)].length = k
            tree[(I, J)].child[1] = k
            tree[(I + 1, k)].parent = np.array([J, 2]).copy()

            Gamma1 = Gamma[:, N0:N]
            tree = self.__gen_tree_lp(m, Gamma1, tree, I + 1, k)
            out = tree.copy()  # FIXME return? copy?
        else:
            tree[(I, J)].child[1] = 0

        return tree

    def __pro_lp(self, m, n, p, Gamma, tree, I, J):
        N0 = tree[(I, J)].split[0]
        N1 = tree[(I, J)].split[1]

        Gamma0 = Gamma[:, 0:N0].copy()
        Gamma1 = Gamma[:, N0:N0 + N1].copy()
        Project = {}
        Project[(1, 1)] = [0, 0, 0]

        count = 1
        I0 = I
        J0 = J
        d = tree[(I0, J0)].split[0]
        S = np.array([d])

        if (tree[(I0, J0)].split[1] > 0):
            J0 = tree[(I0, J0)].child[1]
            I0 = I0 + 1

            for i in range(d):
                if (J0 > 0 and tree[(I, J)].split[1] > 0):
                    count = count + 1
                    S = np.insert(S, i + 1, tree[(I0, J0)].split[0])
                    # S = [S, tree[(I0,J0)].split[0]]
                else:
                    break

                J0 = tree[(I0, J0)].child[1]
                I0 = I0 + 1

        tree[(I, J)].pro_number = count - 1
        k1 = N1

        for i in range(count - 1):  # BUG? range(count-2)
            Gamma0[m - 1, :] = Gamma0[m - 1, :] + 1
            split1 = S[i]
            split2 = S[i + 1]
            if (split1 > split2):
                Pro = np.zeros((3 + split1 - split2))
                l = 0
                for j in range(split1):
                    norm = np.linalg.norm(Gamma0[:, j], p)
                    if (norm > n):
                        l += 1
                        dbg = l + 3 - 1
                        dbg2 = len(Pro)
                        Pro[l + 3 - 1] = j + 1  # BUG? Pro[.] = j+1 in case of Matlab implementation
                Pro[0] = split1 - split2
                Pro[1] = split1
                Pro[2] = split2
                Project[(1, i + 1)] = Pro

            elif (split1 == split2):
                Project[(1, i + 1)] = [0, split1, split2]

            Gamma0 = Gamma1[:, 0:split2]
            Gamma1 = Gamma1[:, split2:k1]
            k1 -= split2

        tree[(I, J)].project = Project

        if (m > 2 and count > 1):
            tree = self.__pro_lp(m - 1, n, p, Gamma[:, 0: N0], tree, I + 1, tree[(I, J)].child[0])
            out = self.__pro_lp(m, n, p, Gamma[:, N0:], tree, I + 1, tree[(I, J)].child[1])
        elif (m > 2 and count == 1):
            out = self.__pro_lp(m - 1, n, p, Gamma[:, 0:N0], tree, I + 1, tree[(I, J)].child[0])
        elif (m == 2 and (count - 1) > 1):
            out = self.__pro_lp(m, n, p, Gamma[:, N0:], tree, I + 1, tree[(I, J)].child[1])  # Replaced N0 + 1 by N0
        else:
            out = tree

        return out

    # API (external, clean, TODO with input verification)
    # TODO API: eval should be defined for a polynomial, not a "tree"
    # NOTE: x is possibly a list of points!
    def eval(self, x, coeffs_newton, verify_input: bool = False):
        return newt_eval(x, coeffs_newton, self.gamma, self.grid_values, verify_input)

        # legacy recursive evaluation implementation:
        # coeffs_shape = coeffs_newton.shape
        # assert len(coeffs_shape) == 1, 'coefficients should be given as a 1D ndarray'
        # assert coeffs_shape[0] == self.N, \
        #     f'the coefficients are expected to have length {self.N} (actual length {coeffs_shape[0]})'
        # # TODO the "gamma" parameter is just a placeholder for exponents
        # #  (called "gamma_placeholder" in the Interpolator class)
        # gamma_placeholder = np.zeros(self.m, dtype=np.int_)
        # assert np.issubdtype(gamma_placeholder.dtype, np.int_), \
        #     'the parameter "gamma" (exponent placeholder) should be given in integer dtype (used for indexing)'
        # return self._eval_lp(x, coeffs_newton, self.m, self.n, self.N, gamma_placeholder,
        #                      self.grid_values, self.lp_degree, 1, 1)

    # internal function signature for recursion
    # ATTENTION: gamma is being manipulated during query -> fresh copy required every time
    # NOTE: all other parameters need not be copied!
    def _eval_lp(self, x, C, m, n, N, gamma, Points, p, I, J):

        # FIXME Local variable 'gamma1' might be referenced before assignment
        # FIXME Local variable 'N0' might be referenced before assignment
        if m > 1 and J > 0:
            N0 = self.tree[(I, J)].split[0]
            N1 = self.tree[(I, J)].split[1]
            gamma1 = gamma
            gamma1[m - 1] = gamma1[m - 1] + 1
        elif m == 1:
            N0 = 1
            N1 = N - 1

        # TODO values for gamma1, N0, N1.
        if N > 0:

            # N0, N1.. number of elements running from 0 to ... N-1
            C0 = C[0:N0]
            C1 = C[N0:N]

            if N0 > 0 and N1 > 1 and m > 2:  # N0>0 && N1>1 && m>2
                tree_child1 = self.tree[(I, J)].child[0]  # child 1
                tree_child2 = self.tree[(I, J)].child[1]  # child 2
                o1 = self._eval_lp(x, C0, m - 1, n, N0, gamma1.copy(), Points, p, I + 1, tree_child1)
                o2 = x[m - 1] - Points[m - 1, gamma1[m - 1] - 1]
                o3 = self._eval_lp(x, C1, m, n - 1, N1, gamma1.copy(), Points, p, I + 1, tree_child2)
                out = o1 + o2 * o3

            elif N0 > 0 and N1 == 1 and m > 2:  # N0>0 && N1==1 && m>2
                tree_child1 = self.tree[(I, J)].child[0]  # child 1
                out = self._eval_lp(x, C0, m - 1, n, N0, gamma1.copy(), Points, p, I + 1, tree_child1) + (
                        x[m - 1] - Points[m - 1, gamma1[m - 1] - 1]) * C1[0]
            elif N0 > 0 and N1 == 0 and m > 2:  # N0>0 && N1==0 && m>2
                tree_child1 = self.tree[(I, J)].child[0]  # child 1
                out = self._eval_lp(x, C0, m - 1, n, N0, gamma1.copy(), Points, p, I + 1, tree_child1)
            elif N0 > 0 and N1 > 1 and m == 2:  # N0>0 && N1>1 && m ==2
                oneD = self._eval_lp(x, C0, 1, n, N0, np.array([0]), Points, p, 0, 0)
                tree_child2 = self.tree[(I, J)].child[1]
                o2 = (x[m - 1] - Points[m - 1, gamma1[m - 1] - 1])
                o3 = self._eval_lp(x, C1, m, n - 1, N1, gamma1.copy(), Points, p, I + 1, tree_child2)
                out = oneD + o2 * o3
            elif N0 > 0 and N1 == 1 and m == 2:  # N0>0 && N1==1 && m ==2
                oneD = self._eval_lp(x, C0, 1, n, N0, np.array([0]), Points, p, 0, 0)
                o2 = (x[m - 1] - Points[m - 1, gamma1[m - 1] - 1])
                o3 = C1[0]
                out = oneD + o2 * o3
            elif N0 > 0 and N1 == 0 and m == 2:  # elseif N0>0 && N1==0 && m ==2
                oneD = self._eval_lp(x, C0, 1, n, N0, np.array([0]), Points, p, 0, 0)
                out = oneD
            elif m == 1 and N1 > 1:  # elseif m ==1 && N1>1
                out_1 = C0[0]
                out_2 = (x[0] - Points[0, gamma[0]])
                out_3 = self._eval_lp(x, C1, 1, n - 1, N1, np.array([gamma[0] + 1]), Points, p, 0, 0)
                out = out_1 + out_2 * out_3
            elif m == 1 and N1 == 1:  # elseif m ==1 && N1==1
                out = C0[0] + (x[0] - Points[0, gamma[0]]) * C1[0]
            elif m == 1 and N1 == 0:  # elseif m ==1 && N1==0
                out = C0[0]
            elif N0 == 0 and N1 > 0:  # elseif N0==0 && N1>0
                tree_child2 = self.tree[(I, J)].child[1]
                out = (x[m - 1] - Points[m - 1, gamma1[m - 1] - 1]) * self._eval_lp(x, C1, m, n - 1, N1, gamma1,
                                                                                    Points, p, I + 1, tree_child2)
            # TODO else
        else:
            out = C[0]  # TODO default?

        return out  # FIXME Local variable 'out' might be referenced before assignment
