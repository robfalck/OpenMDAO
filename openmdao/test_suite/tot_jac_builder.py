"""
A tool to make it easier to investigate coloring of jacobians with different sparsity structures.
"""

import numpy as np
from scipy.sparse import coo_matrix

import openmdao.api as om
from openmdao.utils.coloring import _compute_coloring
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.devtools.debug import compare_jacs


class TotJacBuilder(object):
    def __init__(self, nrows, ncols):
        self.J = np.zeros((nrows, ncols), dtype=bool)
        self.coloring = None

    def add_random_points(self, npoints):
        nrows, ncols = self.J.shape

        zro = self.J == False
        flat = self.J[zro].flatten()
        flat[:npoints] = True
        np.random.shuffle(flat)
        self.J[zro] = flat

    def add_row(self, idx, density=1.0):
        self.add_block(self.create_row(density=density), idx, 0)

    def add_col(self, idx, density=1.0):
        self.add_block(self.create_col(density=density), 0, idx)

    def create_row(self, density=1.0):
        return self.create_block((1, self.J.shape[1]), density=density)

    def create_col(self, density=1.0):
        return self.create_block((self.J.shape[0], 1), density=density)

    def create_block(self, shape, density=1.0):
        if density == 1.0:
            return np.ones(shape, dtype=bool)
        else:
            rows, cols = shape
            num = int((rows * cols) * density)
            vec = np.zeros(int(rows * cols), dtype=bool)
            vec[:num] = True
            np.random.shuffle(vec)
            return vec.reshape(shape)

    def add_block(self, block, start_row, start_col):
        rows, cols = block.shape
        self.J[start_row:start_row + rows, start_col:start_col + cols] = block

    def add_block_diag(self, shapes, start_row, start_col, density=1.0):
        row_idx = start_row
        col_idx = start_col

        for shape in shapes:
            self.add_block(self.create_block(shape, density=density), row_idx, col_idx)
            row_idx += shape[0]
            col_idx += shape[1]

    def color(self, mode='auto', fname=None, direct=True):
        self.coloring = _compute_coloring(self.J, mode, direct=direct)
        if self.coloring is not None and fname is not None:
            self.coloring.save(fname)
        return self.coloring

    def show(self):
        try:
            self.coloring.display_bokeh(show=True)
        except Exception:
            print("Bokeh not available, using ASCII display.")
            self.coloring.display_txt()

        maxdeg_fwd = np.max(np.count_nonzero(self.J, axis=1))
        maxdeg_rev = np.max(np.count_nonzero(self.J, axis=0))

        print("Shape:", self.J.shape)
        print("Density:", np.count_nonzero(self.J) / self.J.size)
        print("Max degree (fwd, rev):", maxdeg_fwd, maxdeg_rev)

        self.coloring.summary()

    def shuffle_rows(self):
        np.random.shuffle(self.J)

    def density_info(self):
        J = self.J
        density = np.count_nonzero(J) / J.size
        row_density = np.count_nonzero(J, axis=1) / J.shape[1]
        max_row_density = np.max(row_density)
        n_dense_rows = row_density[row_density == 1.0].size
        col_density = np.count_nonzero(J, axis=0) / J.shape[0]
        max_col_density = np.max(col_density)
        n_dense_cols = col_density[col_density == 1.0].size
        return density, max_row_density, n_dense_rows, max_col_density, n_dense_cols

    @staticmethod
    def make_blocks(num_blocks, min_shape, max_shape):
        shapes = []
        row_size = col_size = 0
        min_rows, min_cols = min_shape
        max_rows, max_cols = max_shape

        for b in range(num_blocks):
            nrows = np.random.randint(min_rows, max_rows + 1)
            ncols = np.random.randint(min_cols, max_cols + 1)
            shapes.append((nrows, ncols))
            row_size += nrows
            col_size += ncols

        return shapes, row_size, col_size

    @staticmethod
    def make_jac(n_dense_rows=0, row_density=1.0, n_dense_cols=0, col_density=1.0,
                 n_blocks=0, min_shape=(1,1), max_shape=(2,2), n_random_pts=0):
        if n_blocks > 0:
            shapes, nrows, ncols = TotJacBuilder.make_blocks(n_blocks, min_shape, max_shape)
            builder = TotJacBuilder(nrows + n_dense_rows, ncols + n_dense_cols)
            builder.add_block_diag(shapes, n_dense_rows, n_dense_cols)
        else:
            nrows, ncols = (100, 50)
            builder = TotJacBuilder(nrows, ncols)

        # dense rows
        for row in range(n_dense_rows):
            builder.add_row(row, density=row_density)

        # dense cols
        for col in range(n_dense_cols):
            builder.add_col(col, density=col_density)

        builder.add_random_points(n_random_pts)

        return builder

    @staticmethod
    def eisenstat(n):
        """
        Return a builder containing an Eisenstat's example Jacobian of size n+1 x n.

        Should be colorable with n/2 + 2 colors using bidirectional coloring.

        The columns in Eisenstat's example are pairwise structurally nonorthogonal,
        so a fwd directional coloring would require n groups.
        """
        assert n >= 6, "Eisenstat's example must have n >= 6."
        assert n % 2 == 0, "Eisenstat's example must have even 'n'."

        D1 = np.eye(n // 2, dtype=int)
        D2 = np.eye(n // 2, dtype=int)
        D3 = np.eye(n // 2, dtype=int)
        B = np.ones((n // 2, n // 2), dtype=int)
        idxs = np.arange(n // 2, dtype=int)
        B[idxs, idxs] = 0
        C = np.ones((1, n // 2), dtype=int)
        O = np.zeros((1, n // 2), dtype=int)

        A1 = np.hstack([D1, D2])
        A2 = np.vstack([np.hstack([C, O]), np.hstack([D3, B])])

        A = np.vstack([A1, A2])

        builder = TotJacBuilder(n + 1, n)
        builder.J[:, :] = A

        return builder


def rand_jac(minrows=(1, 10), mincols=(1, 10)):
    rnd = np.random.randint
    minr = rnd(*minrows)
    minc = rnd(*mincols)

    return  TotJacBuilder.make_jac(n_dense_rows=rnd(5), row_density=np.random.rand(),
                                   n_dense_cols=rnd(5), col_density=np.random.rand(),
                                   n_blocks=rnd(3,8),
                                   min_shape=(minr,minc),
                                   max_shape=(minr+rnd(10),minc+rnd(10)),
                                   n_random_pts=rnd(15))


class SparsityComp(om.ExplicitComponent):
    """
    A simple component that multiplies a sparse matrix by an input vector.

    The sparsity structure is defined by the 'sparsity' argument, and the data values are
    just the (index + 1) of the nonzeros in the sparsity structure.

    This component is used to test the coloring of the total jacobian.  A Problem is set up
    with a model containing only this component, and the total jacobian is computed with and
    without coloring.  The two jacobians are compared to ensure they are the same.

    Parameters
    ----------
    sparsity : ndarray or coo_matrix
        Sparsity structure to be tested.

    Attributes
    ----------
    sparsity : coo_matrix or ndarray
        Dense or sparse version of the sparsity structure.
    """
    def __init__(self, sparsity, **kwargs):
        super(SparsityComp, self).__init__(**kwargs)
        if isinstance(sparsity, np.ndarray):
            self.sparsity = coo_matrix(sparsity)
        else:
            self.sparsity = sparsity.tocoo()

        self.sparsity.data = np.arange(1, self.sparsity.data.size + 1)

    def setup(self):
        self.add_input('x', shape=self.sparsity.shape[1])
        self.add_output('y', shape=self.sparsity.shape[0])

        self.declare_partials('y', 'x', rows=self.sparsity.row, cols=self.sparsity.col)

    def compute(self, inputs, outputs):
        outputs['y'] = self.sparsity.dot(inputs['x'])

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = self.sparsity.data


def check_sparsity_tot_coloring(sparsity, direct=True, tolerance=1e-15, tol_type='rel', mode='auto'):
    """
    Check total derivatives of a top level SparsityComp with and without total coloring.

    Parameters
    ----------
    sparsity : ndarray or coo_matrix
        Sparsity structure to be tested.
    direct : bool
        If True, use the direct method to compute the column adjacency matrix when bidirectional
        coloring, else use the substitution method.
    """
    import sys

    # compute totals without coloring
    p = om.Problem()
    model = p.model
    model.add_subsystem('comp', SparsityComp(sparsity))
    model.add_design_var('comp.x')
    model.add_constraint('comp.y')
    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.setup(mode=mode)
    p.run_model()
    J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

    # compute totals with coloring
    p = om.Problem()
    model = p.model
    model.add_subsystem('comp', SparsityComp(sparsity))
    model.add_design_var('comp.x')
    model.add_constraint('comp.y')
    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.declare_coloring(direct=direct)
    p.setup(mode=mode)
    p.run_model()
    Jcolor = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

    # make sure totals match for both cases
    try:
        assert_near_equal(J, Jcolor, tolerance=tolerance, tol_type=tol_type)
    except Exception:
        diff = np.abs(J - Jcolor)
        mask = diff <= tolerance
        diff[mask] = 0.0
        if p.driver._coloring_info.coloring is not None and not p.driver._coloring_info._failed:
            print(p.driver._coloring_info.coloring, file=sys.stderr)

        with np.printoptions(linewidth=999, threshold=1000):
            print("Good J\n", J, file=sys.stderr)
            drows, dcols = np.nonzero(diff)
            print("J shape", J.shape, file=sys.stderr)
            print("J diff\n", list(zip(drows, dcols)), file=sys.stderr)
        raise


if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eisenstat",
                        help="Build an Eisenstat's example matrix of size n+1 x n.",
                        action="store", type=int, default=-1, dest="eisenstat")
    parser.add_argument("-m", "--mode", type=str, dest="mode",
                        help="Direction of coloring (default is auto).",
                        default="auto")
    parser.add_argument('-s', '--save', dest="save", default=None,
                        help="Output file for jacobian so it can be reloaded and colored using"
                        " various methods for comparison.")
    parser.add_argument('-l', '--load', dest="load", default=None,
                        help="Input file for jacobian so it can be reloaded and colored using"
                        " various methods for comparison.")

    options = parser.parse_args()

    if options.load is not None:
        with open(options.load, "rb") as f:
            builder = pickle.load(f)
    elif options.eisenstat > -1:
        builder = TotJacBuilder.eisenstat(options.eisenstat)
    else:  # just do a random matrix
        builder = rand_jac()

    builder.color(options.mode)
    builder.show()

    if options.save is not None:
        with open(options.save, "wb") as f:
            pickle.dump(builder, f)
