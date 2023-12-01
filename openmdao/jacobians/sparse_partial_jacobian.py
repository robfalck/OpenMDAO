"""Define the DictionaryJacobian class."""
import numpy as np
import scipy.sparse
import scipy.sparse as sp

from openmdao.jacobians.jacobian import Jacobian
from openmdao.core.constants import INT_DTYPE
from openmdao.utils.indexer import Slicer


class SparsePartialJacobian(Jacobian):
    """
    No global <Jacobian>; use dictionary of user-supplied sub-Jacobians.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _iter_keys : list of (vname, vname) tuples
        List of tuples of variable names that match subjacs in the this Jacobian.
    """

    def __init__(self, system, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(system, **kwargs)

        slicer = Slicer()

        var_rel2meta = system._var_rel2meta
        var_rel_names = system._var_rel_names
        declared_resids = system._declared_residuals
        declared_partials = system._declared_partials

        size_outputs = 0

        input_meta = {rel_name: {} for rel_name in var_rel_names['input']}
        output_meta = {rel_name: {} for rel_name in var_rel_names['output']}
        resid_meta = {resid_name: {} for resid_name in declared_resids.keys()}

        size_inputs = 0
        for rel_name in var_rel_names['input']:
            input_meta = var_rel2meta[rel_name]
            input_size = input_meta['size']
            input_meta['cols'] = size_inputs + np.arange(input_size, dtype=int)
            size_inputs += input_size

        start_row = 0
        for rel_name in var_rel_names['output']:
            output_meta = var_rel2meta[rel_name]
            output_size = output_meta['size']
            size_outputs += output_size
            output_meta['rows'] = start_row + np.arange(output_size, dtype=int)

        print(list(var_rel2meta.keys()))

        start_row = 0
        for resid_name, meta in declared_resids.items():
            resid_size = np.prod(meta['shape'], dtype=int)
            meta['size'] = resid_size
            meta['rows'] = start_row + np.arange(resid_size, dtype=int)
            start_row += resid_size

        self._dok_matrix = scipy.sparse.dok_matrix((size_outputs, size_inputs), dtype=np.float64)

        # Insert the declared partials into the total partial jacobian.
        # The total partial jacobian is stored internally as a scipy.sparse.dok_matrix.
        for (resid_name, input_name), meta in declared_partials.items():
            resid_meta = declared_resids[resid_name]
            input_meta = var_rel2meta[input_name]
            cols = input_meta['cols']
            rows = resid_meta['rows']
            input_size = input_meta['size']
            resid_size = resid_meta['size']

            if 'rows' in meta and 'cols' in meta:
                # provided as sparse
                # r and c are given relative to the subjac, need to convert to the total jac
                r = meta['rows']
                c = meta['cols']
            else:
                # provided as dense
                r, c = np.mgrid[:resid_size, :input_size]

            self._dok_matrix[rows[r], cols[c]] = meta['val']

        print(self._dok_matrix.todense())
        print(list(declared_resids.keys()))
        print(list(var_rel2meta.keys()))

        print(self['x', 'res_a'].todense())
        print(self['x', 'res_b'].todense())
        print(self['resid_res_a', 'res_a'].todense())
        print(self['resid_res_a', 'res_b'].todense())
        print(self['resid_res_b', 'res_a'].todense())
        print(self['resid_res_b', 'res_b'].todense())

        exit(0)

    def __getitem__(self, key):
        """
        Get sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        declared_resids = self._system()._declared_residuals
        var_rel2meta = self._system()._var_rel2meta
        var_rel_names = self._system()._var_rel_names

        if key[1] in var_rel_names['input']:
            c = var_rel2meta[key[1]]['cols']
        else:
            raise KeyError(f'{self.msginfo}: \'wrt\' variable \'{key[1]}\' not found.')

        if key[0] in declared_resids:
            r = declared_resids[key[0]]['rows']
        elif key[0] in var_rel_names['output']:
            r = var_rel2meta[key[0]]['rows']
        else:
            raise KeyError(f'{self.msginfo}: \'of\' variable \'{key[0]}\' not found.')

        rows, cols = np.mgrid[r[0]:r[-1]+1, c[0]:c[-1]+1]
        return self._dok_matrix[rows, cols]

    def _iter_abs_keys(self, system):
        """
        Iterate over subjacs keyed by absolute names.

        This includes only subjacs that have been set and are part of the current system.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.

        Returns
        -------
        list
            List of keys matching this jacobian for the current system.
        """
        if self._iter_keys is None:
            subjacs = self._subjacs_info
            keys = []
            for res_name in system._var_abs2meta['output']:
                for type_ in ('output', 'input'):
                    for name in system._var_abs2meta[type_]:
                        key = (res_name, name)
                        if key in subjacs:
                            keys.append(key)

            self._iter_keys = keys

        return self._iter_keys

    def _apply(self, system, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        fwd = mode == 'fwd'
        d_res_names = d_residuals._names
        d_out_names = d_outputs._names
        d_inp_names = d_inputs._names

        if not d_out_names and not d_inp_names:
            return

        rflat = d_residuals._abs_get_val
        oflat = d_outputs._abs_get_val
        iflat = d_inputs._abs_get_val
        subjacs_info = self._subjacs_info
        is_explicit = system.is_explicit()
        randgen = self._randgen

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            for abs_key in self._iter_abs_keys(system):
                res_name, other_name = abs_key
                if res_name in d_res_names:
                    if other_name in d_out_names:
                        # skip the matvec mult completely for identity subjacs
                        if is_explicit and res_name is other_name:
                            if fwd:
                                val = rflat(res_name)
                                val -= oflat(other_name)
                            else:
                                val = oflat(other_name)
                                val -= rflat(res_name)
                            continue
                        if fwd:
                            left_vec = rflat(res_name)
                            right_vec = oflat(other_name)
                        else:
                            left_vec = oflat(other_name)
                            right_vec = rflat(res_name)
                    elif other_name in d_inp_names:
                        if fwd:
                            left_vec = rflat(res_name)
                            right_vec = iflat(other_name)
                        else:
                            left_vec = iflat(other_name)
                            right_vec = rflat(res_name)
                    else:
                        continue

                    subjac_info = subjacs_info[abs_key]
                    if randgen:
                        subjac = self._randomize_subjac(subjac_info['val'], abs_key)
                    else:
                        subjac = subjac_info['val']
                    rows = subjac_info['rows']
                    if rows is not None:  # our homegrown COO format
                        linds, rinds = rows, subjac_info['cols']
                        if not fwd:
                            linds, rinds = rinds, linds
                        if self._under_complex_step:
                            # bincount only works with float, so split into parts
                            prod = right_vec[rinds] * subjac
                            left_vec[:].real += np.bincount(linds, prod.real,
                                                            minlength=left_vec.size)
                            left_vec[:].imag += np.bincount(linds, prod.imag,
                                                            minlength=left_vec.size)
                        else:
                            left_vec[:] += np.bincount(linds, right_vec[rinds] * subjac,
                                                       minlength=left_vec.size)

                    else:
                        if fwd:
                            left_vec += subjac.dot(right_vec)
                        else:  # rev
                            subjac = subjac.transpose()
                            left_vec += subjac.dot(right_vec)
