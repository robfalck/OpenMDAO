"""Define the DictionaryJacobian class."""
import numpy as np
import scipy.sparse as sp

from openmdao.jacobians.jacobian import Jacobian
from openmdao.core.constants import INT_DTYPE


class SparsePartialJacobian(Jacobian):
    """
    A class for component Jacobians that store all data within a single scipy sparse jacobian.

    This Jacobian is used for components where add_residual is used. In the case of residual assignment,
    `partials[of=resid_name, wrt=input_name]` assigns the jacobian using row indices associated with
    the redidual name. When retrieving partials, `partials[of=output_name, wrt=input_name]` uses row
    indices associated with the output names.

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
    _data : scipy.sparse.dok_array
        A scipy.sparse matrix in dictionary-of-keys format. This stores the entire
        jacobian of a component in a single data structure that allows us to request data using
        a combination of input name and either output name or residual name.
    """

    def __init__(self, system, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(system, **kwargs)
        self._iter_keys = None
        jac_shape = (np.sum(system._var_sizes['output']), np.sum(system._var_sizes['input']))
        self._data = sp.dok_array(jac_shape)

    def _get_residual_rows(self, of):
        """
        Return the rows of the jacobian associated with the named residual or output specified by 'of'.

        Parameters
        ----------
        of : str
            A named residual or output of this system.

        Returns
        -------
        rows : tuple of int
            A tuple of ints providing the rows in the jacobian that belong to the specified residual or output.
        """
        system = self._system()
        if of in system._declared_residuals:
            # of is a residual

        elif of in system._var_abs2meta['output']:
            # of is an output

        else:
            raise KeyError(f'{system.msginfo}: {of} is not a named residual or output of this component')



    def __setitem__(self, key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Output/residual and input name pair of sub-Jacobian. The first string in this sequence
            may be either a component's output name or residual name.
        subjac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        abs_key = self._get_abs_key(key)
        if abs_key is None:
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

        # You can only set declared subjacobians.
        if abs_key not in self._subjacs_info:
            msg = '{}: Variable name pair ("{}", "{}") must first be declared.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

        subjacs_info = self._subjacs_info[abs_key]

        if issparse(subjac):
            self._data['val'] = subjac
        else:
            rows = subjacs_info['rows']

            if rows is None:
                # Dense subjac
                subjac = np.atleast_2d(subjac)
                if subjac.shape != (1, 1):
                    shape = self._abs_key2shape(abs_key)
                    subjac = subjac.reshape(shape)

                subjacs_info['val'][:] = subjac

            else:
                try:
                    subjacs_info['val'][:] = subjac
                except ValueError:
                    subjac = np.atleast_1d(subjac)
                    msg = '{}: Sub-jacobian for key {} has the wrong shape ({}), expected ({}).'
                    raise ValueError(msg.format(self.msginfo, abs_key,
                                                subjac.shape, rows.shape))

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