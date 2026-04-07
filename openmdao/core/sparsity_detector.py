"""
Automatic sparsity detection for Jacobian patterns.

This module provides utilities for automatically determining the sparsity patterns
of the Jacobian (J1) by propagating test vectors through the system using only
the declared sparsity information.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor


class SparsityDeterminer:
    """
    Automatically determine J1 sparsity patterns using declared partials.

    This class uses forward and reverse mode propagation of sparsity patterns
    (1's and 0's) through the system to determine which outputs depend on which
    design variables, without materializing full matrices or computing numerical derivatives.

    Attributes
    ----------
    system : System
        The OpenMDAO System to analyze.
    _J1_sparsity_cache : dict
        Cache of computed J1 sparsity patterns.
    _J1_colors : ndarray or None
        Graph coloring of J1 sparsity pattern (computed on demand).
    """

    def __init__(self, system):
        """
        Initialize the SparsityDeterminer.

        Parameters
        ----------
        system : System
            The OpenMDAO System to analyze for sparsity.
        """
        self.system = system
        self._J1_sparsity_cache = {}
        self._J1_colors = None

    def determine_J1_column_sparsity(self, design_var_index, use_cache=True):
        """
        Determine which outputs depend on one design variable (forward mode).

        Parameters
        ----------
        design_var_index : int
            Index of the design variable to propagate.
        use_cache : bool
            If True, check and use cached results.

        Returns
        -------
        ndarray of bool
            Boolean array indicating which outputs are affected by this design variable.
        """
        if use_cache and design_var_index in self._J1_sparsity_cache:
            return self._J1_sparsity_cache[design_var_index].copy()

        # Get design variable metadata
        design_var_names = [n for n in self.system._var_allprocs_abs2meta['output'].keys()
                           if self.system._var_allprocs_abs2meta['output'][n].get('is_design_var',
                                                                                  False)]

        if design_var_index >= len(design_var_names):
            raise ValueError(f"Design variable index {design_var_index} out of range "
                           f"(system has {len(design_var_names)} design variables)")

        dv_name = list(design_var_names)[design_var_index]

        # Create perturbation vectors using _matvec_context
        with self.system._matvec_context(scope_out=None, scope_in=None, mode='fwd') as vecs:
            d_inputs, d_outputs, d_residuals = vecs

            # Set one element in the design variable
            d_inputs._abs_set_val(dv_name, np.ones(self.system._var_allprocs_abs2meta
                                                   ['output'][dv_name]['size']))

            # Propagate through system in sparsity mode
            self.system._apply_linear_sparsity('fwd', scope_out=None, scope_in=None)

            # Extract sparsity pattern from outputs
            output_names = list(self.system._var_allprocs_abs2meta['output'].keys())
            sparsity = np.zeros(len(output_names), dtype=bool)

            for i, out_name in enumerate(output_names):
                d_out = d_outputs._abs_get_val(out_name, flat=True)
                sparsity[i] = np.any(d_out)

        # Cache the result
        if use_cache:
            self._J1_sparsity_cache[design_var_index] = sparsity.copy()

        return sparsity

    def determine_J1_row_sparsity(self, output_index, use_cache=True):
        """
        Determine which design variables affect one output (reverse mode).

        Parameters
        ----------
        output_index : int
            Index of the output to analyze.
        use_cache : bool
            If True, check and use cached results.

        Returns
        -------
        ndarray of bool
            Boolean array indicating which design variables affect this output.
        """
        # Get output metadata
        output_names = list(self.system._var_allprocs_abs2meta['output'].keys())

        if output_index >= len(output_names):
            raise ValueError(f"Output index {output_index} out of range "
                           f"(system has {len(output_names)} outputs)")

        out_name = output_names[output_index]

        # Create perturbation vectors using _matvec_context
        with self.system._matvec_context(scope_out=None, scope_in=None, mode='rev') as vecs:
            d_inputs, d_outputs, d_residuals = vecs

            # Set one element in the output
            d_outputs._abs_set_val(out_name,
                                   np.ones(self.system._var_allprocs_abs2meta['output'][out_name]['size']))

            # Propagate through system in reverse sparsity mode
            self.system._apply_linear_sparsity('rev', scope_out=None, scope_in=None)

            # Extract sparsity pattern from design variables
            design_var_names = [n for n in output_names if
                               self.system._var_allprocs_abs2meta['output'][n].get('is_design_var',
                                                                                   False)]
            sparsity = np.zeros(len(design_var_names), dtype=bool)

            for i, dv_name in enumerate(design_var_names):
                d_in = d_inputs._abs_get_val(dv_name, flat=True)
                sparsity[i] = np.any(d_in)

        return sparsity

    def determine_full_J1_sparsity(self, parallel=False, n_workers=None, use_reverse=None):
        """
        Determine complete J1 sparsity matrix.

        Parameters
        ----------
        parallel : bool
            If True, use parallel processing.
        n_workers : int or None
            Number of workers for parallel processing. If None, use CPU count.
        use_reverse : bool or None
            If True, use reverse mode. If False, use forward mode.
            If None, auto-select based on problem size.

        Returns
        -------
        ndarray of bool
            J1 sparsity matrix of shape (n_outputs, n_design_vars).
        """
        # Determine which mode to use
        design_var_names = [n for n in self.system._var_allprocs_abs2meta['output'].keys()
                           if self.system._var_allprocs_abs2meta['output'][n].get('is_design_var',
                                                                                  False)]
        output_names = list(self.system._var_allprocs_abs2meta['output'].keys())
        n_design = len(design_var_names)
        n_outputs = len(output_names)

        if use_reverse is None:
            use_reverse = n_outputs < n_design

        if use_reverse:
            # Reverse mode
            if parallel:
                return self._determine_full_J1_sparsity_reverse_parallel(n_workers)
            else:
                return self._determine_full_J1_sparsity_reverse_serial()
        else:
            # Forward mode
            if parallel:
                return self._determine_full_J1_sparsity_forward_parallel(n_workers)
            else:
                return self._determine_full_J1_sparsity_forward_serial()

    def _determine_full_J1_sparsity_forward_serial(self):
        """
        Determine J1 sparsity using forward mode (serial).

        Returns
        -------
        ndarray of bool
            J1 sparsity matrix.
        """
        design_var_names = [n for n in self.system._var_allprocs_abs2meta['output'].keys()
                           if self.system._var_allprocs_abs2meta['output'][n].get('is_design_var',
                                                                                  False)]
        output_names = list(self.system._var_allprocs_abs2meta['output'].keys())

        n_outputs = len(output_names)
        n_design = len(design_var_names)

        J1_sparsity = np.zeros((n_outputs, n_design), dtype=bool)

        for k in range(n_design):
            J1_sparsity[:, k] = self.determine_J1_column_sparsity(k)

        return J1_sparsity

    def _determine_full_J1_sparsity_forward_parallel(self, n_workers=None):
        """
        Determine J1 sparsity using forward mode (parallel).

        Parameters
        ----------
        n_workers : int or None
            Number of workers. If None, use CPU count.

        Returns
        -------
        ndarray of bool
            J1 sparsity matrix.
        """
        design_var_names = [n for n in self.system._var_allprocs_abs2meta['output'].keys()
                           if self.system._var_allprocs_abs2meta['output'][n].get('is_design_var',
                                                                                  False)]
        n_design = len(design_var_names)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            columns = list(executor.map(self.determine_J1_column_sparsity, range(n_design)))

        return np.column_stack(columns)

    def _determine_full_J1_sparsity_reverse_serial(self):
        """
        Determine J1 sparsity using reverse mode (serial).

        Returns
        -------
        ndarray of bool
            J1 sparsity matrix.
        """
        output_names = list(self.system._var_allprocs_abs2meta['output'].keys())
        n_outputs = len(output_names)

        rows = []
        for m in range(n_outputs):
            rows.append(self.determine_J1_row_sparsity(m))

        return np.row_stack(rows)

    def _determine_full_J1_sparsity_reverse_parallel(self, n_workers=None):
        """
        Determine J1 sparsity using reverse mode (parallel).

        Parameters
        ----------
        n_workers : int or None
            Number of workers. If None, use CPU count.

        Returns
        -------
        ndarray of bool
            J1 sparsity matrix.
        """
        output_names = list(self.system._var_allprocs_abs2meta['output'].keys())
        n_outputs = len(output_names)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            rows = list(executor.map(self.determine_J1_row_sparsity, range(n_outputs)))

        return np.row_stack(rows)

    def detect_singular_rows(self, J1_sparsity=None):
        """
        Detect outputs that don't depend on any design variables.

        Parameters
        ----------
        J1_sparsity : ndarray of bool or None
            Pre-computed J1 sparsity matrix. If None, compute it.

        Returns
        -------
        dict
            Dictionary with 'indices' and 'names' of singular rows.
        """
        if J1_sparsity is None:
            J1_sparsity = self.determine_full_J1_sparsity()

        singular_rows = np.where(~np.any(J1_sparsity, axis=1))[0]
        output_names = list(self.system._var_allprocs_abs2meta['output'].keys())
        singular_names = [output_names[i] for i in singular_rows]

        return {
            'indices': singular_rows,
            'names': singular_names,
            'count': len(singular_rows),
        }

    def detect_singular_columns(self, J1_sparsity=None):
        """
        Detect design variables that don't affect any outputs.

        Parameters
        ----------
        J1_sparsity : ndarray of bool or None
            Pre-computed J1 sparsity matrix. If None, compute it.

        Returns
        -------
        dict
            Dictionary with 'indices' and 'names' of singular columns.
        """
        if J1_sparsity is None:
            J1_sparsity = self.determine_full_J1_sparsity()

        singular_cols = np.where(~np.any(J1_sparsity, axis=0))[0]
        design_var_names = [n for n in self.system._var_allprocs_abs2meta['output'].keys()
                           if self.system._var_allprocs_abs2meta['output'][n].get('is_design_var',
                                                                                  False)]
        singular_names = [list(design_var_names)[i] for i in singular_cols]

        return {
            'indices': singular_cols,
            'names': singular_names,
            'count': len(singular_cols),
        }

    def analyze_jacobian_structure(self, parallel=False):
        """
        Perform comprehensive analysis of J1 structure.

        Parameters
        ----------
        parallel : bool
            If True, use parallel processing for sparsity detection.

        Returns
        -------
        dict
            Dictionary with analysis results including sparsity pattern, density,
            singular rows/columns, and estimated rank.
        """
        J1_sparsity = self.determine_full_J1_sparsity(parallel=parallel)
        nnz = np.sum(J1_sparsity)
        total = J1_sparsity.size

        analysis = {
            'J1_sparsity': J1_sparsity,
            'nnz': nnz,
            'density': nnz / total if total > 0 else 0,
            'shape': J1_sparsity.shape,
            'singular_rows': self.detect_singular_rows(J1_sparsity),
            'singular_columns': self.detect_singular_columns(J1_sparsity),
            'estimated_rank': np.linalg.matrix_rank(J1_sparsity.astype(float)),
        }

        return analysis

    def determine_J1_with_coloring(self, parallel=False):
        """
        Determine J1 sparsity and compute graph coloring.

        Parameters
        ----------
        parallel : bool
            If True, use parallel processing for sparsity detection.

        Returns
        -------
        tuple
            (J1_sparsity, J1_colors) where J1_colors is an integer array
            indicating the color of each design variable column.
        """
        J1_sparsity = self.determine_full_J1_sparsity(parallel=parallel)

        # Compute graph coloring: design vars with no shared output dependencies
        # can have the same color
        graph = J1_sparsity.T @ J1_sparsity  # (n_design, n_design) coupling matrix

        # Simple greedy coloring algorithm
        n_design = graph.shape[0]
        self._J1_colors = np.full(n_design, -1, dtype=int)

        for dv_idx in range(n_design):
            # Find colors used by coupled design variables
            used_colors = set()
            for other_idx in range(n_design):
                if graph[dv_idx, other_idx] and self._J1_colors[other_idx] >= 0:
                    used_colors.add(self._J1_colors[other_idx])

            # Assign smallest available color
            color = 0
            while color in used_colors:
                color += 1
            self._J1_colors[dv_idx] = color

        return J1_sparsity, self._J1_colors
