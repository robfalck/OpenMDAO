# Implementation Plan: Add `driver_scaling` Support to JVP/HVP Methods

## Overview

Add `driver_scaling` parameter to `compute_jacvec_product()` and `approx_hessvec_product()` methods in OpenMDAO's Problem class. This enables scaling transformations between model space and optimizer space for Jacobian-vector products and Hessian-vector products.

**Critical Gap Identified:** When `_hesspfunc()` in scipy_optimizer.py calls these methods, results are in model space but scipy optimizers expect optimizer-scaled space. This causes inconsistency with user-defined scaling of design variables and objectives.

## Problem Statement

When users scale design variables or objectives (via `ref=` parameters), the optimization should be performed in that scaled space. Currently:
- `compute_jacvec_product()` returns results in **model space only**
- `approx_hessvec_product()` computes using model-space gradients
- Scaling is applied by autoscaler in `_compute_totals()` path, but NOT in the `_hesspfunc` -> `approx_hessvec_product()` path

This means trust-region optimizers (trust-ncg, trust-constr) using hessp get incorrect Hessian-vector products when scaling is present.

## Solution Overview (Simplified Approach)

Use existing `autoscaler.apply_jac_scaling()` method to scale JVP results:

1. **Modify `compute_jacvec_product()` to support driver_scaling**
   - Add `driver_scaling=False, autoscaler=None` parameters
   - After computing JVP and before returning: if `driver_scaling=True`, apply autoscaler to scale results
   - Use autoscaler's `apply_jac_scaling()` method which already handles the math

2. **`approx_hessvec_product()` inherits scaling automatically**
   - Calls `compute_jacvec_product()` internally at baseline and perturbed points
   - Since both gradients are scaled (if `driver_scaling=True` is passed), the numerical difference is automatically scaled
   - No additional scaling logic needed in HVP method

3. **Update `_hesspfunc()` in scipy_optimizer.py**
   - Pass `driver_scaling=True` and `autoscaler=self._autoscaler` to `approx_hessvec_product()`
   - Which automatically passes them to internal `compute_jacvec_product()` calls

## Scaling Transformations

### Mathematical Basis

**Jacobian Scaling:**
- Model space: `J_model = ∂f/∂x`
- Optimizer space: `J_scaled = (scaler_out * J_model) / scaler_in`
  - `scaler_out`: objective/constraint scaler
  - `scaler_in`: design variable scaler

**Hessian Scaling:**
- Model space: `H_model = ∂²f/∂x²`
- Optimizer space: `H_scaled = scaler_out * H_model / (scaler_in²)`

### Implementation Pattern

**Output Side (Scaling via apply_jac_scaling()):**
1. JVP results in model space are converted to dict format: {(of_name, wrt_name): jvp_array}
2. Call `autoscaler.apply_jac_scaling(jac_dict)` which scales in-place using formula: J_scaled = J_model * out_scaler / in_scaler
3. HVP results automatically inherit scaling through numerical difference of scaled gradients

## Implementation Details

### File 1: `/Users/rfalck/Codes/OpenMDAO.git/openmdao/core/problem.py`

**Modify `compute_jacvec_product()` (line 842):**
- Add parameters: `driver_scaling=False, autoscaler=None`
- Compute JVP normally (no changes to existing logic)
- Before return (line 911): if `driver_scaling=True`, convert result dict to Jacobian format and apply autoscaler
  - Build dict with format expected by `apply_jac_scaling()`: {(of_name, wrt_name): jvp_array}
  - Call `autoscaler.apply_jac_scaling(jac_dict)` to scale in-place
  - Convert back to result format before returning

**Modify `approx_hessvec_product()` (line 913):**
- Add parameters: `driver_scaling=False, autoscaler=None`
- Pass `driver_scaling` and `autoscaler` to all three `compute_jacvec_product()` calls:
  - Line 1011: baseline gradient (forward FD)
  - Line 1098: perturbed gradient (forward FD)
  - Line 1133: backward gradient (central FD only)
- No other changes needed: scaling propagates through numerical differentiation

**Implementation notes:**
- The beauty of this approach: since `(scaled_grad_perturbed - scaled_grad_baseline) / h = scaled(grad_perturbed - grad_baseline) / h`, the HVP result is automatically scaled
- Use same scaler conversion pattern as in apply_jac_scaling()

### File 2: `/Users/rfalck/Codes/OpenMDAO.git/openmdao/drivers/scipy_optimizer.py`

**Modify `_hesspfunc()` (line 912-921):**
```python
hvp_flat = prob.approx_hessvec_product(
    of=list(self._objs),
    wrt=self._dvlist,
    p=p,
    method=method,
    form=form,
    step_calc=step_calc,
    minimum_step=minimum_step,
    step=step,
    driver_scaling=True,          # NEW
    autoscaler=self._autoscaler    # NEW
)
```

## Backward Compatibility

- Default `driver_scaling=False` ensures existing code works unchanged
- Methods work in model space when driver_scaling is not passed
- All existing tests continue to pass without modification

## Testing Strategy

### Unit Tests (New file: `test_jacvec_hvp_scaling.py`)

1. **Test `compute_jacvec_product()` with scaling**
   - Quadratic problem with ref=2.0 (design var), ref=10.0 (objective)
   - Verify: `jvp_scaled = jvp_model * scaler_out / scaler_in`

2. **Test `compute_jacvec_product()` with multiple variables**
   - Different scalers per design variable and objective
   - Verify each is scaled correctly in both fwd/rev modes

3. **Test `approx_hessvec_product()` with scaling**
   - Compare scaled HVP with finite difference of scaled gradients
   - Verify: `hvp_scaled = scaler_f * hvp_model`

4. **Test missing autoscaler (graceful degradation)**
   - Should proceed without scaling error

### Integration Tests (Modify `test_scipy_optimizer.py`)

1. **HVP with scaling in optimization**
   - Run trust-ncg/trust-constr with scaled problem
   - Verify convergence is correct
   - Compare with unscaled baseline

## Challenges and Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| Converting JVP result dict to apply_jac_scaling format | Build flat dict with (of_name, wrt_name) tuples as keys; existing code already handles this format |
| Multiple objectives in HVP | Handled automatically: if multiple objectives, jvp dict has multiple keys; apply_jac_scaling processes all of them |
| JVP result is keyed differently in fwd vs rev mode | In compute_jacvec_product return (line 911), result is keyed by lnames (which are 'of' in fwd, 'wrt' in rev); must match apply_jac_scaling expectations |
| Autoscaler missing (no scaling configured) | Check `if autoscaler is not None` before calling apply_jac_scaling; gracefully skip scaling if autoscaler not provided |

## Files to Modify

1. **openmdao/core/problem.py**
   - Add `driver_scaling=False, autoscaler=None` parameters to `compute_jacvec_product()` and `approx_hessvec_product()`
   - In `compute_jacvec_product()`: apply autoscaler scaling before return (if driver_scaling=True)
   - In `approx_hessvec_product()`: pass parameters to internal compute_jacvec_product() calls

2. **openmdao/drivers/scipy_optimizer.py**
   - Update `_hesspfunc()` call to `approx_hessvec_product()` to pass `driver_scaling=True` and `autoscaler=self._autoscaler`

## Files to Reference

1. **openmdao/drivers/autoscalers/autoscaler.py** - Study `apply_jac_scaling()` method (line 579) for scaling patterns
   - `_var_meta['objective'][name]['total_scaler']` - gets objective scaler
   - `_var_meta['design_var'][name]['total_scaler']` - gets design variable scaler
   - None values treated as 1.0
2. **openmdao/core/driver.py** - Study `_compute_totals()` for driver_scaling pattern (line 1429)
3. **openmdao/vectors/optimizer_vector.py** - Understand OptimizerVector structure and scaling

## Scaling Formula Reference (from autoscaler.py line 579-640)

The autoscaler uses: `J_scaled = out_scaler * J_model / in_scaler`

Key insights:
- `apply_jac_scaling()` modifies Jacobian in-place: `block *= out_scaler / in_scaler`
- For HVP: gradient itself gets scaled, then step size math applies
- Multiple objectives: must track per-objective contributions

## Seed Format Reference (from compute_jacvec_product)

Lines 893-902: Seeds can be dict (keyed by varnames) or sequence (list/array):
- Check seed format: `try: seed[rnames[0]] except (IndexError, TypeError): sequence_mode`
- Dict mode: `rvec[resolver.source(name)] = seed[name]`
- Sequence mode: `rvec[resolver.source(name)] = seed[i]`

## Key Implementation Lines

**problem.py compute_jacvec_product():**
- Line 842: Method signature
- Lines 868-879: Mode determination and seed format detection
- Line 884-886: Resolver and vector access
- Lines 893-902: Seed parsing
- Line 909: `run_solve_linear(mode)` call
- Line 911: Return statement

**problem.py approx_hessvec_product():**
- Line 913: Method signature
- Lines 1011, 1098, 1133: compute_jacvec_product() calls (pass driver_scaling=True)
- Lines 1062-1090: Forward perturbation loop
- Lines 1062-1126: Central difference backward perturbation
- Lines 1149-1205: HVP result computation and concatenation
- Line 1211: Return hvp_flat

**scipy_optimizer.py _hesspfunc():**
- Line 912-921: approx_hessvec_product() call (add driver_scaling, autoscaler params)

## Success Criteria

- ✓ `compute_jacvec_product()` with `driver_scaling=True` returns scaled results
- ✓ `approx_hessvec_product()` with `driver_scaling=True` returns scaled HVP
- ✓ Scaled HVP matches numerical validation with `(grad_scaled(x+εp) - grad_scaled(x)) / ε`
- ✓ trust-ncg and trust-constr converge correctly with scaling enabled
- ✓ Backward compatibility maintained: default behavior unchanged
- ✓ All existing tests pass
- ✓ New tests for scaling validation pass
- ✓ Code passes ruff check
