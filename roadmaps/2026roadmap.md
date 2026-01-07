# OpenMDAO Development Map 2026

Author: Rob Falck
Date: 2026-01-12

## Retrospective 2025

Some restructuring in our team last year resulted in the departure of a few of our team members. They were great people and will be missed.

We achieved some of our main targets for the year, including relevance for partial derivatives and the passing of units information based on connection. While good progress was made towards post-optimality sensitivity, the lack of second derivatives in the current implementation of [MAUD](https://websites.umich.edu/~mdolaboratory/pdf/Hwang2018a.pdf) is a hindrance that needs to be addressed.

## 2026 Focus

Last year saw incredible changes in the AI-related world. It's become clear that things like agentic coding and GPU-based workflows will be the norm. This year, we're aiming for some agressive advances to OpenMDAO to decrease the workload for users, improve modularity, and make OpenMDAO easier to integrate into a larger ecosystem that may involve other analysis frameworks or AI-driven workflows.

These changes are intended to make working with OpenMDAO simpler, but we also recognize that we cannot expect users to go through a painful transition process. We still need to support the existing mode of operation while transitioning users to a better way of doing things

### Decoupling Models from Execution

In OpenMDAO, models are an ephemeral thing that is assembled during the setup stack, executed using the same classes used to assemble the model itself, and then discarded. The user cannot know the exact structure of their model except by interrogating it at various points in the setup or execution process.

We want to make those models their own thing which is persistent, serializable, and swappable.

Groups will no longer be a class to be overridden by the user, but a more basic container for subsystems, connections, and solver algorithms. We will provide pydantic.BaseModel-derived specifications for Groups and Components such that models can be serialized and deserialized. Once assembled, the notion of setting up a model again is unnecessary.

With inputs, outputs, and options defined as pydantic fields, the need to call `setup()` on a model is no longer there. If we utilize AD-related tools like Jax or Julia for our compute methods, then components can be defined as a set of inputs, outputs, and the function which provides the compute.
User-defined derivatives would require letting the component specification know the function that computes the partials or the jacobian-vector products. Critically, we will still need to support
existing components as well.

#### Easier Introspection

This should make complex-OpenMDAO use-cases that involve introspection considerably easier.
A huge amount of dymos' codebase is dedicated to introspection. With these changes, theres no more messing around with setup, configure, and figuring out what exists at any given time.  Instead, the next generation of dymos would accept the users ODE system not as a class to be instantiated, but as fully defined ODE including specifications of metadata like units and shapes.  The notion of binding state metadata to that ODE system also becomes easier. 

If I have a complicated aerodynamics model and I want to swap it in place of another model in my system, as long as the interfaces are compatible the user can just replace their model in the model specification or JSON.

#### Enabling Graphical User Interfaces

We've often had users request something along the lines of a GUI for OpenMDAO. In the past, with models defined imperatively, we couldn't do this because you'd inevitably want to recreate the code which provides the model defined in your GUI. What if the inputs to that model were defined programmatically, how could we know. Switching to a declarative approach with pydantic negates that limitation, meaning going from graphically assembled [XDSM diagram](https://mdolab.engin.umich.edu/wiki/xdsm-overview) to a functional computational would be possible.

#### A Separate Executive

In OpenMDAO <= 3, the same Systems/Groups used to define the model also provide much of the computation algoirthm. If we first build this model instead, we can run it through a separate _MAUD-Executive_ (for the lack of a better term) to convert inputs into outputs and to provide relevant derivatives. While the mathematics of MAUD is fairly compute-efficient, it does involve some iteration. Keeping the definition of the model itself in Python makes sense for usibilty reasons, but separating the computational aspect
makes it possible to define the _MAUD-Executive_ in some other language if we choose to do so.

### Functional Access for Wider Generality

OpenMDAO should not be required to be the top-level of a user's workflow. We spent a considerable amount of time developing AnalysisDriver, but life for users would be simpler if they could just call their OpenMDAO model as a function to get the data they want from it.  When OpenMDAO was first started, there was some expectation that users who wanted to use a particular optimization algorithm would develop a `Driver` to handle that. By and large, that hasn't happened. Users of OpenMDAO use those Drivers that the development team have provided. Theres an ever-increasing number of optimization techniques out there, and many just need a simple functional interface. If we just make it possible to generate that functional interface, we expand the utility of OpenMDAO by quite a bit.

Notionally, we would want to be able to work with any optimizer that just needs functions for the primal computation and then derivatives:

```python
f, df = om.build_functions(model,
                           inputs=['x', 'y'],
                           outputs=['obj', 'con'],
                           compute_totals=True,
                           color_totals=False)

result = scipy.optimize.minimize(f, x0, jac=df)
```

Or for the example of parameter sweeps:

```python
f = om.build_functions(model,
                       inputs=['x', 'y'],
                       outputs=['obj', 'con'])

x_vals = np.linspace(0, 10, 100)
y_vals = np.linspace(0, 10, 100)

results = []
for x, y in itertools.product(x_vals, y_vals):
    obj, con = f(x, y)
    results.append((x, y, obj, con))
```


## 2025 Focus Areas

1. Continued expansion of coupling with AD tools. Given what we can do with JAX, it makes sense to make similar efforts towards the portion of the community that relies upon PyTorch for similar capability. We will look at building components that wrap PyTorch models in much the same way that we can wrap JAX models today.

2. Post-Optimality Sensitivity and Suboptimization

We're finishing up some work that generalizes generation of lagrange multipliers following optimization. These multipliers provide the sensitivty of the objective function to the constraints and bounds imposed by the user, and we can use these to obtain sensitivities wrt other model inputs.

In viewing the optimization problem as its own implicit process, we're also interested in obtaining sensitivities for the resulting design variable values with respect to the bounds/constraints and inputs. The math to accomplish this requires second derivatives, and at least initially, we'll be relying upon finite differences **across compute_totals** (not across the optmization) to obtain these.

Ultimately the most robust way to do this would involve directly computing second total derivatives using the MAUD machinery in OpenMDAO. In the past this has always been hampered by the need for the user to compute their own second derivatives. Perhaps AD tools will open up a path to this.

3. AnalysisDriver to Surrogate model tool

One of the biggest use-cases for AnalysisDriver is to inform the creation of surrogate models. It mkes sense that OpenMDAO should provide some automation of this capability.

4. Partial Derivative Relevance

`compute_partials` currently has no way of knowing what partials are actually needed for the current optimization problem.  In many cases this is moot because the partial calculations are generally less expensive than conditionally checking for them.

However, there are cases where individual partial calculations are not cheap. If we could use relevance to determine which ones need to be calculated, OpenMDAO would get considerably faster in some situations.

This is also the case for using finite-difference or complex-step across big models, especially something like a file-wrapped external code. Using `declare_partials(of='*', wrt='*', method='fd')` will cause OpenMDAO to compute all of the partials, even those not needed. There should be room for some considerable performance gains here.

5. Expand use of shape_by_conn and implement units_by_conn

We have had a shape-by-connection capability for some time, but it hasn't gotten significant uptake because computing partials by hand when the shapes of inputs can be changing is too challenging. This is another scenario where AD should help.

We also frequently find ourselves in situations where the units of outputs depend upon the units of inputs. This is often the case when "pass-thru" components are used, or with something like a simple matrix-vector product. Dymos is probably the best example of this.  Given a generic ODE model, it performs some significant introspection to determine the shapes and units of variables in the ODE. Having a units-by-conn capability would make life easier here as well.
