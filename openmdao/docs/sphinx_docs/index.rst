Welcome to OpenMDAO
===================

OpenMDAO is an open-source high-performance computing platform for
systems analysis and multidisciplinary optimization, written in Python.
It enables you to decompose your models, making them easier to build and
maintain, while still solving them in a tightly coupled manner with
efficient parallel numerical methods.

The OpenMDAO project is primarily focused on supporting gradient-based
optimization with analytic derivatives to allow you to explore large
design spaces with hundreds or thousands of design variables, but the
framework also has a number of parallel computing features that can
work with gradient-free optimization, mixed-integer nonlinear
programming, and traditional design space exploration.

If you are using OpenMDAO, please :doc:`cite <other/citing>` us!

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started/getting_started
   basic_user_guide/basic_user_guide
   advanced_user_guide/advanced_user_guide
   Reference Guide <theory_manual/theory_manual>
   other_useful_docs/other_useful_docs

User Guide
----------

These are a collection of tutorial problems that teach you important concepts and techniques for using OpenMDAO.
For new users, you should work through all material in **Getting Started** and **Basic User Guide**.
That represents the minimum set of information you need to understand to be able to work with OpenMDAO models.

You will also find tutorials in the **Advanced User Guide** to be very helpful as you grow more familiar with OpenMDAO,
but you don't need to read these right away.
They explain important secondary concepts that you will run into when working with more complex OpenMDAO models.

Reference Guide
---------------

These docs are intended to be used by as a reference by users looking for explanation of a particular feature in detail or
documentation of the arguments/options/settings for a specific method, Component, Driver, or Solver.

Other Useful Docs
-----------------

Documentation for command-line tools, API translation guides, developer documentation, and more.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/getting_started

.. toctree::
   :maxdepth: 2
   :caption: Basic User Guide
   :hidden:

   basic_user_guide/basic_user_guide
   basic_user_guide/single_disciplinary_optimization/component_types
   basic_user_guide/single_disciplinary_optimization/first_analysis
   basic_user_guide/single_disciplinary_optimization/first_optimization
   basic_user_guide/multidisciplinary_optimization/sellar
   basic_user_guide/multidisciplinary_optimization/linking_vars
   basic_user_guide/multidisciplinary_optimization/sellar_opt
   basic_user_guide/command_line/check_setup
   basic_user_guide/command_line/make_n2
   basic_user_guide/reading_recording/basic_recording_example

.. toctree::
   :maxdepth: 2
   :caption: Advanced User Guide
   :hidden:

   advanced_user_guide/advanced_user_guide
   advanced_user_guide/models_implicit_components/models_with_solvers_implicit
   advanced_user_guide/models_implicit_components/implicit_with_balancecomp
   advanced_user_guide/analytic_derivatives/derivs_of_coupled_systems
   advanced_user_guide/analytic_derivatives/partial_derivs_implicit
   advanced_user_guide/analytic_derivatives/partial_derivs_explicit
   advanced_user_guide/recording/advanced_case_recording
   advanced_user_guide/example/euler_integration_example
   advanced_user_guide/complex_step
   advanced_user_guide/analysis_errors/analysis_error

.. toctree::
   :maxdepth: 2
   :caption: Reference Guide
   :hidden:

   theory_manual/theory_manual
   theory_manual/class_structure
   theory_manual/implicit_transformation_of_vars
   theory_manual/setup_stack
   theory_manual/solver_api
   theory_manual/scaling
   theory_manual/iter_count
   theory_manual/total_derivs_theory
   theory_manual/setup_linear_solvers
   theory_manual/advanced_linear_solvers_special_cases/advanced_linear_solvers_special_cases
   theory_manual/mpi
   features/features
   features/core_features/main.md
   features/building_blocks/building_blocks
   features/recording/main.md
   features/model_visualization/main.md
   features/debugging/debugging
   features/debugging/mpi_debugging
   features/debugging/listing_variables
   features/debugging/debugging_solvers
   features/debugging/debugging_your_optimizations
   features/debugging/newton_solver_not_converging
   features/debugging/controlling_mpi
   features/debugging/debugging_drivers
   features/debugging/profiling/index
   features/warning_control/warnings
   features/units
   features/experimental/main.md
   examples/examples
   examples/tldr_paraboloid
   examples/paraboloid
   examples/betz_limit
   examples/hohmann_transfer/hohmann_transfer
   examples/keplers_equation
   examples/circuit_analysis_examples
   examples/beam_optimization_example
   examples/beam_optimization_example_part_2
   examples/simul_deriv_example

.. toctree::
   :maxdepth: 2
   :caption: Other Useful Docs
   :hidden:

   other_useful_docs/other_useful_docs
   other_useful_docs/om_command
   other/citing
   other_useful_docs/building_a_tool/building_a_tool
   other_useful_docs/building_a_tool/repository_structure
   other_useful_docs/building_a_tool/release_process
   other_useful_docs/auto_ivc_api_translation
   other_useful_docs/api_translation
   other_useful_docs/environment_vars
   other_useful_docs/file_wrap
   _srcdocs/index
   other_useful_docs/developer_docs/developer_docs
   other_useful_docs/developer_docs/signing_commits
   other_useful_docs/developer_docs/unit_testing
   other_useful_docs/developer_docs/ci_testing
   other_useful_docs/developer_docs/doc_build
   other_useful_docs/developer_docs/doc_style_guide
   other_useful_docs/developer_docs/sphinx_decorators
   other_useful_docs/developer_docs/writing_plugins
