QuantumSpectra-2024
===============================================

Welcome to the documentation for QuantumSpectra-2024, or **QS-2024** for short.

QS-2024 is a Python package for **simulating absorption spectra.**

A number of different models are implemented, all of which are based in quantum mechanical or semiclassical theory.

To get started, read the :doc:`Insallation Guide <startup/installation>`.

------------

.. toctree:: 
   :hidden:

   self

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   startup/installation
   startup/usage

.. toctree:: 
   :maxdepth: 1
   :caption: Models

   models/two_state
   models/mlj
   models/stark

.. toctree:: 
   :maxdepth: 1
   :caption: Computation Information

   computation/two_state_computation
   computation/mlj_computation
   computation/stark_computation

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   cli/config
   package/absorption_spectrum
