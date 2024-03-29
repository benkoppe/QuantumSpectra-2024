.. currentmodule:: quantumspectra_2024.absorption

Stark Model
==============================

The stark model is a general implementation for the impact of an electric field applied to any other ``Model``.

All other ``Model`` classes have an implemented ``apply_electric_field()`` method. 
This model uses this method to calculate a new absorption spectrum from the effects of an electric field on the submodel's absorption spectrum.

More details on the implementation of the Stark effect can be found in the :ref:`Stark Computation Docs <stark-electric-field-calculation>`.

This model requires a **submodel** to be passed in as an argument. The submodel is the model that will be used to calculate the absorption spectrum.

Model Name
----------------
In **config files**, this Model is named ``stark``. This can be specified in ``your-config.toml`` with:

.. code-block:: toml

    model.name = "stark"

In this **package**, this Model is named ``StarkModel``. This can be imported with:

.. code-block:: python

    from quantumspectra_2024.absorption.stark import StarkModel