Config Files
=======================

Config files contain all of the information needed to generate a model's absorption spectrum.

.. important::

    Config files are written in the ``toml`` format. 
    As a result, all config files should be named with the ``.toml`` extension.

Config files are divided into :ref:`config-tables`. Two tables are used:

- ``out``: Contains parameters that detail the output of the absorption spectrum. :ref:`# <config-out-table>`
- ``model``: Contains parameters that detail the model used to generate the absorption spectrum. :ref:`# <config-model-table>`

To see sample config files for each model, see the :ref:`config-samples` section.


.. _config-tables:

Tables
------

Config files are divided into tables. Each table contains a set of parameters specific to that table.

For example, a ``str`` parameter in the the ``out`` table is specified as follows:

.. code-block:: toml

    out.parameter_name = "parameter_value"

Some parameters have different types, like ``array``, ``int``, ``float``, or ``bool``.
These are specified in the same way, but with the appropriate type:

.. code-block:: toml

    out.array_parameter = [1, 2, 3] # this is an array[int] parameter
    out.float_parameter = 4.5 # this is a float parameter
    out.bool_parameter = true # this is a bool parameter

.. _config-out-table:

``out`` Table
------------------

These parameters detail the output of the absorption spectrum.
They can be used across all models.

All output parameters belong in the ``out`` table.

.. warning:: 

    All output parameters are **required**.

.. attribute:: out.filename
    :type: str

    Path and name of output file relative to the current working directory.
    Output csv file will append a ".csv" extension to the filename.
    Output plot will append a ".png" extension to the filename.

.. attribute:: out.data
    :type: bool

    Whether to output the absorption spectrum data to a csv file.
    If False, no ".csv" file will be generated.

.. attribute:: out.plot
    :type: bool

    Whether to output the absorption spectrum plot.
    If False, no ".png" file will be generated.

.. attribute:: out.overwrite
    :type: bool

    Whether to overwrite the output file if it already exists.
    If False, the program will exit if the output file already exists.


.. _config-model-table:

``model`` Table
-------------------------

These parameters detail the model used to generate the absorption spectrum.
Besides ``name``, these parameters are specific to the model used. See the documentation for the specific model for more information.

All output parameters belong in the ``model`` table.

Required Model Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: model.name
    :type: str

    Name of the model to use.
    The name of each model is listed in the documentation for that model.

- all model parameters without default values

    These parameters are specific to the model used. Most of a model's parameters fall in this category. Each value is the same as the parameter name in the model's documentation.

Optional Model Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

- all model parameters *with* default values

    These parameters are specific to the model used. Each value is the same as the parameter name in the model's documentation.

.. attribute:: model.start_energy
    :type: float

    The starting energy for the absorption spectrum in wavenumbers.
    Defaults to 0.

.. attribute:: model.end_energy

    The ending energy for the absorption spectrum in wavenumbers.
    Defaults to 20,000.

.. attribute:: model.num_points

    The number of points in the absorption spectrum.
    Defaults to 2,001.


Parameterizing Submodels
^^^^^^^^^^^^^^^^^^^^^^^^

Some models have submodels that must be parameterized (such as :doc:`../models/stark`).
With these models, the submodel parameter can be specified as a table, like ``model``, that contains the submodel's parameters.
This table must be a subtable of the ``model`` table.

For example:

.. code-block:: toml

    model.submodel_parameter_name.name = "submodel_name"
    model.submodel_parameter_name.submodel_parameter = "submodel_parameter_value"

.. important::
    The ``submodel_parameter_name`` is the name of the submodel parameter in the model, in the same way that other parameters exactly match the parameter name in the model.


.. _config-samples:

Sample Configs
----------------

Each model has a sample config file provided. 
They are reproduced here, or can be found in the `sample configs <https://github.com/benkoppe/QuantumSpectra-2024/tree/main/sample_configs>`_ directory on GitHub.

Two-State Example
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../sample_configs/two_state.toml
    :caption: `two_state.toml <https://github.com/benkoppe/QuantumSpectra-2024/blob/main/sample_configs/two_state.toml>`_
    :language: toml

MLJ Example
^^^^^^^^^^^^

.. literalinclude:: ../../../sample_configs/mlj.toml
    :caption: `mlj.toml <https://github.com/benkoppe/QuantumSpectra-2024/blob/main/sample_configs/mlj.toml>`_
    :language: toml


Stark Example
^^^^^^^^^^^^

.. literalinclude:: ../../../sample_configs/stark.toml
    :caption: `stark.toml <https://github.com/benkoppe/QuantumSpectra-2024/blob/main/sample_configs/stark.toml>`_
    :language: toml