Config Files
=======================

Config files contain all of the information needed to generate a model's absorption spectrum.

.. note::

    Config files are written in the ``toml`` format. 
    As a result, all config files should be named with the ``.toml`` extension.

This page details the structure of config files, and provides an example config file.

Config File Samples
----------------------

Each Model has a sample config file provided.
These files can be found in the `sample configs <https://github.com/benkoppe/QuantumSpectra-2024/tree/main/sample_configs>`_ directory on GitHub.

The documentation for each Model also has specific information about the details of its config file.


Example Config File
----------------------

Here, a full example of a config file is provided. 
This config file describes a two-state model, and saves output data and plot files.

.. code-block:: toml

    out.filename = "two_state_output" # output files will be named "two_state_output.csv" and "two_state_output.png"
    out.data = true # output data to a csv file
    out.plot = true # output a plot
    out.overwrite = true

    model.name = "two_state" # use the two-state model

    # skip describing model start and end energy, and number of points, as they have default values

    model.temperature_kelvin = 300 # temperature in Kelvin
    model.broadening = 200 # broadening in wavenumbers

    model.transfer_integral = 100 # transfer integral
    model.energy_gap = 8000 # energy gap in wavenumbers

    model.mode_basis_sets = [20, 200] # number of basis functions for each mode
    model.mode_frequencies = [1200, 100] # frequency of each mode in wavenumbers
    model.mode_couplings = [0.7, 2.0] # coupling of each mode with excited state

This config file can be run with the ``qs_2024`` cli script.

Tables
------

Config files are divided into tables. Each table contains a set of parameters specific to that table.

For example, parameters in the the ``out`` table are specified like:

.. code-block:: toml

    out.parameter_name = "parameter_value"

Output Parameters
------------------

These parameters detail the output of the absorption spectrum.
They can be used across all models.

All output parameters belong in the ``out`` table.

.. note:: 

    All output parameters are **required**.

- ``out.filename``

    Path and name of output file relative to the current working directory.
    Output csv file will append a ".csv" extension to the filename.
    Output plot will append a ".png" extension to the filename.

- ``out.data``

    Whether to output the absorption spectrum data to a csv file.
    If False, no ".csv" file will be generated.

- ``out.plot``

    Whether to output the absorption spectrum plot.
    If False, no ".png" file will be generated.

- ``out.overwrite``

    Whether to overwrite the output file if it already exists.
    If False, the program will exit if the output file already exists.


Model Parameters
-------------------------

These parameters detail the model used to generate the absorption spectrum.
Besides ``name``, these parameters are specific to the model used. See the documentation for the specific model for more information.

All output parameters belong in the ``model`` table.

Required Model Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``model.name``

    Name of the model to use.
    The name of each model is listed in the documentation for that model.

- all model parameters without default values

    These parameters are specific to the model used. Most of a model's parameters fall in this category. Each value is the same as the parameter name in the model's documentation.

Optional Model Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

- all model parameters *with* default values

    These parameters are specific to the model used. Each value is the same as the parameter name in the model's documentation.

- ``model.start_energy``

    The starting energy for the absorption spectrum in wavenumbers.
    Defaults to 0.

- ``model.end_energy``

    The ending energy for the absorption spectrum in wavenumbers.
    Defaults to 20,000.

- ``model.num_points``

    The number of points in the absorption spectrum.
    Defaults to 2,001.


Parameterizing Submodels
^^^^^^^^^^^^^^^^^^^^^^^^

Some models have submodels that must be parameterized (such as Stark).
With these models, the submodel parameter can be specified as a table, like ``model``, that contains the submodel's parameters.
This table must be a subtable of the ``model`` table.

For example:

.. code-block:: toml

    model.submodel_parameter_name.name = "submodel_name"
    model.submodel_parameter_name.submodel_parameter = "submodel_parameter_value"

.. note::
    The ``submodel_parameter_name`` is the name of the submodel parameter in the model, in the same way that other parameters exactly match the parameter name in the model.


Sample Configs
----------------

The following contains a full sample config file for each model.
They can be found in the GitHub repository.
See the :ref:`cli/config:Example Config File` section for a clear-cut example.

.. literalinclude:: ../../../sample_configs/two_state.toml
    :caption: `two_state.toml <https://github.com/benkoppe/QuantumSpectra-2024/blob/main/sample_configs/two_state.toml>`_
    :language: toml