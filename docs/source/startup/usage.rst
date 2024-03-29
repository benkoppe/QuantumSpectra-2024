Usage
==========

Once installed (see :doc:`/startup/installation`), QS-2024 can be used in two ways:

* As a command line script (CLI), which requires no Python programming and outputs results as files.
* As a Python package, which allows for more complex workflows and custom analysis.

Both of these methods are described below.

.. warning:: 
    Before using QS-2024, don't forget to activate the virtual environment where it was installed!

    .. code-block:: bash

        source /path/to/venv/bin/activate


Running from the command line (CLI)
------------------------------------------

QS-2024 can be run using config files from the command line.

#. **Create a config file**

    Config files contain all the information needed to run a QS-2024 absorption model.
    They are written in the TOML format, and should be named with the ``.toml`` extension.

    Example configs for each model can be found in the `sample configs <https://github.com/benkoppe/QuantumSpectra-2024/tree/main/sample_configs>`_ directory on GitHub.
    Create a config file for the model you want to run, and fill in the necessary fields.

    More information about config files can be found on the :doc:`../cli/config` page.

#. **Run the model**

    To run the model with the config file, call the ``qs_2024`` script with the path to the config file as an argument:

    .. code-block:: bash

        qs_2024 /path/to/config.toml

    This will save the requested output files specified in the config.


*That's it!*
That's all you need to do to run a QS-2024 model from the command line.

Using as a package
---------------------------------

QS-2024 can be run as a package from Python programs or Jupyter notebooks.

#. **Import the package**

    Import models from the package. An asterisk imports all models, or a specific model can be specified.
    To see specific model names, see the Model documentation. 

    .. code-block:: python

        from quantumspectra_2024.absorption import *

#. **Initialize a Model object**

    Create a model object by calling the model's constructor with all parameters specified:

    .. code-block:: python

        model = ModelName(param1=value1, param2=value2, ...)

#. **Call the model's** ``get_absorption()`` **method**

    All models come with a ``get_absorption()`` method that calculates the absorption spectrum and returns it as a ``AbsorptionSpectrum`` object.

    .. code-block:: python

        spectrum = model.get_absorption()

    This will return an ``AbsorptionSpectrum`` instance to the ``spectrum`` variable.
    Details on the ``AbsorptionSpectrum`` class can be found in the :doc:`Absorption Spectrum Docs <../package/absorption_spectrum>`.

    Accessing spectrum data:

    .. code-block:: python

        x, y = spectrum.energies, spectrum.intensities
        print(x)
        print(y)

    Saving spectrum data:

    .. code-block:: python

        spectrum.save_data("path/to/output/file.csv")
        spectrum.save_plot("path/to/output/plot.png")

*That's it!*
That's all you need to do to run a QS-2024 model from Python.