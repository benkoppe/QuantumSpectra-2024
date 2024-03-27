Usage
==========

Once installed (see :doc:`/startup/installation`), QS-2024 can be used in two ways:

* As a simple command line script, which requires no Python programming and outputs results as files.
* As a Python package, which allows for more complex workflows and custom analysis.

Both of these methods are described below.

.. note:: 
    Before using QS-2024, don't forget to activate the virtual environment where it was installed!

    .. code-block:: bash

        source /path/to/venv/bin/activate


Using QS-2024 from the command line
--------------------------------------

QS-2024 can be run using config files from the command line.

#. **Create a config file**

    Config files contain all the information needed to run a QS-2024 absorption model.
    They are written in the TOML format, and should be named with the ``.toml`` extension.

    Example configs for each model can be found in the `sample configs <https://github.com/benkoppe/QuantumSpectra-2024/tree/main/sample_configs>`_ directory on GitHub.
    Create a config file for the model you want to run, and fill in the necessary fields.

    More information about config files can be found at: TODO

#. **Run the model**

    To run the model with the config file, call the ``qs_2024`` script with the path to the config file as an argument:

    .. code-block:: bash

        qs_2024 /path/to/config.toml

    This will save the requested output files specified in the config.


*That's it!*
That's all you need to do to run a QS-2024 model from the command line.

Using QS-2024 as a package
---------------------------------

TODO