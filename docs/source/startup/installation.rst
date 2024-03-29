Installation
==================

Install QS-2024
------------------------------

.. contents:: 
   :local:
   :depth: 2

#. **Install Python**

    If you don't have Python installed, you can download it from the official website: https://www.python.org/downloads/

    .. note::

        Some Python distributions use ``python3`` instead of ``python``. 
        
        
        If you are using a distribution that uses ``python3``, you should replace ``python`` with ``python3`` in all commands.
        
        
        Similarly, ``pip3`` may need to be used instead of ``pip``.

#. **Create a directory**

    Create a directory where you want to install QS-2024. Run the following command:

    .. code-block:: bash

        mkdir qs-2024

    This will create a new directory called ``qs-2024``. Next, call:

    .. code-block:: bash

        cd qs-2024

    This will move you to the directory you just created.

#. **Set up a virtual environment**

    It is recommended to set up a virtual environment to install QS-2024. Run the following command:

    .. code-block:: bash

        python -m venv .venv

    This will create a new directory called ``.venv`` with the virtual environment. Then, call:

    .. code-block:: bash

        source .venv/bin/activate

    This will activate the virtual environment.

#. **Install from** ``pip``

    Install QS-2024 from ``pip`` with the following command:

    .. code-block:: bash

        pip install quantumspectra-2024


    This will install all dependencies, the QS-2024 package, and the command line interface.
    Everything is now installed! You're ready to proceed to :doc:`usage`.


Update QS-2024
------------------------------

    To update QS-2024, activate the environment and run the following command:

    .. code-block:: bash

        pip install --upgrade quantumspectra-2024


Configure GPU calculations
----------------------------

TODO