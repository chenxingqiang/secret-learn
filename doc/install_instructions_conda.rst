Install conda using the
`conda-forge installers <https://conda-forge.org/download/>`__ (no
administrator permission required). Then run:

.. prompt:: bash

  conda create -n xlearn-env -c conda-forge secret-learn
  conda activate xlearn-env

In order to check your installation, you can use:

.. prompt:: bash

  conda list secret-learn  # show secret-learn version and location
  conda list               # show all installed packages in the environment
  python -c "import xlearn; xlearn.show_versions()"
