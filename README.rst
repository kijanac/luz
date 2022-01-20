==========
Luz Module
==========

.. image:: https://codecov.io/gh/kijanac/luz/branch/main/graph/badge.svg
  :target: https://codecov.io/gh/kijanac/luz

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github

**Lightweight framework for structuring arbitrary reproducible neural network learning procedures using PyTorch.**

PyTorch code can easily become complex, unwieldy, and difficult to understand as a project develops. Luz aims to provide a common scaffold for PyTorch code in order to minimize boilerplate, maximize readability, and maintain the flexibility of PyTorch itself.

The basis of Luz is the Runner, an abstraction representing batch-wise processing of data over multiple epochs. Runner has predefined hooks to which code can be attached and a State which can be manipulated to define essentially arbitrary behavior. These hooks can be used to compose multiple Runners into a single algorithm, enabling dataset preprocessing, model testing, and other common tasks.

To further reduce boilerplate, the Learner abstraction is introduced as shorthand for the extremely common Preprocess-Train-Validate-Test algorithm. Simply inherit luz.Learner and define a handful of methods to completely customize your learning algorithm.

Two additional abstractions are provided for convenience: Scorers, which score (i.e. evaluate) a Learner according to some predefined procedure, and Tuners, which tune Learner hyperparameters. These abstractions provide a common interface which makes model selection a two-line process.

---------------
Getting Started
---------------

Installing
----------
From `pip <https://pypi.org/project/luz/>`_:

`pip install luz`

From `conda <https://anaconda.org/kijana/luz>`_:

`conda install -c conda-forge -c pytorch -c kijana luz`

Documentation
-------------
See documentation `here <https://kijanac.github.io/luz/>`_.

Examples
--------
See example scripts in [Examples](examples).

-------
Authors
-------
Ki-Jana Carter

-------
License
-------
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

------------
Contributing
------------
See [CONTRIBUTING](CONTRIBUTING).