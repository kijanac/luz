==============
Luz Module
==============

.. image:: https://codecov.io/gh/kijanac/luz/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/kijanac/luz

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github

**Framework for rapid research and development of machine learning projects using PyTorch.**

Longer description coming soon!

Basic usage: custom modules with luz.Module functionality. Simply define your model, inherit from luz.Module, and use model.train/model.test.

Next level: simplified training algorithms with transforms and handlers.

Next level: straightforward hyperparameter tuning with various scoring mechanisms.

#. Data preparation scheme.

  #. Transforms: Functions for conditionining the training process which are invertible and whose inverse is applied to the resulting predictor.
  #. Preprocessors: Functions for altering the prediction task which are not necessarily invertible.
#. Model.
#. Training scheme.
#. Overall learning algorithm.

  #. Hyperparameter selectin.

---------------
Getting Started
---------------

Prerequisites
-------------

Installing
----------

To install, open a shell terminal and run::

`conda create -n luz -c conda-forge -c pytorch -c kijana luz`

----------
Versioning
----------

-------
Authors
-------

Ki-Jana Carter

-------
License
-------
This project is licensed under the MIT License - see the LICENSE file for details.
