**************************
pspipe_utils tutorials
**************************



Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)


Flow chart
===================

The parameters for the tutorials are specified in global_tuto.dict
For some of the tutorials, you should precompute

.. code:: shell

    python tuto_prepare_data.py global_tuto.dict
    python tuto_simulation.py global_tuto.dict

then e.g

.. code:: shell

    python tuto_covariances.py global_tuto.dict
    python tuto_combine_spectra.py global_tuto.dict
