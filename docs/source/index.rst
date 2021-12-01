.. MSCG Net documentation master file, created by
sphinx-quickstart on Wed Nov 24 07:25:06 2021.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

MSCG-Net Documentation
====================================
This project was designed to adapt the work of Liu et al. and
convert it to a mobile device which would allow for
in-the-field processing of images and much faster response
time and lower network requirements versus a computing cluster
that would typically be utilized for these sorts of tasks. While
working on this adaptation, we utilized two separate methods to
ensure flexibility of classification: a local method powered entirely
by the Android device for those phones with the computing capacity
to spare, and a REST-based method designed to take advantage of
existing networks and send the image back to a computer for offsite
processing, storage, and evaluation. The following paper
describes implementation, downloading and running instructions,
screenshots, and finally comments/critiques.



Development
==================

.. Hidden TOCs

.. toctree::
   :caption: Overview
   :maxdepth: 1
   :hidden:

   ./quickstart
   ./installation
   ./configuring
   ./development
   ./changelog

.. toctree::
    :caption: MSCG-Net Android Documentation
    :maxdepth: 2
    :titlesonly:
    :hidden:

    mscg/android/overview
    mscg/android/demo
    mscg/android/preprocessing
    mscg/android/models
    mscg/android/deployment

.. toctree::
    :caption: CLI Documentation
    :maxdepth: 2
    :hidden:

    mscg/cli/overview
    mscg/cli/installation
    mscg/cli/usage


.. toctree::
    :caption: MSCG-Net API Documentation
    :maxdepth: 2
    :titlesonly:
    :hidden:

    mscg/api/overview
    mscg/api/demo
    mscg/api/preprocessing
    mscg/api/models
    mscg/api/utilities

.. toctree::
    :caption: Results
    :maxdepth: 2
    :titlesonly:
    :hidden:

    results/tables/2021
    results/tables/2020




