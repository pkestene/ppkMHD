Build Doc
=========

This documentation should be built by the CI system and be made available as a gitlab-page.

This sphinx documentation can be build using CMake.

To build doxygen API documentation:

.. code-block:: bash

   mkdir build
   cd build
   cmake -DBUILD_CODE:BOOL=OFF -DBUILD_DOC:BOOL=ON -DDOC:STRING=doxygen ..
   make doc
   # the output is in doc/doxygen/html/index.html


To build sphinx/html documentation:		

.. code-block:: bash

   mkdir build
   cd build
   cmake -DBUILD_CODE:BOOL=OFF -DBUILD_DOC:BOOL=ON -DDOC:STRING=html ..
   make doc
   # the output web page is in build/doc/html/index.html

