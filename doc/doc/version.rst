Version
=======

.. note::

	If there is no root commit yet git will emit a fatal error.

Based upon the current state of the git repository CMake automatically updates
the version number with details. The format is
``major.minor.patch.tweak-suffix-commit+``. The ``+`` is only added when the
repository is dirty and the ``-commit`` is hidden when the current version is
exactly a tag.

When the latest git tag does not match the version defined in the main
``CMakeLists.txt`` an error is emitted and configuring is aborted. This helps to
ensure version number consistency.

If the ``.git`` directory is missing version updating will be skipped silently.
The version defined in the main ``CMakeLists.txt`` wil be used. This is useful
for e.g. release tarballs, which don't contain the git history.
