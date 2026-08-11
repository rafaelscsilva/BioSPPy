Contributing to BioSPPy
=======================

Thank you for helping improve ``BioSPPy``. This page summarizes the contribution
workflow in a simple, practical format.

Before You Start
----------------

- You need a `GitHub account <https://github.com/signup>`_.
- Contributions are made through your fork of
  `scientisst/BioSPPy <https://github.com/scientisst/BioSPPy>`_.

Quick Workflow
--------------

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally.
3. **Create a branch** for your work.
4. **Install dependencies** and make your changes.
5. **Commit and push** your branch.
6. **Open a Pull Request** to ``scientisst/BioSPPy:main``.

Setup Commands
--------------

.. code-block:: bash

   git clone https://github.com/yourusername/biosppy.git
   cd biosppy
   git checkout -b your-feature-name
   pip install -r requirements.txt

You can also clone using GitHub Desktop if you prefer a GUI workflow.

Making Good Changes
-------------------

- Keep commits small and focused (one logical change per commit).
- Write clear commit messages.
- Follow existing project conventions.
- Update docstrings and docs when behavior changes.

Example commit:

.. code-block:: bash

   git add .
   git commit -m "Improve ECG peak detection edge-case handling"

Code Style
----------

- Follow `PEP 8 <https://peps.python.org/pep-0008/>`_.
- Use ``snake_case`` for variables and functions.
- Prefer clear, well-structured code over clever shortcuts.
- Use docstrings in
  `numpydoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
- Avoid adding new dependencies unless they are necessary.

Open a Pull Request
-------------------

After committing locally, push your branch:

.. code-block:: bash

   git push origin your-feature-name

Then open a Pull Request from your fork to ``scientisst/BioSPPy`` ``main``.

When writing your PR:

- Use a clear title.
- Explain *what* changed and *why*.
- Mention any important trade-offs or limitations.

If your fork is behind, sync it with upstream before opening the PR:
`Syncing a fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork>`_.

Need Help?
----------

- Open an issue:
  `github.com/scientisst/BioSPPy/issues/new <https://github.com/scientisst/BioSPPy/issues/new>`_
- Contact maintainers: `developer@scientisst.com <mailto:developer@scientisst.com>`_

Thanks again for contributing.

