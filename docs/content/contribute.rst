Contributor Guidelines
======================

Contributions are welcome and encouraged!  Please follow these guidelines:

* The package use `poetry <https://python-poetry.org/>`_ for dependency
  management.  Please install poetry and run `poetry install` to install all
  dependencies. You can run :code:`poetry shell` to activate the virtual
  environment.
* Use dependency groups: for example :code:`poetry add <package> --dev` to add a
  development dependency, :code:`poetry add <package>` to add a runtime
  dependency. Groups used in the package are `dev` for development dependencies,
  `tests` for test dependencies, and `docs` for documentation dependencies.
* Contribution should follow the black code style.  You can run `black .` to
  automatically format your code.
* Type annotation is not mandatory, but it is encouraged, especially for
  public functions.
* Please add tests for your code.  You can run `pytest` to run all tests.
* If possible, use semantic commit messages.  For example, use `feat` for new
  features, `fix` for bug fixes, `docs` for documentation updates, `refactor`
  for code refactoring, `test` for test updates, `chore` for build updates, and
  `ci` for CI updates. They will be enforced in the future.
* Ruff is used for linting.  You can run `ruff` to lint your code and auto-fix
  some issues.

License
-------

This project is licensed under the terms of the MIT license.

See `LICENSE <LICENSE>`_ for more details.

Code of Conduct
---------------

This project follows the `Contributor Covenant <../code_of_conduct.md>`_.