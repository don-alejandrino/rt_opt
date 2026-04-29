- [ ] README for setting up environment when working with source code (README_DEV.md)?
    1. install uv
    2. `uv tool install ruff@latest`
    3. `uv tool install pyright`
- [ ] README: Add section for interface changes (most parameters are now inside config
  objects). List all parameters from the old version and where to find them now
- [x] Delete `setup.cfg`
- [x] Put `.idea` to `.gitignore`
- [x] Put `uv.lock` to `.gitignore`
- [ ] Set up CI-pipeline
    - [x] Set up test coverage
    - [x] Set up code quality checks (ruff and pyright)
    - [ ] Set up branch protection rules for main branch
    - [x] Test matrix (multiple Python versions)
    - [ ] Add publish test results action
    - [x] Regression tests
    - [ ] Set up automatic releases on PyPI when pushing to main branch
- [x] Set up dependabot
- [x] Review and fix AI-generated tests
- [x] Add test for SPSA algorithm
- [x] Fix warnings in tests
- [x] Add regression tests (see 2D case in demo.py)
- [ ] Add `is_integer` option and `objective_function_budget` argument to `optimize`
  interface
- [ ] Add second entrypoint for `optimize` either by singledispatch or by adding a
  second function that accepts an objective function with "pythonic" arguments instead
  of a single numpy array. This would be more user-friendly for hyperparameter tuning.
  The optimization variables would be passed as a dictionary, like
  ```
  vars={
      "x": {"min": 0, "max": 1, "init": 0.5, "is_integer": False},
      "y": {"min": 0, "max": 2, "init": 1: "is_integer": True}
  }
  ```
  Or even better, use a custom dataclass for the variable configuration instead of a
  dictionary.
  These provided arguments would then be converted to the numpy format internally, and
  finally the normal `optimize` function would be called