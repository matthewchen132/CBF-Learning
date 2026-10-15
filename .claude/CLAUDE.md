## Feature Wishlist (Note for self)
- Logger class for all hard-coded constants 
    - Needed for replicable results
- Class for Sobol sequence initialization of input states

## Working style
- Explain the concept and the specific lines involved BEFORE editing; one edit at a time, wait for approval.
- Use guiding questions for learning; reference exact `file:line` locations.
- Run scripts via `conda run -n gibo_env python …` (scripts call input(); set MPLBACKEND=Agg for headless runs).