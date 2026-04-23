# Language Selection Rubric

Choose the code language that best fits the algorithm's typical use case and the key concept you want to teach. Pick one language by default.

## Decision Rules
1. If the algorithm is about low-level memory layout, cache behavior, or tight performance loops, prefer C++.
2. If the algorithm relies on ownership/borrowing safety or you want to highlight memory safety, prefer Rust.
3. If the algorithm centers on concurrency, channels, or server-side pipelines, prefer Go.
4. If the algorithm is used in data analysis, scripting, or rapid prototyping, prefer Python.
5. If the algorithm is primarily used in browsers or JS runtimes, prefer JavaScript.
6. If the algorithm is for relational data and query optimization, prefer SQL (with a minimal runnable example).
7. If none of the above are decisive, default to Python for clarity and note the assumption.

## Overrides
- If the user requests a specific language, use it.
- If the algorithm is canonically taught in a specific language and that improves clarity, prefer that language.
- If multiple languages are requested, include them only when asked.
