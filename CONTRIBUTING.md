# Contributing to raginspect

Thanks for your interest in contributing! This document covers everything you need to get started.

---

## 🍴 How to Contribute

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/raginspect.git
   cd raginspect
   ```
3. **Create a branch** from `develop`:
   ```bash
   git checkout -b feat/my-feature develop
   ```
4. **Make your changes** following the code style below.
5. **Test your changes**:
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --all-features
   ```
6. **Commit** using conventional commits (see below).
7. **Push** and open a pull request against `develop`.

---

## 🛠️ Development Setup

### Prerequisites

- **Rust** (stable toolchain) — install via [rustup](https://rustup.rs):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Python 3.10+** (for bindings development, optional)
- **maturin** (for building Python wheels, optional):
  ```bash
  pip install maturin
  ```

### Building

```bash
# Build the CLI
cargo build --release

# Build Python bindings
maturin develop --release

# Run the CLI
./target/release/raginspect --help
```

### Running Tests

```bash
# All tests
cargo test --all-features

# Integration tests only
cargo test --test '*' --all-features

# With output
cargo test -- --nocapture
```

---

## 🎨 Code Style

### Rust

- **Formatting**: `cargo fmt --all` is mandatory. CI will fail if code is not formatted.
- **Linting**: `cargo clippy` with `-D warnings`. Fix all warnings before committing.
- **Naming**: Follow standard Rust naming conventions (`snake_case` for functions/variables, `PascalCase` for types/structs).
- **Error handling**: Use `thiserror` for library errors, `anyhow` for application/bin errors. Don't `unwrap()` in library code — use `?` with proper error types.
- **Documentation**: All public items must have doc comments (`///`). Module-level docs (`//!`) at the top of each file.

### Python Bindings

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Type hints required on all public functions.
- Use `maturin` for building wheels.

---

## 📝 Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/). Each commit message should be structured:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Use for |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or correcting tests |
| `chore` | Build, CI, tooling, dependencies |
| `ci` | CI/CD changes |

### Scopes

Common scopes: `profiler`, `classifier`, `reporter`, `metrics`, `bindings`, `cli`, `docs`, `ci`.

### Examples

```
feat(profiler): add p99 latency tracking for retrieval stage
fix(classifier): correct HyDE detection when no transform stage exists
docs(readme): add quick start for Python bindings
chore(deps): bump pyo3 to 0.22
```

---

## 🐛 Reporting Issues

### Bug Reports

Use the [Bug Report template](https://github.com/fiqihfathor/raginspect/issues/new?template=bug_report.md). Include:

- raginspect version (`raginspect --version`)
- Rust version (`rustc --version`)
- RAG architecture type (Naive, Advanced, Modular, etc.)
- Pipeline details (stages, retriever type, LLM provider)
- Minimal reproduction steps
- Expected vs actual behavior

### Feature Requests

Use the [Feature Request template](https://github.com/fiqihfathor/raginspect/issues/new?template=feature_request.md). Include:

- The problem you're trying to solve
- Proposed solution
- RAG architecture this applies to
- Alternatives you've considered

---

## 🏗️ Architecture Overview

raginspect is structured as a Rust workspace with multiple crates:

| Crate | Responsibility |
|---|---|
| `raginspect-core` | Core profiling engine, pipeline model, timing infrastructure |
| `raginspect-metrics` | Quality metrics (faithfulness, relevance, answer quality) |
| `raginspect-arch` | Architecture detection and classification (7 types) |
| `raginspect-bindings` | pyo3 Python bindings |

See [docs/architecture.md](docs/architecture.md) for the full breakdown.

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

*Questions? Open an issue or reach out on [Discord](https://discord.gg/raginspect).*
