# Contributing

Thank you for your interest in contributing to the Hierarchical Planner Agent!

## Development Setup

1. Fork and clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install Ollama and pull a model: `ollama pull qwen2.5-coder:7b`
4. Run the test suite: `python -m pytest tests/ -v`

## Code Style

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Every class should have a clear docstring describing its **single responsibility**
- Schema classes contain **no logic** — only data and serialization

## Testing

- Write tests for all new components
- Run the full suite before submitting: `python -m pytest tests/ -v`
- Test files go in the `tests/` directory

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commit messages
3. Ensure all tests pass
4. Update documentation if your changes affect the public API
5. Submit a pull request with a description of what and why

## Architecture Guidelines

- **Single Responsibility**: Each class does ONE thing
- **No God Objects**: Orchestrators delegate, never implement
- **Schema = Data**: Schema classes are pure data containers
- **Pluggable Design**: New strategies should be swappable without modifying core logic

## Reporting Issues

When reporting bugs, please include:
- Python version
- Ollama version and model name
- Error traceback
- Steps to reproduce
