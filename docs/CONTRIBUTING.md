# Contributing to Charon

Thank you for your interest in contributing to Charon! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/charon.git`
3. Add upstream: `git remote add upstream https://github.com/Valkyra-Labs/charon.git`
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Make your changes
5. Run tests: `cargo test --all`
6. Run benchmarks: `cargo bench`
7. Commit your changes: `git commit -am 'Add amazing feature'`
8. Push to the branch: `git push origin feature/amazing-feature`
9. Open a Pull Request

## Development Setup

### Prerequisites

- Rust 1.70 or later
- (Optional) CUDA toolkit for GPU acceleration
- (Optional) ONNX Runtime for model testing

### Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# With all features
cargo build --all-features
```

### Testing

```bash
# Run all tests
cargo test --all

# Run specific module tests
cargo test --lib audio

# Run with output
cargo test -- --nocapture

# Run integration tests
cargo test --test '*'
```

### Benchmarking

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench audio_bench
```

## Code Style

- Follow Rust standard formatting: `cargo fmt`
- Run clippy for linting: `cargo clippy -- -D warnings`
- Write documentation for public APIs
- Include tests for new functionality

## Areas for Contribution

### High Priority

- [x] Complete Candle backend implementation
- [x] Real-time CPAL integration
- [x] Pre-trained model zoo
- [x] WebAssembly support

### Medium Priority

- [ ] Additional audio processing algorithms
- [ ] More comprehensive tests
- [ ] Performance optimizations
- [ ] Documentation improvements

### Low Priority

- [ ] Python bindings (PyO3)
- [ ] GUI application
- [ ] Additional examples
- [ ] Benchmarking suite expansion

## Commit Messages

Follow conventional commits format:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Example: `feat: add SIMD-optimized convolution operator`

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md if applicable
5. Request review from maintainers

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what's best for the community

## Questions?

Open an issue or start a discussion on GitHub!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
