# Contributing to DeepSeek Trader

Thank you for considering contributing to DeepSeek Trader! This document provides guidelines for contributions.

## How to Contribute

### Reporting Bugs

If you find a bug:

1. Check if it's already reported in Issues
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Relevant logs

### Suggesting Features

For feature requests:

1. Check existing issues for similar requests
2. Create a new issue with:
   - Clear description of the feature
   - Use case and benefits
   - Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Write/update tests if applicable
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Create a Pull Request

### Code Style

**Python (Backend):**
- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small

**JavaScript (Frontend):**
- Use functional components
- Follow React best practices
- Use meaningful variable names
- Add comments for complex logic

### Testing

**Backend:**
```bash
cd backend
pytest tests/ -v
```

**Frontend:**
```bash
cd frontend
npm run test  # if tests are added
```

### Documentation

- Update README.md if adding major features
- Update SETUP.md for setup changes
- Update DEPLOYMENT.md for deployment changes
- Add inline comments for complex logic

## Development Setup

See [SETUP.md](SETUP.md) for development environment setup.

## Areas for Contribution

Here are some areas where contributions are especially welcome:

### High Priority

- [ ] Additional exchange integrations (Binance, Coinbase, etc.)
- [ ] PostgreSQL support as alternative to SQLite
- [ ] More comprehensive backtesting features
- [ ] Additional technical indicators
- [ ] Improved error handling and recovery
- [ ] Performance optimizations

### Medium Priority

- [ ] News sentiment analysis integration
- [ ] Telegram bot interface
- [ ] Mobile app (React Native)
- [ ] Advanced charting in dashboard
- [ ] Multi-strategy support
- [ ] Portfolio rebalancing features

### Lower Priority

- [ ] Additional LLM providers (OpenAI, Claude, etc.)
- [ ] Machine learning model training pipeline
- [ ] Social trading features
- [ ] Advanced portfolio analytics
- [ ] Options trading support

## Code Review Process

1. Maintainers will review PRs within 1-2 weeks
2. Feedback will be provided for changes needed
3. Once approved, PR will be merged
4. Your contribution will be credited

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Questions?

Feel free to open an issue for questions or discussions!

## Acknowledgments

Contributors will be added to the README.md credits section.

Thank you for helping make DeepSeek Trader better! 🚀


