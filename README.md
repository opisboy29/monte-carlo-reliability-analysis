# Monte Carlo Reliability Analysis Framework

A comprehensive Python framework for analyzing system reliability using Monte Carlo simulation. Perfect for making data-driven decisions about hardware upgrades, infrastructure investments, and capacity planning.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

## 🎯 Features

- **Monte Carlo Simulation Engine**: Run thousands of scenarios to model system reliability
- **Multiple Statistical Distributions**: Support for Exponential, Weibull, and Gamma distributions
- **Comprehensive Visualization**: Professional charts and graphs for stakeholder presentations
- **Configurable Analysis**: Environment-driven configuration for different scenarios
- **Business-Ready Reports**: Executive summaries with clear recommendations
- **Extensible Architecture**: Easy to adapt for various hardware configurations

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/monte-carlo-reliability-analysis.git
cd monte-carlo-reliability-analysis
pip install -r requirements.txt
```

### Basic Usage

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your system specifications
nano .env

# Run the analysis
python monte_carlo_reliability.py
```

## 📊 Use Cases

- **Server Upgrade Analysis**: Quantify reliability improvements from hardware upgrades
- **Vendor Comparison**: Compare different hardware vendors based on reliability metrics
- **Capacity Planning**: Determine optimal system configurations for target reliability
- **Risk Assessment**: Evaluate failure probabilities for business continuity planning
- **Investment Justification**: Provide statistical backing for infrastructure investments

## 🔧 Configuration

The framework uses environment variables for configuration. Key parameters include:

```env
# System configuration
CURRENT_CPU_COUNT=14
CURRENT_SSD_COUNT=24
CURRENT_RAM_COUNT=32

# Failure rates (FIT)
CURRENT_CPU_FIT=50e-9
CURRENT_SSD_FIT=150e-9
CURRENT_RAM_FIT=100e-9

# Simulation parameters
N_SIMULATIONS=10000
CONFIDENCE_LEVEL=95
```

See `.env.example` for complete configuration options.

## 📈 Output

The framework generates:

1. **Statistical Analysis**: MTBF calculations, confidence intervals, failure probabilities
2. **Visual Dashboard**: Four-panel comparison showing distributions and key metrics
3. **Executive Summary**: Business-ready recommendations and justifications
4. **Detailed Logs**: Step-by-step analysis progress and validation

## 🧮 Methodology

### Statistical Modeling

- **CPU/RAM Components**: Exponential distribution (constant failure rate)
- **Storage Devices**: Weibull distribution (wear-out patterns)
- **Complex Components**: Gamma distribution (bathtub curve modeling)

### Monte Carlo Process

1. Sample failure times from component distributions
2. Calculate system failure time (minimum of all components)
3. Repeat for thousands of iterations
4. Analyze statistical distributions of results
5. Generate confidence intervals and risk assessments

### Reliability Theory

Uses series reliability model where system fails when any component fails:
```
System Reliability = ∏ Component Reliability
System Failure Rate = Σ Component Failure Rates
```

## 📚 Examples

### Basic Server Analysis
```python
from monte_carlo_reliability import ReliabilityConfig, MonteCarloReliabilityAnalyzer

config = ReliabilityConfig('examples/server_config.env')
analyzer = MonteCarloReliabilityAnalyzer(config)
results = analyzer.run_simulation()
```

### Custom Configuration
```python
config = ReliabilityConfig()
config.config['current_cpu_count'] = 20
config.config['upgraded_ssd_count'] = 50
results = analyzer.run_simulation()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/yourusername/monte-carlo-reliability-analysis.git
cd monte-carlo-reliability-analysis
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running Tests

```bash
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with the Python scientific computing stack (NumPy, SciPy, Matplotlib)
- Inspired by reliability engineering best practices
- Statistical methods based on industry standards for failure analysis

## 📞 Support

If you have questions or need help:

1.. Connect on [LinkedIn](https://www.linkedin.com/in/nabima-reyhan-687b1b130/)

---

**Made with ❤️ for the reliability engineering community**
