# python-package-template

This is a template repository for Python package projects.

## In this README :point_down:

- [Features](#features)
- [Usage](#usage)
  - [Initial setup](#initial-setup)
  - [Creating releases](#creating-releases)
- [Projects using this template](#projects-using-this-template)
- [FAQ](#faq)
- [Contributing](#contributing)

## Features

This template repository comes with all of the boilerplate needed for:

⚙️ Robust (and free) CI with [GitHub Actions](https://github.com/features/actions):
  - Unit tests ran with [PyTest](https://docs.pytest.org) against multiple Python versions and operating systems.
  - Type checking with [mypy](https://github.com/python/mypy).
  - Linting with [ruff](https://astral.sh/ruff).
  - Formatting with [isort](https://pycqa.github.io/isort/) and [black](https://black.readthedocs.io/en/stable/).

🤖 [Dependabot](https://github.blog/2020-06-01-keep-all-your-packages-up-to-date-with-dependabot/) configuration to keep your dependencies up-to-date.

📄 Great looking API documentation built using [Sphinx](https://www.sphinx-doc.org/en/master/) (run `make docs` to preview).

🚀 Automatic GitHub and PyPI releases. Just follow the steps in [`RELEASE_PROCESS.md`](./RELEASE_PROCESS.md) to trigger a new release.

## Usage

### Initial setup

1. [Create a new repository](https://github.com/allenai/python-package-template/generate) from this template with the desired name of your project.

    *Your project name (i.e. the name of the repository) and the name of the corresponding Python package don't necessarily need to match, but you might want to check on [PyPI](https://pypi.org/) first to see if the package name you want is already taken.*

2. Create a Python 3.8 or newer virtual environment.

    *If you're not sure how to create a suitable Python environment, the easiest way is using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). On a Mac, for example, you can install Miniconda using [Homebrew](https://brew.sh/):*

    ```
    brew install miniconda
    ```

    *Then you can create and activate a new Python environment by running:*

    ```
    conda create -n my-package python=3.9
    conda activate my-package
    ```

3. Now that you have a suitable Python environment, you're ready to personalize this repository. Just run:

    ```
    pip install -r setup-requirements.txt
    python scripts/personalize.py
    ```

    And then follow the prompts.

    :pencil: *NOTE: This script will overwrite the README in your repository.*

4. Commit and push your changes, then make sure all GitHub Actions jobs pass.

5. (Optional) If you plan on publishing your package to PyPI, add repository secrets for `PYPI_USERNAME` and `PYPI_PASSWORD`. To add these, go to "Settings" > "Secrets" > "Actions", and then click "New repository secret".

    *If you don't have PyPI account yet, you can [create one for free](https://pypi.org/account/register/).*

6. (Optional) If you want to deploy your API docs to [readthedocs.org](https://readthedocs.org), go to the [readthedocs dashboard](https://readthedocs.org/dashboard/import/?) and import your new project.

    Then click on the "Admin" button, navigate to "Automation Rules" in the sidebar, click "Add Rule", and then enter the following fields:

    - **Description:** Publish new versions from tags
    - **Match:** Custom Match
    - **Custom match:** v[vV]
    - **Version:** Tag
    - **Action:** Activate version

    Then hit "Save".

    *After your first release, the docs will automatically be published to [your-project-name.readthedocs.io](https://your-project-name.readthedocs.io/).*

### Creating releases

Creating new GitHub and PyPI releases is easy. The GitHub Actions workflow that comes with this repository will handle all of that for you.
All you need to do is follow the instructions in [RELEASE_PROCESS.md](./RELEASE_PROCESS.md).

## Projects using this template

Here is an incomplete list of some projects that started off with this template:

- [ai2-tango](https://github.com/allenai/tango)
- [cached-path](https://github.com/allenai/cached_path)
- [beaker-py](https://github.com/allenai/beaker-py)
- [gantry](https://github.com/allenai/beaker-gantry)
- [ip-bot](https://github.com/abe-101/ip-bot)
- [atty](https://github.com/mstuttgart/atty)

☝️ *Want your work featured here? Just open a pull request that adds the link.*

## FAQ

#### Should I use this template even if I don't want to publish my package?

Absolutely! If you don't want to publish your package, just delete the `docs/` directory and the `release` job in [`.github/workflows/main.yml`](https://github.com/allenai/python-package-template/blob/main/.github/workflows/main.yml).

## Contributing

If you find a bug :bug:, please open a [bug report](https://github.com/allenai/python-package-template/issues/new?assignees=&labels=bug&template=bug_report.md&title=).
If you have an idea for an improvement or new feature :rocket:, please open a [feature request](https://github.com/allenai/python-package-template/issues/new?assignees=&labels=Feature+request&template=feature_request.md&title=).

# Anomalous TS: Time Series Anomaly Detection

A Python package for detecting anomalies in time series data using multiple detection methods and advanced preprocessing techniques.

## Overview

Anomalous TS provides robust tools for identifying unusual patterns and outliers in time series data. It implements multiple detection algorithms, supports both univariate and multivariate time series, and includes comprehensive preprocessing and visualization capabilities.

## Key Features

- **Multiple Detection Methods**:
  - Isolation Forest based detection
  - K-means clustering based detection
  - Support for both point and contextual anomalies
  - Confidence scores for detected anomalies

- **Advanced Preprocessing**:
  - Multiple imputation methods for missing values
  - Various scaling options
  - Seasonal decomposition
  - Trend analysis

- **Visualization Tools**:
  - Interactive anomaly plots
  - Cluster visualization
  - Confidence score visualization
  - Preprocessing results visualization

- **Time Series Features**:
  - Sliding window analysis
  - Seasonal pattern recognition
  - Trend detection
  - Support for various time frequencies

## Installation

```bash
pip install anomalous-ts
```

## Quick Start

```python
import pandas as pd
from anomalous_ts import TimeSeriesIsolationForest

# Load your time series data
data = pd.Series(your_data)

# Initialize detector
detector = TimeSeriesIsolationForest(
    window_size=24,  # for daily seasonality
    seasonal_period=24
)

# Detect anomalies
anomalies, scores = detector.detect_and_visualize(data)

# Print results
print(f"Found {anomalies.sum()} anomalies")
```

## Detailed Example

```python
from anomalous_ts import (
    TimeSeriesKMeansDetector,
    TimeSeriesPreprocessor,
    TimeSeriesVisualizer
)

# Initialize components
preprocessor = TimeSeriesPreprocessor(
    imputation_method='hybrid',
    scaling_method='robust'
)

visualizer = TimeSeriesVisualizer()

# Initialize detector
detector = TimeSeriesKMeansDetector(
    window_size=24,
    n_clusters=3,
    seasonal_period=24,
    preprocessor=preprocessor,
    visualizer=visualizer
)

# Detect and visualize anomalies
anomalies, scores = detector.detect_and_visualize(your_data)
```

## Key Use Cases

1. **Financial Time Series**:
   - Detect unusual trading patterns
   - Identify market anomalies
   - Monitor trading volumes

2. **IOT and Sensor Data**:
   - Detect equipment failures
   - Identify sensor malfunctions
   - Monitor system health

3. **Business Metrics**:
   - Detect unusual sales patterns
   - Identify website traffic anomalies
   - Monitor performance metrics

4. **Environmental Data**:
   - Detect unusual weather patterns
   - Identify environmental anomalies
   - Monitor climate indicators

## Features in Detail

### Preprocessing Options

- **Imputation Methods**:
  - Linear interpolation
  - Spline interpolation
  - KNN imputation
  - Forward/backward fill
  - Hybrid methods

- **Scaling Methods**:
  - Standard scaling
  - Robust scaling
  - Min-max scaling

### Detection Algorithms

1. **Isolation Forest**:
   - Unsupervised detection
   - Handles high-dimensional data
   - Provides anomaly scores

2. **K-means Detection**:
   - Cluster-based detection
   - Pattern recognition
   - Distance-based scoring

### Visualization Capabilities

- Time series plots with highlighted anomalies
- Confidence score visualization
- Cluster visualization
- Preprocessing results visualization

## Requirements

- Python ≥ 3.8
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy
- Statsmodels

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
Made using Claude:
   Anthropic. (2024). Claude (Version 3.5) [AI Assistant]. https://anthropic.com/claude

If you use this package in your research, please cite:

```bibtex
@software{anomalous_ts,
  title = {Anomalous TS: Time Series Anomaly Detection},
  author = {inkenbrandt},
  year = {2024},
  url = {https://github.com/inkenbrandt/anomalous_ts}
}
```

## Support

For support, please:
1. Check the [documentation](https://anomalous-ts.readthedocs.io)
2. Create an issue on [GitHub](https://github.com/yourusername/anomalous_ts/issues)
3. Contact the maintainers