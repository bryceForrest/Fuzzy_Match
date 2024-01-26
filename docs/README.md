# Overview

The `Fuzzy_Match` library is a fuzzy match search system designed for finding close matches within a corpus of documents. It is implemented in Python, and therefore (obviously) requires a Python installation.

# Python and Anaconda Installation

You only need to choose **one** of the following installation options. The official Python installation uses `pip` for package management, while Anaconda uses primarily `conda` for package management. The official Python installation might be the easier option, and should meet all the needs of `Fuzzy_Match`, but research the difference if you are unsure.

## From the Official Python Website

1. Visit the [official Python website](https://www.python.org/).
2. Navigate to the "Downloads" section.
3. Download the latest version of Python for your operating system (Windows, macOS, or Linux).
4. Run the installer and follow the installation instructions.
5. During installation, make sure to check the box that says "Add Python to PATH."

## Anaconda Installation

1. **Download Anaconda:**
   - Visit the [Anaconda download page](https://www.anaconda.com/products/distribution).
   - Download the appropriate installer for your operating system (Windows, macOS, or Linux).

2. **Run the Installer:**
   - Run the downloaded installer.
   - Follow the installation instructions.
   - During installation, you can choose whether to add Anaconda to your system PATH. It's recommended to select this option as it makes it easier to use Anaconda from the command line.

3. **Verification:**
   - After installation, open a new terminal or command prompt.
   - Type `conda --version` to check that Conda, the package manager included with Anaconda, is installed.

That's it! Once you have Anaconda installed, you can proceed with the library installation.


# Fuzzy_Match Installation (Windows)

Fuzzy_Match must be installed from the command line.

To use the library, you can install it from a local folder using pip:

```bash
# Navigate to the folder containing the library
cd "C:\path\to\your\Fuzzy_Match"

# Install the library
pip install .
```

Alternatively, you can use the following command to install directly:

```bash
pip install "C:\path\to\your\Fuzzy_Match"
```

# Usage
```python
# Import the Fuzzy_Match object from the Fuzzy_Match library
from Fuzzy_Match import Fuzzy_Match

# Initialize the Fuzzy_Match object
fuz = Fuzzy_Match()

# Assume themes is a table containg valid theme names that search terms
# can be matched to, and Theme_Name is the relevant column.
fuz.fit(themes['Theme_Name'])

# The variable input_data can be either a string, for a single search term,
# or an iterable container (list, table column, etc.) of search terms.
# The runtime between a single search term and several thousand are comparable,
# so it makes sense to search several at once when possible.
# The parameter top_n defines the number of matches desired for each search term.
results = fuz.search(input_data, top_n=num_matches)

# The output is of a dictionary of lists of values (matches, scores)
# so this line just forces it into a nice table.
output_table = pd.DataFrame.from_dict(results).T.explode([0,1]).reset_index()
```
