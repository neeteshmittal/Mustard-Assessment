# Monte Carlo Simulation of a Golf Tournament

This project implements a Monte Carlo simulation of a golf tournament. It takes a list of golfers with their ratings (mean score and standard deviation) and determines the probability of winning and finishing in the top five for each golfer.

## Requirements

- Python 3.6 or higher
- Libraries: numpy, pandas, matplotlib

## Setup

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # or
   venv\Scripts\activate  # On Windows
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the simulation with default parameters:

### Command-line options:

- `--file <path>`: Specify the path to the golfer data file (default: golfers.csv)
- `--iterations <number>`: Specify the number of Monte Carlo iterations (default: 1000)
- `--example`: Show a single tournament simulation example for validation
- `--no-plot`: Skip plotting the results

Example:


## Input File Format

The input file should be a CSV file with the following columns:
- Name: Golfer's name
- Mean: Mean score per round
- Std: Standard deviation of the score

## Output

The script generates:
1. Console output showing the simulation results
2. `simulation_results.csv`: CSV file with win and top 5 probabilities for each golfer
3. `golf_simulation_results.png`: Visual representation of the results