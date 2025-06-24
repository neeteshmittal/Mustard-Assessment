#!/usr/bin/env python3
"""
Monte Carlo Simulation of a Golf Tournament

This script implements a Monte Carlo simulation of a golf tournament to determine
the probability of winning and finishing in the top five for each golfer.
"""


import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import List, Dict
import time
import argparse


def simulate_tournament(golfers: List[Dict], rounds: int = 4) -> List[Dict]:
    """
    Simulate a single golf tournament.

    Args:
        golfers: List of golfer dictionaries with name, mean, and std
        rounds: Number of rounds in the tournament (default: 4)

    Returns:
        List of golfer dictionaries with total scores added
    """
    tournament_results = []

    for golfer in golfers:
        # Simulate 4 rounds for each golfer
        round_scores = [random.normalvariate(golfer['mean'], golfer['std']) for _ in range(rounds)]

        # Add results to tournament_results
        tournament_results.append({
            'name': golfer['name'],
            'mean': golfer['mean'],
            'std': golfer['std'],
            'score': sum(round_scores)
        })

    # Sort golfers by score (ascending, since lower is better in golf)
    tournament_results.sort(key=lambda x: x['score'])

    return tournament_results


def calculate_fractions(tournament_results: List[Dict]) -> List[Dict]:
    """
    Calculate win fraction and top 5 fraction for each golfer.

    Args:
        tournament_results: Sorted list of golfer dictionaries with scores

    Returns:
        List of golfer dictionaries with win_fraction and top5_fraction added
    """
    n_golfers = len(tournament_results)
    results_with_fractions = tournament_results.copy()

    # Initialize fractions
    for golfer in results_with_fractions:
        golfer['win_fraction'] = 0
        golfer['top5_fraction'] = 0

    # Calculate win fraction
    winning_score = tournament_results[0]['score']
    winners = [g for g in results_with_fractions if g['score'] == winning_score]
    win_fraction = 1 / len(winners)

    for winner in winners:
        winner['win_fraction'] = win_fraction

    # Calculate top 5 fractions
    # Track positions and how many players are tied at each position
    position = 0
    while position < min(5, n_golfers):
        current_score = tournament_results[position]['score']
        tied_golfers = [g for g in results_with_fractions if g['score'] == current_score]
        n_tied = len(tied_golfers)

        # Calculate how many positions in top 5 are available
        positions_left = min(5 - position, n_tied)

        # Assign top 5 fraction
        top5_fraction = positions_left / n_tied

        for golfer in tied_golfers:
            golfer['top5_fraction'] = top5_fraction

        position += n_tied

    return results_with_fractions


def run_monte_carlo_simulation(golfers: List[Dict], iterations: int = 1000) -> pd.DataFrame:
    """
    Run a Monte Carlo simulation of golf tournaments.

    Args:
        golfers: List of golfer dictionaries with name, mean, and std
        iterations: Number of simulations to run

    Returns:
        DataFrame with golfer names and their average win and top 5 fractions
    """
    # Initialize results dictionary
    results = {golfer['name']: {'win_sum': 0, 'top5_sum': 0} for golfer in golfers}

    # Run iterations
    for i in range(iterations):
        # Simulate tournament
        tournament_results = simulate_tournament(golfers)
        # Calculate fractions
        results_with_fractions = calculate_fractions(tournament_results)

        # Update running sums
        for golfer in results_with_fractions:
            name = golfer['name']
            results[name]['win_sum'] += golfer['win_fraction']
            results[name]['top5_sum'] += golfer['top5_fraction']

    # Calculate averages and create DataFrame
    df_results = pd.DataFrame(columns=['Golfer', 'Win Probability', 'Top 5 Probability'])

    for name, stats in results.items():
        win_prob = stats['win_sum'] / iterations
        top5_prob = stats['top5_sum'] / iterations
        df_results = df_results._append({
            'Golfer': name,
            'Win Probability': win_prob,
            'Top 5 Probability': top5_prob
        }, ignore_index=True)

    # Sort by win probability (descending)
    df_results = df_results.sort_values('Win Probability', ascending=False)

    return df_results


def read_golfer_data(filename: str) -> List[Dict]:
    """
    Read golfer data from a file.

    Args:
        filename: Path to the file containing golfer data

    Returns:
        List of golfer dictionaries with name, mean, and std
    """
    try:
        # Try to read as CSV
        df = pd.read_csv(filename)
        golfers = []

        for _, row in df.iterrows():
            # Assuming the CSV has columns: Name, Mean, Std
            golfers.append({
                'name': row['Name'],
                'mean': row['Mean'],
                'std': row['Std']
            })

        return golfers
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # If CSV reading fails, try plain text format
        golfers = []
        with open(filename, 'r') as f:
            lines = f.readlines()

            for line in lines:
                # Skip empty lines or comments
                if not line.strip() or line.strip().startswith('#'):
                    continue

                # Try to parse the line
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    name = parts[0].strip()
                    try:
                        mean = float(parts[1].strip())
                        std = float(parts[2].strip())
                        golfers.append({
                            'name': name,
                            'mean': mean,
                            'std': std
                        })
                    except ValueError:
                        print(f"Warning: Could not parse data for {name}")

        return golfers


def plot_results(results: pd.DataFrame, top_n: int = 10):
    """
    Plot the win and top 5 probabilities for the top N golfers.

    Args:
        results: DataFrame with simulation results
        top_n: Number of top golfers to include in the plot
    """
    # Get top N golfers by win probability
    top_golfers = results.head(top_n)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot win probabilities
    ax1.barh(top_golfers['Golfer'][::-1], top_golfers['Win Probability'][::-1], color='green')
    ax1.set_title('Win Probability')
    ax1.set_xlabel('Probability')

    # Plot top 5 probabilities
    ax2.barh(top_golfers['Golfer'][::-1], top_golfers['Top 5 Probability'][::-1], color='blue')
    ax2.set_title('Top 5 Probability')
    ax2.set_xlabel('Probability')

    plt.tight_layout()
    plt.savefig('golf_simulation_results.png')
    print("Plot saved to golf_simulation_results.png")
    plt.show()


def show_single_tournament_example(golfers):
    """
    Demonstrate a single tournament simulation for validation.
    """
    # Simulate one tournament
    print("Simulating a single tournament:")
    tournament_results = simulate_tournament(golfers)

    # Calculate fractions
    results_with_fractions = calculate_fractions(tournament_results)

    # Display results
    print("\nTournament Leaderboard:")
    print("{:<25} {:<10} {:<15} {:<15}".format("Golfer", "Score", "Win Fraction", "Top 5 Fraction"))
    print("-" * 65)

    for i, golfer in enumerate(results_with_fractions, 1):
        print("{:<25} {:<10.2f} {:<15.2f} {:<15.2f}".format(
            golfer['name'],
            golfer['score'],
            golfer['win_fraction'],
            golfer['top5_fraction']
        ))

    # Verify calculations
    win_sum = sum(g['win_fraction'] for g in results_with_fractions)
    top5_sum = sum(g['top5_fraction'] for g in results_with_fractions)

    print("\nVerification:")
    print(f"Sum of win fractions: {win_sum} (should be 1.0)")
    print(f"Sum of top 5 fractions: {top5_sum} (should be 5.0)")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation of a Golf Tournament')
    parser.add_argument('--file', type=str, default='golfers.csv', help='Path to golfer data file')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of Monte Carlo iterations')
    parser.add_argument('--example', action='store_true', help='Show a single tournament example')
    parser.add_argument('--no-plot', action='store_true', help='Do not show plot')

    args = parser.parse_args()

    print("=" * 60)
    print("MONTE CARLO SIMULATION OF A GOLF TOURNAMENT")
    print("=" * 60)

    # Try to read data from file
    try:
        golfers = read_golfer_data(args.file)
        print(f"Read {len(golfers)} golfers from {args.file}")
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        return

    # Show the golfers data
    print("\nGolfer Data:")
    print("{:<25} {:<15} {:<15}".format("Name", "Mean", "Std"))
    print("-" * 55)
    for golfer in golfers:
        print("{:<25} {:<15.5f} {:<15.5f}".format(golfer['name'], golfer['mean'], golfer['std']))

    # Show a single tournament example if requested
    if args.example:
        show_single_tournament_example(golfers)

    # Set number of simulations
    n_simulations = args.iterations
    print(f"\nRunning {n_simulations} Monte Carlo simulations...")

    # Start timer
    start_time = time.time()

    # Run simulations
    results = run_monte_carlo_simulation(golfers, n_simulations)

    # End timer
    elapsed_time = time.time() - start_time
    print(f"Simulations completed in {elapsed_time:.2f} seconds")

    # Display results
    pd.set_option('display.precision', 5)
    print("\nResults:")
    print(results)

    # Save results to CSV
    results.to_csv('simulation_results.csv', index=False)
    print("Results saved to simulation_results.csv")

    # Show plot if not disabled
    if not args.no_plot:
        plot_results(results)


if __name__ == "__main__":
    main()