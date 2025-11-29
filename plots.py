#!/bin/python3
#graphs.sh


import pandas as pd
import matplotlib.pyplot as plt

# Read timing data
data = pd.read_csv('timeData.csv')

# Convert time to milliseconds for better readability
data[' Time.'] = data[' Time.'] * 1000  # Convert seconds to milliseconds

# Define the three groups of Algorithm for plotting
plot_groups = [
    {'Algorithm': ['thrust'], 'title': 'Thrust Solution', 'filename': 'thrustPlot.png'},
    {'Algorithm': ['singlethread'], 'title': 'Quick Sort vs STL', 'filename': 'singlethreadPlot.png'},
    {'Algorithm': ['multithread'], 'title': 'Merge Sort vs STL', 'filename': 'multithreadPlot.png'}
]

# Generate a separate plot for each group
for group in plot_groups:
    plt.figure(figsize=(10, 6))
    for algo in group['Algorithm']:
        algo_data = data[data['Algorithm'] == algo]
        plt.plot(algo_data[' Size'], algo_data[' Time'], marker='o', label=algo)

    # Customize plot
    plt.xlabel('Array Size')
    plt.ylabel('Time (milliseconds)')
    plt.title(group['title'])
    plt.legend()
    plt.grid(True)
    plt.xscale('log')  # Log scale for array sizes
    plt.yscale('log')  # Log scale for times to handle wide range
    plt.xticks(data[' Size'].unique(), labels=[f'{x:,}' for x in data[' Size'].unique()], rotation=45)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(group['filename'])
