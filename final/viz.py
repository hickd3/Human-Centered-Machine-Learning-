'''
filename: viz.py
date: 12.16.24 (edits)
Authors: Dean Hickman Duilio Lucio
Purpose: Create a plot to visualize survey data'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data= {
    "Duration": [407, 229, 311, 583, 278, 532, 1168, 313, 292, 666, 382, 696, 389, 67220, 567, 298],
    "Time Riddle 2": [33.795, 29.788, 35.019, 46.506, 57.788, 21.984, 100.8, 37.057, 17.66, 42.656, 50.054, 27.864, 58.983, 32.111, 75.511, 67.636],
    "Click Riddle 2": [6, 5, 4, 4, 5, 4, 8, 2, 11, 7, 5, 5, 4, 8, 4, 10],
    "Time Logic 2": [21.432, 6.25, 8.454, 24.69, 9.631, 14.185, 82.877, 16.693, 11.19, 16.478, 9.575, 10.942, 13.665, 12.9, 17.241, 22.84],
    "Click Logic 2": [3, 3, 2, 3, 2, 3, 2, 6, 2, 3, 2, 2, 2, 2, 4, 2],
    "Time General 2": [14.386, 6.036, 14.424, 37.627, 11.398, 18.973, 38.827, 16.229, 7.863, 15.685, 6.381, 16.425, 10.577, 21.741, 17.812, 11.446],
    "Click General 2": [3, 2, 2, 2, 3, 3, 5, 2, 6, 2, 2, 2, 2, 2, 2, 2],
    "Time Riddle 4": [165.092, 57.836, 64.673, 213.373, 69.375, 180.852, 667.869, 177.342, 105.223, 171.727, 125.775, 385.411, 129.826, 100.887, 232.873, 103.955],
    "Click Riddle 4": [46, 10, 12, 17, 22, 11, 20, 14, 45, 34, 11, 24, 12, 32, 21, 34],
    "Time Logic 4": [100.444, 59.975, 116.038, 169.824, 76.642, 178.074, 125.257, 68.225, 121.972, 213.857, 120.138, 202.572, 137.405, 217.936, 155.196, 54.553],
    "Click Logic 4": [8, 4, 5, 5, 4, 4, 10, 4, 11, 5, 6, 4, 4, 19, 10, 4],
    "Time General 4": [39.245, 20.027, 53.589, 73.586, 43.363, 29.001, 85.749, 27.284, 18.351, 57.143, 45.876, 135.709, 25.392, 21.373, 50.214, 18.582],
    "Click General 4": [2, 4, 4, 5, 8, 10, 4, 6, 7, 7, 6, 5, 5, 5, 3, 3],
    "Q39": ["General", "General", "General", "Logic", "Riddles", "Riddles", "Riddles", "Riddles", "General", "Riddles", "Riddles", "Riddles", "Riddles", "Logic", "Riddles", "Riddles"],
    "Q40_1": [5, 7, 3, 3, 2, 4, 2, 2, 3, 3, 3, 3, 2, 6, 4, 6],
    "Q40_2": [6, 6, 5, 7, 7, 7, 5, 3, 4, 5, 6, 6, 5, 4, 7, 7],
    "Q40_3": [4, 3, 3, 6, 6, 6, 5, 5, 5, 7, 6, 6, 6, 7, 6, 7],
    "Time Riddle": [198.887, 87.624, 99.692, 259.879, 127.163, 202.836, 768.669, 214.399, 122.883, 214.383, 175.829, 413.275, 188.809, 132.998, 308.384, 171.591],
    "Click Group Riddle": [52, 15, 16, 21, 27, 15, 28, 16, 56, 41, 16, 29, 16, 40, 25, 44],
    "Accuracy Riddle 2": [0, 1, 1, 2, 1, 2, 0, 2, 0, 1, 1, 1, 2, 0, 2, 0],
    "Accuracy Logic 2": [1, 1, 1, 1, 1, 2, 2, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    "Accuracy General 2": [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 0, 0, 1],
    "Accuracy Block 2": [2, 4, 3, 4, 4, 5, 4, 4, 2, 4, 3, 4, 4, 1, 3, 2],
    "Accuracy Riddle 4": [3, 4, 1, 4, 2, 5, 1, 4, 0, 1, 3, 1, 4, 0, 1, 0],
    "Accuracy Logic 4": [2, 4, 1, 0, 2, 4, 3, 2, 1, 1, 3, 1, 2, 2, 3, 3],
    "Accuracy General 4": [3, 2, 2, 4, 2, 4, 1, 2, 2, 1, 3, 2, 3, 0, 3, 4],
    "Accuracy Group Riddle": [3, 3, 2, 4, 1, 2, 3, 1, 0, 2, 4, 1, 5, 3, 3, 2],
    "Accuracy Group Logic": [2, 5, 2, 4, 2, 2, 3, 4, 0, 3, 3, 3, 4, 3, 3, 0],
    "Accuracy Group General": [4, 3, 3, 2, 3, 2, 6, 2, 2, 2, 1, 3, 4, 4, 3, 4],
    "Accuracy Total": [10, 13, 8, 8, 8, 13, 12, 9, 7, 6, 9, 7, 18, 6, 9, 8]
}


df = pd.DataFrame(data)
sns.set_palette("muted")

def create_visualizations(df):
    df_accuracy = df[['Accuracy Riddle 2', 'Accuracy Logic 2', 'Accuracy General 2', 
                      'Accuracy Riddle 4', 'Accuracy Logic 4', 'Accuracy General 4']]
    df_accuracy = df_accuracy.melt(value_vars=['Accuracy Riddle 2', 'Accuracy Logic 2', 'Accuracy General 2', 
                                               'Accuracy Riddle 4', 'Accuracy Logic 4', 'Accuracy General 4'],
                                   var_name='Accuracy Type', value_name='Accuracy')

    # Adjust accuracy by scaling the values based on the type (Riddle: 2, Logic: 2, General: 4)
    df_accuracy['Adjusted Accuracy'] = df_accuracy['Accuracy'] 

    # Scale the accuracy values based on the type
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy Riddle 2', 'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 2
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy Logic 2', 'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 2
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy General 2', 'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 2
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy Riddle 4', 'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 4
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy Logic 4', 'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 4
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy General 4', 'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 2

    df_time = df[['Time Riddle 2', 'Time Logic 2', 'Time General 2', 'Time Riddle 4', 'Time Logic 4', 'Time General 4']]
    df_time = df_time.melt(value_vars=['Time Riddle 2', 'Time Logic 2', 'Time General 2', 
                                       'Time Riddle 4', 'Time Logic 4', 'Time General 4'],
                           var_name='Time Type', value_name='Time')

    df_clicks = df[['Click Riddle 2', 'Click Logic 2', 'Click General 2', 
                    'Click Riddle 4', 'Click Logic 4', 'Click General 4']]
    df_clicks = df_clicks.melt(value_vars=['Click Riddle 2', 'Click Logic 2', 'Click General 2', 
                                          'Click Riddle 4', 'Click Logic 4', 'Click General 4'],
                               var_name='Click Type', value_name='Clicks')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy plot
    sns.boxplot(x='Accuracy Type', y='Adjusted Accuracy', data=df_accuracy, ax=axes[0], hue='Accuracy Type', palette='Set1')
    axes[0].set_title('Adjusted Accuracy across size and group')
    axes[0].set_ylabel('Adjusted Accuracy')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)

    # Time plot
    sns.boxplot(x='Time Type', y='Time', data=df_time, ax=axes[1], hue='Time Type', palette='Set1')
    axes[1].set_title('Time across size and group')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=45)

    # Click plot
    sns.boxplot(x='Click Type', y='Clicks', data=df_clicks, ax=axes[2], hue='Click Type', palette='Set1')
    axes[2].set_title('Clicks across size and group')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('Clicks')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

create_visualizations(df)
