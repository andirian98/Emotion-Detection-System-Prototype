import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize(filename):
    df = pd.read_csv(filename)
    x = df['labels']
    total = df['labels'].count()

    unique, frequency = np.unique(x,
                                  return_counts=True)

    # convert both into one numpy array
    #count = np.asarray((unique, frequency))

    position = []

    left = len(unique)
    for i in range(0,left):
        position.append(i)

    colors = ['tab:blue','tab:red','tab:green','tab:brown','tab:grey','tab:cyan','tab:orange']
    fig1, axs1 = plt.subplots()
    # plotting a bar chart
    axs1.bar(position, frequency, tick_label=unique,
            width=0.8, color=colors)

    # naming the x-axis
    axs1.set_xlabel('Emotion Labels')
    # naming the y-axis
    axs1.set_ylabel('Total Emotion Captured')
    # plot title
    axs1.set_title('Total Player\'s Emotion Captured')

    my_colors = ['tab:blue','tab:red','tab:green','tab:brown','tab:grey','tab:cyan','tab:orange']
    my_explode = []
    for i in range(0, len(unique)):
        my_explode.append(frequency[i] / total)

    fig2, axs2 = plt.subplots()
    axs2.pie(frequency, labels=unique,autopct='%1.1f%%',colors=my_colors,startangle=180,explode=my_explode,shadow=True)
    axs2.set_title('Emotion Label\'s Percentage')
    axs2.axis('equal')

    # function to show the plot
    plt.tight_layout()
    plt.show()

