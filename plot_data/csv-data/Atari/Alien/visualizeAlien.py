import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from os.path import dirname, abspath

CUSTOM_PARAMS = {
    'figure.figsize': (8, 6),
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'grid.alpha': 0.5,
    'legend.fontsize': 12,
    #'legend.frameon': False,
    'axes.grid': True,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    # Font settings for LaTeX
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
}

mpl.rcParams.update(CUSTOM_PARAMS)

FIGURE_PARAMS = {
    "uncertainty_corridors_alpha" : 0.3,
    "filetype" : ".pdf",
    "dpi" : 600,
}

# 1. --- Copy your .csv data file into the folder "csv-data" + the respective environment and specify the name of the file(s).
# Example:
game = "Alien"
plot = "VMF Trajectory Length"
OUTPUT_FILENAME = "plot_" + game + "_" + plot + FIGURE_PARAMS["filetype"]

human_score = 7127.7
random_score = 227.8
normalize_scores = False

#filenames = ["Entropy/alien-ppo-entropy.csv", "Entropy/alien-noisynet-entropy.csv", "Entropy/alien-rle-entropy.csv", "Entropy/alien-rnd-entropy.csv"]
filenames = ["TrajectoryLength/alien-ppo-trajlen.csv", "TrajectoryLength/alien-rle-trajlen.csv", "alien-neural-adaptive-vmf.csv", "alien-queue-adaptive-vmf-trajlen.csv"]
#NICKNAMES = ["PPO", "NoisyNet", "RLE", "RND"]
NICKNAMES = ["PPO", "RLE", "NA-VMF", "Q-VMF"]
#COLORS = [(127/255, 127/255, 127/255), (238/255, 134/255, 54/255), (84/255, 113/255, 171/255), (81/255, 158/255, 63/255)]
COLORS = [(127/255, 127/255, 127/255), (84/255, 113/255, 171/255), "magenta", "purple"]

# 2. --- Specify which column shall represent the x-axis and which ones the y-axis.
# Example:
globalStep = "Step"
plotThis = "TrajLen"

yTitle = "Trajectory Length"
xTitle = "Global Step"
legendPosition = "best"

# Done --- You can run the script now. You can find the plot in the folder "output".

def main():
    def scoreNormalization(score, normalize=normalize_scores, h_score=human_score, r_score=random_score):
        if normalize:
            return (score - r_score)/(h_score - r_score)
        else:
            return score

    csv = {}
    _, ax = plt.subplots()
    for idx, file in enumerate(filenames):
        csv[NICKNAMES[idx]] = pd.read_csv(dirname(abspath(__file__)) + "/" + file)[[globalStep, plotThis]]
        csv[NICKNAMES[idx]]["Normalized"] = csv[NICKNAMES[idx]][plotThis].apply(scoreNormalization)
        if normalize_scores:
            df = csv[NICKNAMES[idx]].dropna()
            ax.plot(df[globalStep], df["Normalized"], color=COLORS[idx], label=NICKNAMES[idx])
        else:
            df = csv[NICKNAMES[idx]].dropna()
            ax.plot(df[globalStep], df[plotThis], color=COLORS[idx], label=NICKNAMES[idx])
    ax.set_xlabel(xTitle)
    ax.set_ylabel(yTitle)
    ax.legend(loc=legendPosition)
    plt.savefig(dirname(abspath(__file__)) + "/" + OUTPUT_FILENAME, dpi=FIGURE_PARAMS["dpi"])

if __name__ == "__main__":
    main()
    print('File ' + OUTPUT_FILENAME + ' has been plotted successfully.')