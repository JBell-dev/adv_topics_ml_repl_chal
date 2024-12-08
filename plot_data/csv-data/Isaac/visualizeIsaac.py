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
game = "Cartpole"
plot = "Extrinsic Return"
OUTPUT_FILENAME = "plot_" + game + "_" + plot + FIGURE_PARAMS["filetype"]

human_score = 10250
random_score = 664
normalize_scores = False

#filenames = ["Score/stargunner-ppo-score.csv", "Score/stargunner-noisynet-score.csv", "Score/stargunner-rle-score.csv", "Score/stargunner-rnd-score.csv"]
filenames = ["extrinsic_returns_ppo_rndr.csv", "extrinsic_returns_ppo_running_avg1.csv", "extrinsic_returns_rle_running_avg.csv"]
#NICKNAMES = ["PPO", "NoisyNet", "RLE", "RND"]
NICKNAMES = ["PPO + white noise", "PPO", "RLE"]
#COLORS = [(127/255, 127/255, 127/255), (238/255, 134/255, 54/255), (84/255, 113/255, 171/255), (81/255, 158/255, 63/255)]
COLORS = [(238/255, 134/255, 54/255), (127/255, 127/255, 127/255), (84/255, 113/255, 171/255)]

# 2. --- Specify which column shall represent the x-axis and which ones the y-axis.
# Example:
globalStep = "Episode"
plotThis = "Extrinsic Return"

yTitle = "Running Average of Extrinsic Return"
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
            df = csv[NICKNAMES[idx]].fillna(0)
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