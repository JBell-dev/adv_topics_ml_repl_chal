import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from os.path import dirname, abspath

# 1. --- Copy your .csv data file into the folder "csv-data" + the respective environment and specify the name of the file(s).
# Example:
environment = "Atari"
filenames = ["fullatari_envp_run_data.csv", "fullNOisyNet_run_data.csv", "fullRLE_run_data.csv", "fullrndenvp_run_data.csv"]
nicknames = ["PPO", "NoisyNet", "RLE", "RND"]
colors = [(127/255, 127/255, 127/255), (238/255, 134/255, 54/255), (84/255, 113/255, 171/255), (81/255, 158/255, 63/255)]

# 2. --- Specify which column shall represent the x-axis and which ones the y-axis.
# Example:
globalStep = "_step"
plotThis = ["charts/game_score", "charts/game_score_lower", "charts/game_score_upper"]

# Specify what exactly is being plotted how.
linePlot = [0]
rangePlot = [1, 2]

plotTitle = "Game Score with Range"
yTitle = "Game Score"
xTitle = "Global Step"
legendPosition = "lower right"

# 3. --- Specify output. I strongly recommend to choose pdf so that we can include it in our report without loss of quality.
outputFilename = "plotAtari"
filetype = "pdf"

# Done --- You can run the script now. You can find the plot in the folder "output".

# Optional Hyperparameters
uncertainty_corridors_alpha = 0.5
linewidth = 1

def main():
    csv = {}
    max_x, max_y = 0, 0
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, file in enumerate(filenames):
        csv[nicknames[idx]] = pd.read_csv(dirname(abspath(__file__)) + "/csv-data/" + environment + "/" + file)[[globalStep] + plotThis]
        running_maxX, running_maxY = csv[nicknames[idx]][globalStep].iloc[-1], csv[nicknames[idx]][plotThis].max().max()
        if len(linePlot) != 0:
            for jdx in linePlot:
                df = csv[nicknames[idx]][[globalStep, plotThis[jdx]]].dropna()
                ax.plot(df[globalStep], df[plotThis[jdx]], color=colors[idx], linewidth=linewidth)#, label=nicknames[idx]+" "+plotThis[jdx])
        if len(rangePlot) == 2:
            plot = [plotThis[i] for i in rangePlot]
            df = csv[nicknames[idx]][[globalStep] + plot].dropna()
            ax.fill_between(df[globalStep], df[plot[0]], df[plot[1]], facecolor=colors[idx], alpha=uncertainty_corridors_alpha, label=nicknames[idx])
        if running_maxX > max_x:
            max_x = running_maxX
        if running_maxY > max_y:
            max_y = running_maxY
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y*1.1)
    ax.set_title(plotTitle)
    ax.set_ylabel(yTitle)
    ax.set_xlabel(xTitle)
    ax.legend(loc=legendPosition)
    plt.savefig(dirname(abspath(__file__))+"/output/"+outputFilename+"."+filetype)

if __name__ == "__main__":
    main()
    print('File ' + outputFilename + '.' + filetype + ' has been plotted successfully and you can find it in folder "ouput".')