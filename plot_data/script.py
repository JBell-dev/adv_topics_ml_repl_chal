import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from os import makedirs
from os.path import exists, dirname, abspath

# 1. --- Copy your .csv data file into the folder "csv-data" and specify the name of the file.
# Example: 
filename = "test.csv"

# 2. --- Give the data columns a name as a list. Specify which element represents the x-axis, which one the y-axis, and which one (whether at all) the uncertainty.
# Example:
columns = ["x_value", "y1_value", "y1_uncertainty_lower", "y1_uncertainty_upper", "y2_value", "y2_uncertainty_lower", "y2_uncertainty_upper"]
x_value_index = 0
x_axis_name = "x-axis"
y_value_indices = [1, 4]
y_axis_name = "y-axis"
data_contains_y_uncertainties = True
y_uncertainty_lower_indices = [2, 5]
y_uncertainty_upper_indices = [3, 6]

# 
# 
# 
# 
# 
# 
# 
# #

# XX. --- (Optional) Specify a filetype. I strongly recommend to choose pdf so that we can include it in our report without loss of quality.
filetype = "pdf"

# Done --- You can run the script now. You can find the plot in the folder "output".

# Optional Hyperparameters
uncertainty_corridors_alpha = 0.2









def assertions():
    assert isinstance(x_value_index, int)
    assert len(y_value_indices) > 0
    assert isinstance(data_contains_y_uncertainties, bool)
    assert len(y_uncertainty_lower_indices) == len(y_uncertainty_upper_indices)



def main():
    csv = pd.read_csv(dirname(abspath(__file__)) + "/csv-data/" + filename)
    csv.columns = columns
    relevant_columns = [columns[x_value_index]] + [columns[index] for index in y_value_indices]
    data = csv[relevant_columns]
    ax = sns.lineplot(x=columns[x_value_index], y=columns[y_value_indices], data=data)
    #ax = sns.lineplot(x=x_axis_name, y=y_axis_name, data=data)
    if data_contains_y_uncertainties:
        for uncertainty in zip(y_uncertainty_lower_indices, y_uncertainty_upper_indices):
            ax.fill_between(columns[x_value_index], csv[columns[uncertainty[0]]], csv[columns[uncertainty[1]]], alpha=uncertainty_corridors_alpha)

if __name__ == "__main__":
    main()
    plt.show()
    print('File ' + filename + ' has been plotted successfully and you can find it in folder "ouput" as ' + filename + '.' + filetype)