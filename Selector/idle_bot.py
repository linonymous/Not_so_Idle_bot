from Dataset.data_handler.CSVFileManager import CSVFileManager
from Dataset.data_visualization.DataVisualizer import DataVisualizer
import math
class Selector:
    def __init__(self, pred_target):
        """
        Selector to select the optimal time for the execution of process based on predicted targets
        :param pred_target: list of predicted targets CSVFileManagers
        """
        self.pred_target = pred_target

    def select(self, wt, time_slot_length):
        """
        Selects the optimal time for the process to run
        :param wt: list of weights measuring the importance of each predicted target in deciding the time
        :return:
        """
        # weights mus be provided for all the targets with the same order as pred_target
        if wt.length != self.pred_target.length:
            print("Error: Length mismatch between two arrays")
        l = wt.length
        rmse_list_list = []

        # Multiply the targets with the weights and find the minimum simultaneously
        for i in range(0, l):
            min_val = self.pred_target[i].data.loc[:, 1]
            rmse_list_list.append(calc_rmse(self.pred_target[i], min_val, time_slot_length))


def calc_rmse(pred_target, val, time_slot_length):
    """
    Calculate the rmse on the pred_target.data for moving time window of size time_slot_length,
    note that mse is not from mean but val, in most common case here is from min
    :param pred_target: CSVFileManager object containing predicted dataframe
    :param val: value w.r.t which the rmse is calculated
    :return: return the list of rmse values for the time slot sequence
    """
    pred_data = pred_target.data.loc[:, 1]  # Need to add the data column in the CSVFileManager itself, rather than
    # hardcore
    l = pred_data.size(0)
    rmse_list = []
    for j in range(0, l-time_slot_length):
        se = 0
        start_time_slot = j
        end_time_slot = start_time_slot + time_slot_length
        timed_pred_data = pred_data.iloc[start_time_slot:end_time_slot, 1]
        for i in range(0, time_slot_length):
            se += math.pow((timed_pred_data[i]-val), 2)
        rmse = math.sqrt((se/time_slot_length))
    return rmse_list.append(rmse)