from Dataset.data_handler.CSVFileManager import CSVFileManager
from Dataset.data_visualization.DataVisualizer import DataVisualizer
import math
import numpy as np


def calc_rmse(pred_target, val, time_slot_length):
    """
    Calculate the rmse on the pred_target.data for moving time window of size time_slot_length,
    note that mse is not from mean but val, in most common case here is from min
    :param pred_target: CSVFileManager object containing predicted dataframe
    :param val: value w.r.t which the rmse is calculated
    :param time_slot_length: Length od the time slot window
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
        rmse_list.append(rmse)
    return rmse_list


def calc_tolerance(wt):
    """
    returns the tolerance values
    :param wt: weight indication importance in selecting the time slot interval
    :return:
    """
    return 1-wt


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
        optimal_rmse_list = []

        # Find the minimum and calculate the rmse of the time slot sequence from it
        for i in range(0, l):
            min_val = self.pred_target[i].data.loc[:, 1].min()

            # Calculate list of rmse for particular target list pred_target[i].data, each value in rmse_list is rmse
            # of window of time slot length. This window is moved by one time interval each time till end of the target,
            # therefore length of the rmse_list is pred_target[i].data - time_slot_length
            rmse_list = calc_rmse(self.pred_target[i], min_val, time_slot_length)

            # Calculate the optimal rmse value and its index and stor it as tuple in the list
            index_min = np.argmin(np.asarray(rmse_list))
            min_rmse = rmse_list.min()
            optimal_rmse_list.append((min_rmse, index_min))

            # Add individual rmse list of target to list of rmses
            rmse_list_list.append(rmse_list)

        # Calculate the tolerance based on weights assigned, greater weights indicate lesser tolerance
        for i in range(0, l-time_slot_length):
            min_rmse  = optimal_rmse_list[i][0]
            tolerance = calc_tolerance(wt[i])