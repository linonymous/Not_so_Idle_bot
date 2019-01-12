import math


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
    # hardcore values
    pred_length = pred_data.size(0)
    rmse_list = []
    deviation_list = []
    for j in range(0, pred_length-time_slot_length):
        se = 0
        start_time_slot = j
        end_time_slot = start_time_slot + time_slot_length
        timed_pred_data = pred_data.iloc[start_time_slot:end_time_slot, 1]
        for i in range(0, time_slot_length):
            se += math.pow((timed_pred_data[i]-val), 2)
        rmse = math.sqrt((se/time_slot_length))
        rmse_list.append(rmse)

    # Calculate the optimal rmse value, and the deviation of other rmse values from the optimal value
    min_rmse = min(rmse_list)
    for j in range(0, pred_length - time_slot_length):
        deviation_list.append((abs(min_rmse - rmse_list[j])/min_rmse)*100)
    return deviation_list, rmse_list


def calc_tolerance(wt):
    """
    returns the tolerance values
    :param wt: weight indication importance in selecting the time slot interval
    :return: return the tolerance from optimal value
    """
    return 1 - wt


def calc_conf(deviation, tolerance, mape):
    """
    Calculate the confidence of the time slot based on deviation, tolerance, MAPE. At this point deviation must be less
    than tolerance
    :param deviation: deviation from optimal time slot
    :param tolerance: accepted deviation
    :param mape: MAPE test error of predicted target
    :return: return the confidence
    """
    return (1 - ((mape / 100) * (deviation/tolerance))) * 100


class Selector:
    def __init__(self, pred_target, mape_list):
        """
        Selector to select the optimal time for the execution of process based on predicted targets
        :param pred_target: list of predicted targets CSVFileManagers
        :param mape_list: test MAPE list of all the targets
        """
        self.pred_target = pred_target
        self.MAPE_list = mape_list

    def select(self, wt, time_slot_length):
        """
        Selects the optimal time for the process to run
        Glossary:
        rmse - Root mean squared error from optimal target; defined per time slot per time series
        tolerance - % accepted percentage deviation from optimal time slot; defined per target series
        deviation - Actual % deviation from the optimal time slot; defined per time slot per target series
        confidence - % of confidence in selected time slot; defined per time slot per time series
        max-confidence - Max of all the values of the confidence; defined as a single value for all time slots of all time series
        :param wt: list of weights measuring the importance of each predicted target in deciding the time
        :param time_slot_length: Length of time slot
        :return:
        """
        # weights must be provided for all the targets with the same order as pred_target
        if wt.length != self.pred_target.length or wt.length != self.MAPE_list.length:
            print("Error: Length mismatch between two arrays")
        no_targets = wt.length
        rmse_list_list = []
        deviation_list_list = []
        tolerance_list = []

        # Find the minimum from predicted targets and calculate the rmse of the time slot sequence w.r.t it
        for i in range(0, no_targets):
            min_val = self.pred_target[i].data.loc[:, 1].min()
            # Calculate list of rmses for particular target list pred_target[i].data, each value in rmse_list is rmse
            # of window of time slot length. This window is moved by one time interval each time till end of the target,
            # therefore length of the rmse_list is pred_target[i].data.length - time_slot_length
            rmse_list, deviation_list = calc_rmse(self.pred_target[i], min_val, time_slot_length)

            # Append individual rmse list of target to list of rmses and append deviation list to the list of deviations
            rmse_list_list.append(rmse_list)
            deviation_list_list.append(deviation_list)

            # Calculate the tolerance based on the weights assigned, tolerance is inversely proportional to weights
            tolerance_list.append(calc_tolerance(wt[i]))

        max_conf = 0
        selected_time_start = -1

        # We assumed here that the length of all the predicted targets is same
        pred_length = self.pred_target[0].data.size(0)
        for i in range(0, pred_length - time_slot_length):
            flg = 0

            # Check if selecting time slot is in the acceptance of the tolerance, if not we need to break, and start
            # over from immediate next window
            for j in range(0, no_targets):
                if deviation_list_list[j][i] > tolerance_list[j]:
                    break
                if j + 1 == no_targets:
                    flg = 1

            # Calculate the confidence
            conf_sum = 0
            if flg == 1:
                for j in range(0, no_targets):
                    conf = calc_conf(deviation_list_list[j][i], tolerance_list[j], self.MAPE_list[j])
                    conf_sum += conf
                # We take mean confidence in account, I think we already have taken care of the weights in
                # calculating the individual confidence, so taking mean should not be biased and should not account
                #  again the relative weights of targets
                if (conf_sum/no_targets) > max_conf:
                    selected_time_start = i
                    max_conf = (conf_sum/no_targets)
        return selected_time_start, max_conf
