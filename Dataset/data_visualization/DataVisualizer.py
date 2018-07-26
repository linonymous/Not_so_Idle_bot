
import pandas as pd
import matplotlib.pyplot as plt
from Dataset.data_handler.CSVFileManager import CSVFileManager


class DataVisualizer:

    def __init__(self, csv_mgr, x_col=None, y_col=None):
        self.csv_mgr = csv_mgr
        self.x_col = x_col
        self.y_col = y_col

    @property
    def csv_mgr(self):
        return self.__csv_mgr

    @csv_mgr.setter
    def csv_mgr(self, csv_mgr):
        if type(csv_mgr) is CSVFileManager:
            self.__csv_mgr = csv_mgr
        else:
            print("ERROR: Not a valid CSVFileManager object.")

    @property
    def x_col(self):
        return self.__x_col

    @x_col.setter
    def x_col(self, x_col):
        self.__x_col = x_col

    @property
    def y_col(self):
        return self.__y_col

    @y_col.setter
    def y_col(self, y_col):
        self.__y_col = y_col

    def forecast(self, compare_data=None, column_list=None):
        """
        Forecast the data using matplotlib.pyplot
        :param compare_data: Dataframe for the data which needs to be \
                            plotted for comparision
        :param column_list: List of two elements indicating \
                            two columns participating in comparison
        :return: None
        """
        if self.__x_col is None or self.__y_col is None:
            print("ERROR: Please update the values for x_col and y_col.")
            return None
        if compare_data is not None:
            if column_list is None:
                print("ERROR: Please provide the column list to compare")
                return None
            if len(column_list) < 2:
                print("ERROR: please provide columnlist with len 2 for plotting")
        x = self.__csv_mgr.data[self.__x_col].tolist()
        y = self.__csv_mgr.data[self.__y_col].tolist()
        if compare_data is not None and type(compare_data) is pd.DataFrame:
            # TO-DO: decide what to do with other parameter of the DF for comparison
            x_cp = compare_data[column_list[0]].tolist()
            y_cp = compare_data[column_list[1]].tolist()
            plt.plot(y_cp, 'b')
        plt.plot(range(len(x)), y, 'r')
        plt.xticks(range(len(x)), x, rotation='vertical')
        plt.show()

if __name__ == "__main__":
    PATH = 'C://Users//Mahesh.Bhosale//PycharmProjects//Idle_bot//Dataset//Data//IO_STATS.csv'
    csv_mgr = CSVFileManager(filename=PATH, interval=60)
    csv_mgr.get_by_interval(interval=3600)
    print(csv_mgr.data)
    csv_mgr.read_file()
    dv = DataVisualizer(csv_mgr=csv_mgr, x_col='timestamp', y_col='tps')
    dv.forecast()