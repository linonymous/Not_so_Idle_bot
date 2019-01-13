
import pandas as pd
import numpy as np
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

    def forecast(self, compare_data=None, column_list=None, file_path=None, show=0):
        """
        Forecast the data using matplotlib.pyplot
        :param compare_data: Dataframe for the data which needs to be \
                            plotted for comparision
        :param column_list: List of two elements indicating \
                            two columns participating in comparison
        :param file_path: result file path
        :return: None
        """
        plt.figure(figsize=(18.0, 10.0))
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
        plt.plot(x, y, 'r')
        if compare_data is not None and type(compare_data) is pd.DataFrame:
            # TO-DO: decide what to do with other parameter of the DF for comparison
            x_cp = compare_data[column_list[0]].tolist()
            y_cp = compare_data[column_list[1]].tolist()
            plt.plot(x_cp, y_cp, 'b')

        # plt.xticks(range(len(x)), x, rotation='vertical')
        if show == 1:
            plt.show()
        if file_path != None:
            plt.savefig(file_path, figsize=(10, 10), dpi=100)


if __name__ == "__main__":
    PATH = 'C://Users//Mahesh.Bhosale//PycharmProjects//Idle_bot//Dataset//data//IO_STAT//IO_STAT-06.csv'
    #csv_mgr = CSVFileManager(filename=PATH, interval=60)
    pf=pd.DataFrame(np.arange(0,4), columns=['A'])
    pf['B'] = ['a', 'b', 'c', 'd']
    csv_mgr = CSVFileManager(df=pf, interval=60)
    print(pf)
    #csv_mgr.get_by_interval(interval=3600)
    #print(csv_mgr.data)
    #csv_mgr.read_file()
    pf1 = pd.DataFrame(np.arange(0,4) * 3, columns=['A'])
    pf1['B'] = ['a', 'b', 'c', 'd']
    pf1 = pf1.iloc[1:3, :]
    print(pf1)
    dv = DataVisualizer(csv_mgr=csv_mgr, x_col='B', y_col='A')
    dv.forecast(compare_data=pf1, column_list=['B', 'A'], file_path="C://Users//Mahesh.Bhosale//PycharmProjects//Idle_bot//Dataset//data//d.png")