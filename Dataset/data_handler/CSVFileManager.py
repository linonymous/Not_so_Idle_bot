import os
import pandas as pd
import numpy as np


class CSVFileManager:
    DELIMITER = ','

    def __init__(self, interval, filename=None, df=None):
        """
        Manages CSV data, and would be used a s primary class while writing and reading the data
        :param filename: string csv file to read the data
        :param interval: integer interval by which data is separated in sec
        :param df: dataframe from which data would be loaded
        """
        if filename is None and df is None:
            raise RequiredParameterError()
        self.filename = filename
        self.data = None
        self.interval = interval
        self.read_file()
        if filename is None:
            self.data = df

    def read_file(self):
        if self.filename is None:
            return
        self.data = pd.read_csv(self.filename, delimiter=self.DELIMITER, index_col=None, encoding='utf-8')

    def write_file(self, filename=None):
        if filename is None:
            raise RequiredParameterError()
        self.data.to_csv(filename, index=False)

    def delete_column(self, column_index=None):
        if column_index is not None:
            self.data.drop(column_index, axis=1, inplace=True)

    def delete_row(self, row_index=None):
        if row_index is None:
            raise RequiredParameterError()
        self.data.drop(self.data.index[[row_index]])

    def get_by_interval(self, interval=1):
        """
        Function separates self.data by given interval in sec, and updates self.data and self.interval to new values
        :param interval:Integer interval by which data would be separated in sec in data of CSVFileManager, must be greater
         than self.interval
        :return: None
        """
        if self.interval > interval:
            print('Error:002 Intervals can not be smaller than {}'.format(self.interval))
            return
        if self.interval == interval:
            return self.data
        gap = (int)(interval / self.interval)
        length = self.data.shape[0]
        tmp = (self.data.loc[[0]])
        rows = [i for i in range(gap, length, gap)]
        for row in rows:
            tmp = tmp.append(self.data.iloc[[row]], ignore_index=True)
        self.data = tmp
        self.interval = interval


class Error(Exception):
    """
   Base class for other exceptions
   """
    pass


class RequiredParameterError(Error):
    """
   Raised required parameters are are not passed
   """
    pass


def merge_csv_files(path, file_identifier=None, output_file=None, columns_to_drop=None):
    """
    Function to merge the files
    :param path: path to CSV files
    :param output_file: file name for output
    :param columns_to_drop: list of columns to drop
    :param file_identifier: substring that matches the file names in path e.g. "sec" or "min" for now
    :return: Merged Data frame
    """

    if path is None or output_file is None:
        raise RequiredParameterError()
    if file_identifier is None:
        file_identifier = '.'

    file_objs = [CSVFileManager(filename=path + filename, interval=60) for filename in os.listdir(path) if
                 file_identifier in filename]
    merged_data = None
    for file_obj in file_objs:
        file_obj.read_file()
        if columns_to_drop is not None and type(columns_to_drop) is list:
            file_obj.delete_column(columns_to_drop)
        if merged_data is None:
            merged_data = file_obj.data
        else:
            merged_data = merged_data.append(file_obj.data, ignore_index=True)
    file_objs[0].data = merged_data
    file_objs[0].write_file(filename=output_file)

