import os
import pandas as pd


class CSVFileManager:

    DELIMITER = ','

    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def read_file(self):
        self.data = pd.read_csv(self.filename, delimiter=self.DELIMITER, index_col=None)

    def write_file(self, filename=None):
        if filename is None:
            filename = self.filename
        self.data.to_csv(filename, index=False)

    def delete_column(self, column_index=None):
        if column_index is not None:
            self.data.drop(column_index, axis=1, inplace=True)

    def delete_row(self, row_index=None):
        pass


def merge_csv_files(path, file_identifier=None, output_file=None, columns_to_drop=None):
    """
    Function to merge the files
    :param path: path to CSV files
    :param output_file: file name for output
    :param columns_to_drop: list of columns to drop
    :param file_identifier: substring that matches the file names in path
    :return: Merged Data frame
    """

    if path is None or output_file is None:
        return None
    if file_identifier is None:
        file_identifier = '.'

    file_objs = [CSVFileManager(path + filename) for filename in os.listdir(path) if file_identifier in filename]
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


if __name__ == "__main__":

    path = 'path-to-csvs'
    merge_csv_files(path=path)