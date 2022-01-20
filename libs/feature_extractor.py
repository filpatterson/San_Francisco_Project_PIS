import numpy as np
import pandas as pd

class FeatureExtractor:
    def make_streets_intersections_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """generate streets intersections columns based on available streets data

        Args:
            df (pd.DataFrame): dataset with address column

        Returns:
            pd.DataFrame: modified dataset with new columns, where one describes street and
                            another one intersections
        """
        intersection = df.Address.str.extract(r'(\w+\s\w+\s[/]\s\w+\s\w+)').fillna(' ')
        street       = df.Address.str.extract(r'\d+\s\w+\s\w+\s(\w+\s\w+)').fillna(' ')

        intersection = intersection.rename(columns = {0 : 'Intersection'})
        street       = street.rename(columns = {0 : 'Street'})

        df = pd.concat([df, street, intersection], axis=1)
        return df


    def make_time_cols(self, df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """generate new time columns based on string-formatted timestamps

        Args:
            df (pd.DataFrame): dataset with string-formatted timestamps
            timestamp_column (str): name of column with string-formatted
                                    timestamps

        Returns:
            pd.DataFrame: modified dataframe with new time columns
        """
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df["Year"]           = df[timestamp_column].dt.year
        df["Month"]          = df[timestamp_column].dt.month
        df["Day"]            = df[timestamp_column].dt.date
        df["DayOfYear"]      = df[timestamp_column].dt.dayofyear
        df["Hour"]           = df[timestamp_column].dt.hour
        df["Minute"]         = df[timestamp_column].dt.minute
        return df


    def make_weekday_to_num(self, df: pd.DataFrame, column: str):   
        """generate new column with where string-typed weekdays will be replaced
        by numerical values
        
        Args:
            df (pd.DataFrame): dataset for which generation of new weekday
                                column based on string weekday representation
            column (str): name of column where are located weekdays in string form
            
        Returns:
            pd.DataFrame: modified dataset with weekdays transformed in numerical
                            form
        """
        week_day_mapper = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                           'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df["weekdayNumerical"] = df[column].map(week_day_mapper).astype("int64")
        return df


    def make_address_encoding_col(self, df: pd.DataFrame, delimeter: str, column: str):
        """perform address encoding of the dataset
        
        Args:
            df (pd.DataFrame): dataset where is required to encode addresses
            delimeter (str): delimeter for the street data
            column (str): name of the addresses column in the dataset

        Returns:
            pd.DataFrame: modified dataframe with encoded addresses
        """
        address = df[column].apply(lambda record: any([delimeter in record]))
        df['address_encoded'] = np.fromiter(address, dtype=bool).astype(int)
        return df


    def make_seasons_col(self, df: pd.DataFrame, column: str):
        """create seasons column depending on month columns

        Args:
            df (pd.DataFrame): dataset with available months column
                                in numerical form
            column (str): name of the column that contains months data
                            in numerical form

        Returns:
            pd.DataFrame: modified dataset with added seasons column
        """
        df['Season'] = df[column]
        df['Season'] = df['Season'].map({1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 2, 6 : 3, 
                                         7 : 3, 8 : 3, 9 : 4, 10: 4, 11: 4, 12: 1})
        return df
    
    
    def get_harmonic_tuple(self, value, period=24):
        """remap cyclical data from line axis to the circular axis ->
        important to make model understand cycle

        Args:
            value (numerical): numerical value for which is required to
                                generate harmonic representation
            period (numerical): period covered by the cyclic data (24 for hours,
                                12 for months, 7 for weekdays and so on, defaults to 24)

        Returns:
            pair of coordinates of the current numerical value on the X and Y axis
        """
        value *= 2 * np.pi / period
        return np.cos(value), np.sin(value)


    def get_outlier_removed_col(self, df: pd.DataFrame, column, up_threshold, low_threshold):
        """remove outliers by chosen column

        Args:
            column: column of column group for which is required to pick and
                    remove outliers
            up_threshold: biggest value possible in given column or column group
            low_threshold: smallest value possible in given column or column group

        Returns:
            pd.DataFrame: modified dataframe with removed outliers for specific column
                            or column group
        """
        return df[(df[column] < up_threshold) & (df[column] > low_threshold)]


    def get_count_table(self, df: pd.DataFrame, category, time):
        """count elements by given category and time category
        
        Args:
            df (pd.DataFrame): dataset to count elements
            category: what records category to count
            time: what time metrics to use for counting records

        Returns:
            pd.DataFrame: count table to count records by category and time
        """
        count_df = df
        count_df["Count"] = 1
        count_df = count_df[[category, time, 'Count']]
        count_df = count_df.groupby([category, time]).agg('sum')
        count_df = count_df.reset_index()
        return count_df


    def get_cols_names_below_threshold(self, df: pd.DataFrame, threshold):
        """get columns that are below specified threshold

        Args:
            df (pd.DataFrame): dataset
            threshold (numerical): threshold all values below which will be recorded

        Returns:
            pd.DataFrame: dataset of columns that are below specified threshold
        """
        categories_below_threshold = df[df["Count"] < threshold]['Category'].unique()
        return categories_below_threshold


    def get_count_table_by_street(self, df: pd.DataFrame):
        """Count elements by street
        
        Args:
            df (pd.DataFrame): dataset

        Returns:
            pd.DataFrame: count table by streets
        """
        local_df = pd.DataFrame({})
        for elem in df['Street'].unique():
            if elem != 0:
                count = df[df['Street'] == elem]['Category'].value_counts()
                local_df = pd.concat([local_df, pd.Series(count, name=elem)], axis=1)
                
        return local_df


    def get_nan_records(self, df: pd.DataFrame):
        """get all NaN values records in any column
        
        Args:
            df (pd.DataFrame): dataset

        Returns:
            pd.DataFrame: dataset of all NaN values records in any column
        """
        return df[df.isnull().any(axis=1)]
    
    def get_true_pred_perc(self, predictions, answers):        
        """find how percents of answers were answered correctly by the predictor

        Returns:
            percentage of correct predictions
        """
        collisions = 0
        for index in range(len(answers)):
            if answers[index] == predictions[index]:
                collisions += 1

        return collisions * 100/len(answers)