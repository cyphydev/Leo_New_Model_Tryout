import pandas as pd
import jsonlines
from tqdm import tqdm
import pickle
import bisect
from possibility_model import PossibilityModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RESPONSE_IN_DAYS = 14

class PossibilityModel:

    def __init__(self, pickle_addr = None):

        self.model_ready = False
        self.pandas_dataset = None
        self.verbose = True
        self.maxium_timestamp = None
        self.minium_timestamp = None



    def build_df_from_pickle(self, pickle_addr):
        """
        Load the model from a pickle file.
        """
        self.pandas_dataset = self.process_data(pd.read_pickle(pickle_addr))


    def calculate_rate_respond_in_2_weeks(self, timestamps, current_time=None):

        if current_time is None:
            current_time = self.maxium_timestamp
    
        # Recency: Time difference from the last message to the current time

        latest_message = timestamps[-1]
        target = (RESPONSE_IN_DAYS * 60 * 60 * 24) + current_time - latest_message

        # Consistency: Standard deviation of time intervals between messages
        if len(timestamps) > 2:
            intervals = [(timestamps[i] - timestamps[i-1]) for i in range(1, len(timestamps))]
            std = np.std(intervals)
            mean = np.mean(intervals)
            probability = stats.norm.cdf(target, mean, std) - stats.norm.cdf(0, mean, std)
            return probability
        else:
            return len(timestamps) * RESPONSE_IN_DAYS / ((self.maxium_timestamp - self.minium_timestamp) / (60 * 60 * 24))

    
    
    def process_data(self, df):
        """
        Process the data to get the pandas dataframe.
        """
        self.maxium_timestamp = max(df["timestamp"])
        self.minium_timestamp = min(df["timestamp"])
        # Assuming df is your DataFrame and 'author' is the column you want to group by
        cols_to_exclude = ['author', 'msg', 'timestamp', 'contentText']
        cols_to_average = [col for col in df.columns if col not in cols_to_exclude]

        # Define the aggregation dictionary.
        agg_dict = {col: ['mean', 'var'] for col in cols_to_average}
        agg_dict.update({col: list for col in ['msg', 'timestamp', 'contentText']})

        # Group by 'author' and apply the aggregation.
        df_grouped = df.groupby("author").agg(agg_dict).reset_index()
        # convert the timestamp to days

        df_grouped["response_rate"] = df_grouped[("timestamp", "list")].apply(self.calculate_rate_respond_in_2_weeks)
        data = df_grouped
        sett = set()

        for col, subcol in data.columns:
            if subcol != "":
                sett.add(col)
                data[f"{col}-{subcol}"] = data[col][subcol]

        for col in sett:
            del data[col]

        data["total_num_msg"] = data["msg-list"].apply(lambda x: len(x))

        data.drop(columns=["msg-list", "timestamp-list", "contentText-list"], inplace=True)

        df_more_than_5 = df_grouped[df_grouped["total_num_msg"] > 5]
        df_more_than_5.columns = df_more_than_5.columns.get_level_values(0)#.difference(['author', 'response_rate'])
            
        self.pandas_dataset = df_author_timeseries
        self.model_ready = True

    def check_model_ready(self):
        """
        Check if the model is ready.
        """
        if not self.model_ready:
            raise Exception("Model not ready, please load the data first.")
        
        
    
    def dump_model(self, pickle_addr):
        """
        Dump the model to a pickle file.
        """
        if self.model_ready:
            pickle.dump(self, pickle_addr)
        else:
            print("Model not ready, please load the data first.")

    def predict(self, actor_id, prediction_time):
        """
        Get the possibility of a feature with a value in the dataset.
        """
        self.check_model_ready()
        response_rate = 0

        # if actor_id in self.dataset, use the historical data to calculate the possibility
        actor_data = self.pandas_dataset[self.pandas_dataset['author'] == actor_id]
        if len(actor_data) > 0:
            dates = actor_data['dayPublished'].iloc[0][0]
            counts = actor_data['dayPublished'].iloc[0][1]

            # get the index of the prediction_time
            cur_index = bisect.bisect(dates, prediction_time)
            in_3_days_index = bisect.bisect_left(dates, prediction_time - 3 * 24 * 60 * 60 * 1000)
            in_7_days_index = bisect.bisect_left(dates, prediction_time - 7 * 24 * 60 * 60 * 1000)
            in_14_days_index = bisect.bisect_left(dates, prediction_time - 14 * 24 * 60 * 60 * 1000)
            in_30_days_index = bisect.bisect_left(dates, prediction_time - 30 * 24 * 60 * 60 * 1000)
            # number of days that the actor has published within 3 days
            in_3_days = counts[in_3_days_index:cur_index]
            in_7_days = counts[in_7_days_index:cur_index]
            in_14_days = counts[in_14_days_index:cur_index]
            in_30_days = counts[in_30_days_index:cur_index]
            # calculate the response rate
            in_3_days_num, in_7_days_num, in_14_days_num, in_30_days_num = 0, 0, 0, 0
            in_3_days_rate, in_7_days_rate, in_14_days_rate, in_30_days_rate = 0, 0, 0, 0

            if len(in_3_days) > 0:
                in_3_days_num = sum(in_3_days)
                in_3_days_rate = (len(in_3_days)+1) / (3+1)
            if len(in_7_days) > 0:
                in_7_days_num = sum(in_7_days)
                in_7_days_rate = max(response_rate, (len(in_7_days)+1) / (7+1))
            if len(in_14_days) > 0:
                in_14_days_num = sum(in_14_days)
                in_14_days_rate = max(response_rate, (len(in_14_days)+1) / (14+1))
            if len(in_30_days) > 0:
                in_30_days_num = sum(in_30_days)
                in_30_days_rate = max(response_rate, (len(in_30_days)+1) / (30+1))
            
            response_rate = len(dates) / ((self.end_time - self.start_time) // (24 * 60 * 60 * 1000 + 1))
            response_rate = max(response_rate, in_3_days_rate, in_7_days_rate, in_14_days_rate, in_30_days_rate)
            if self.verbose:
                print("--------------------")
                print("Number of days that the actor has published within 3 days: {}".format(len(in_3_days)))
                print("Number of messages that the actor has published within 3 days: {}".format(in_3_days_num))
                print("Number of days that the actor has published within 7 days: {}".format(len(in_7_days)))
                print("Number of messages that the actor has published within 7 days: {}".format(in_7_days_num))
                print("Number of days that the actor has published within 14 days: {}".format(len(in_14_days)))
                print("Number of messages that the actor has published within 14 days: {}".format(in_14_days_num))
                print("Number of days that the actor has published within 30 days: {}".format(len(in_30_days)))
                print("Number of messages that the actor has published within 30 days: {}".format(in_30_days_num))
                print("Number of days that the actor has published in total: {}".format(len(dates)))
                print("Number of messages that the actor has published in total: {}".format(sum(counts)))
                print("Response rate of actor {} is {}".format(actor_id, response_rate))
            
            return response_rate
        else:
            # if actor_id not in self.dataset, use the total data to calculate the possibility
            return 1 / ((self.end_time - self.start_time) // (24 * 60 * 60 * 1000) + 1)

        
        
    

if __name__ == "__main__":
    dataset_addr = '/data/shared/incas/Israel_Hamas/data/challenge_problem_two_21NOV.jsonl'
  
    model = PossibilityModel()
    print(model.get_possibility('actor_1', 'Tom Cruise'))