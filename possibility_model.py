import pandas as pd
import jsonlines
from tqdm import tqdm
import pickle
import bisect

class PossibilityModel:

    def __init__(self, pickle_addr = None):

        self.model_ready = False
        self.pandas_dataset = None
        self.verbose = True

        if pickle_addr is not None:
            self = pickle.load(pickle_addr)
        else:
            go_get_date = input("Do you want to go get the data? (y/n)")
            if go_get_date == 'y':
                raw_data_addr = input("Please enter the address of the raw data: ")
                if raw_data_addr == "":
                    raw_data_addr = '/data/shared/incas/Israel_Hamas/data/challenge_problem_two_21NOV.jsonl'
                self.build_df(raw_data_addr)
            else:
                print("Could use build_df() to get data")
                

    def build_df(self, dataset_addr):
        """
        Get the data from the raw jsonl database.
        """
        # Open the JSONL file in read mode
        with jsonlines.open(dataset_addr) as reader:
            # Initialize an empty list to store the JSON objects
            data = []
            # Iterate over each line in the file
            for line in tqdm(reader, desc='Loading data from jsonl file'):
                # Process each line as a JSON object
                data.append(line)
            # Create a pandas DataFrame from the list of JSON objects
            df = pd.DataFrame(data)
            self.start_time = df['timePublished'].min()
            self.end_time = df['timePublished'].max()
            df_author = df[df['author'].notnull()]
            df_author_timeseries = df_author.groupby('author')['timePublished'].agg(list).reset_index()
            df_author_timeseries['list_length'] = df_author_timeseries['timePublished'].apply(len)
            df_author_timeseries = df_author_timeseries.sort_values(by='list_length', ascending=False)
            df_author_timeseries = df_author_timeseries.drop('list_length', axis=1)
            

            # convert the timestamp to days
            def cleanTimeHelper(lst):
                for i in range(len(lst)):
                    lst[i] -= lst[i] % (24 * 60 * 60 * 1000)

                dic = {}
                for i in lst:
                    if i in dic:
                        dic[i] += 1
                    else:
                        dic[i] = 1
                
                date = []
                count = []

                seq = sorted(dic.keys())
                for key in seq:
                    date.append(key)
                    count.append(dic[key])

                return [date, count]

                
            df_author_timeseries['dayPublished'] = df_author_timeseries['timePublished'].apply(cleanTimeHelper)

            
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