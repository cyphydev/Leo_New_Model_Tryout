from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class KMeansModel:
    def __init__(self, data_path, random_state=42):
        self.random_state = random_state
        scaler = StandardScaler()
        self.df = pd.read_pickle(data_path)
        self.X_scaled = scaler.fit_transform(self.df[self.df.columns.difference(['author', 'msg', 'timestamp', 'contentText'])])
        self.kmeans = None

    def elbow_method(self, show_plot=False, range_start=1, range_end=20):
        # Calculate the SSE for each value of k
        print(f'Running elbow method for k = {range_start} to {range_end}')
        
        sse = []
        for k in range(range_start, range_end):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(self.X_scaled)
            sse.append(kmeans.inertia_)

        if show_plot:
            plt.plot(range(1, 20), sse)
            plt.title('Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('SSE')
            plt.show()

        # Calculate the second derivative of the SSE
        second_derivative = np.diff(sse, n=2)

        # The elbow point is where the second derivative is the largest (most negative)
        elbow_point = np.argmin(second_derivative) + range_start + 1  # +1 to adjust since np.diff reduces the length by 1

        self.n_clusters = elbow_point
        print(f'Elbow point is at k = {elbow_point}')
        return elbow_point, sse


    def fit(self, n_clusters=None):
        # Assuming X is already provided in the correct format and just needs scaling
        if n_clusters is None:
            inputt = input('Enter number of clusters \nIf you want to specify the number, type a integer. \n Otherwise, press Enter: ')
            if inputt.isdigit():
                n_clusters = int(inputt)
            else:
                print('Running elbow method to find optimal number of clusters')
                show_plot = False
                inputt = input('Show elbow plot? (y/n): ')
                if inputt == 'y':
                    show_plot = True
                n_clusters, sse = self.elbow_method(show_plot=show_plot)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        kmeans.fit(self.X_scaled)
        self.kmeans = kmeans

    def add_labels_to_df(self):
        self.df['cluster'] = self.kmeans.labels_
        return self.df

    
