import luigi
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from math import cos, asin, sqrt, pi

class CleaningPipeline(luigi.Task):

    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='clean.csv')

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        raw_df_tweet = pd.read_csv(self.tweet_file, header=0, encoding='unicode_escape')
        raw_df_tweet.dropna(subset=['tweet_coord'], how='any', inplace=True)
        df2 = raw_df_tweet.tweet_coord.str.split(expand=True, )
        df2 = pd.DataFrame(df2)
        raw_df_tweet['Latitude'] = df2[0].str.replace('[', '').str.replace(',', '').astype(float).fillna(0.0)
        raw_df_tweet['Longitude'] = df2[1].str.replace(']', '').str.replace(',', '').astype(float).fillna(0.0)
        raw_df_tweet = raw_df_tweet.query('Latitude != 0.0 & Longitude != 0.0')
        raw_df_tweet.to_csv(self.output().path, index=False)


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.
        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')
    clean_file = luigi.Parameter(default='clean.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def requires(self):
        return CleaningPipeline()

    def run(self):
        cities_df = pd.read_csv(self.cities_file, header = 0, encoding='unicode_escape')
        clean_df = pd.read_csv(self.clean_file, header= 0, encoding='unicode_escape')
        clean_df = clean_df[['Latitude', 'Longitude', 'airline_sentiment']]
        clean_df['y'] = clean_df['airline_sentiment'].apply(lambda x:  1 if 'neutral' in x else 2 if 'positive' in x else 0)
        x0 = []

        def havsindist(latitude_1, longitude_1, latitude_2, longitude_2):
            h = 0.5 - cos((latitude_2 - latitude_1) * pi / 180) / 2 + cos(latitude_1 * pi / 180) * cos(
                latitude_2 * pi / 180) * (1 - cos((longitude_2 - longitude_1) * pi / 180)) / 2
            return 12742 * asin(sqrt(h))

        def Eucdist(latitude_1, longitude_1, latitude_2, longitude_2):
            h = (latitude_1-latitude_2)**2 + (longitude_1-longitude_2)**2
            return (sqrt(h))

        for row1 in clean_df[:].itertuples(index=True, name='Pandas'):
            x1 = getattr(row1, "Latitude")
            x2 = getattr(row1, "Longitude")
            temp1 = []
            temp2 = []

            for row2 in cities_df[:].itertuples(index=True, name='Pandas'):
                y1 = getattr(row2, 'latitude')
                y2 = getattr(row2, 'longitude')
                y3 = getattr(row2, 'asciiname')
                y0 = Eucdist(x1, x2, y1, y2)
                temp1.append(y0)
                temp2.append(y3)

            temp_df = pd.DataFrame({'city_name': temp2, 'Distance': temp1})
            temp_df.sort_values(by=['Distance'], ascending=[1], inplace=True)
            x0.append(temp_df.iloc[0, 0])
            del(temp_df)
        clean_df['Closest_City'] = pd.DataFrame(x0)
        # Create features dataframe with Closest city variable(string) and sentiment encodings:y
        features_df = clean_df[['Closest_City', 'airline_sentiment','y']]
        features_df.to_csv(self.output_file,index=False,header=True)

class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    features_file = luigi.Parameter(default='features.csv')
    output_file = luigi.Parameter(default='model.pkl')

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def requires(self):
        yield TrainingDataTask()


    def run(self):
        features_df = pd.read_csv(self.features_file , header = 0, encoding='unicode_escape')
        encoding_features = pd.get_dummies(features_df['Closest_City'], prefix=None)
        x_features = pd.DataFrame(encoding_features)
        y = features_df['y']
        clf = LogisticRegression(random_state=0, multi_class='auto', solver='lbfgs')
        clf = clf.fit(x_features, y)
        outFile = open(self.output().path, 'wb')
        pickle.dump(clf, outFile, protocol=pickle.HIGHEST_PROTOCOL)
        outFile.close()


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.
        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    features_file = luigi.Parameter(default='features.csv')
    output_file = luigi.Parameter(default='scores.csv')
    input_file =luigi.Parameter(default='model.pkl')
    def output(self):
        return luigi.LocalTarget(self.output_file)

    def requires(self):
        yield TrainModelTask()

    def run(self):
        # generate cities to be scored
        features_df = pd.read_csv(self.features_file, delimiter=",")
        encoding_features = pd.get_dummies(features_df['Closest_City'], prefix=None)
        x_features = pd.DataFrame(encoding_features)

        inFile=open(self.input_file, 'rb')
        clf = pickle.load(inFile)
        inFile.close()

        pred = clf.predict(x_features)

        pred_df = pd.DataFrame(pred)
        pred_df = pred_df.rename(columns={0: 'model_predicted_score'})
        pred_prob = clf.predict_proba(x_features)
        prob_df = pd.DataFrame(pred_prob)
        prob_df = prob_df.rename(columns={0: 'Prob(0):negative', 1: 'Prob(1):neutral', 2: 'Prob(2):positive'})

        output_df = pd.concat([prob_df, pred_df, features_df['Closest_City']], axis=1)
        output_df = output_df.rename(columns={'Closest_City': 'City_asciiname'})

        output_df.drop_duplicates(subset='City_asciiname', keep='first', inplace=True)
        output_df.sort_values(by=['model_predicted_score'], ascending=[0], inplace=True)
        final_df= output_df[['City_asciiname', 'Prob(0):negative', 'Prob(1):neutral', 'Prob(2):positive']]
        final_df.to_csv(self.output_file, index=False, header=True)


if __name__ == "__main__":
    luigi.build([CleaningPipeline(),TrainingDataTask(),TrainModelTask(),ScoreTask()], workers=5)
