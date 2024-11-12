
# Question 1: Data Structure and Processing Pipeline
class IrisDataProcessor:
    def __init__(self):
        self.data = load_iris()
        self.df = None
        self.scaler = StandardScaler()

    def prepare_data(self):
        # Convert to DataFrame and name columns
        self.df = pd.DataFrame(data=self.data.data, columns=self.data.feature_names)
        self.df['target'] = self.data.target

        # Scale features
        features = self.df.drop(columns=['target'])
        self.df[features.columns] = self.scaler.fit_transform(features)

        # Train-test split
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_feature_stats(self):
        # Print basic statistical information
        print("Feature Statistics:\n", self.df.describe())


