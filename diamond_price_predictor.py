# Loading the dataset
import pandas as pd
df=pd.read_csv("diamond.csv")

# Data Preprocessing
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
df.drop_duplicates(inplace=True)
cut_categories = ['Fair', 'Good', 'Very Good','Excellent','Ideal']
color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J','K','L','M']
clarity_categories = ['I3','I2','I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
symmetry_categories = ['Poor','Fair', 'Good', 'Very Good','Excellent']
polish_categories = ['Poor','Fair', 'Good', 'Very Good','Excellent']
y=df['price']
X = df.drop(columns=['price'],axis=1)
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns
numeric_transformer = StandardScaler()
or_transformer = OrdinalEncoder()
preprocessor = ColumnTransformer(
    [
        ("OrdinalEncoder", or_transformer, cat_features),
         ("StandardScaler", numeric_transformer, num_features),        
    ]
)
X = preprocessor.fit_transform(X)

# Splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape

# Model Building
# Decision Tree Regressor
from xgboost import XGBRegressor
model=XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Creating a pickle file for the classifier
import pickle
filename = 'diamond_price_prediction_tree_model.pkl'
pickle.dump(model, open(filename, 'wb'))