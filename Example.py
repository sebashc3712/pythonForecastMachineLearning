
import pandas as pd # pandas to dataframe handle
from ForecastingML import bestForecastModel # Machine learning algorithm
from sklearn.externals import joblib # joblib library




df=pd.read_csv('insurance.csv',sep=',')

cat1={'southwest':1,'southeast':2,'northwest':3,'northeast':4}

df['region']=df['region'].map(cat1)

cat2={'yes':1,'no':2}

df['smoker']=df['smoker'].map(cat2)

cat3={'female':1,'male':2}

df['sex']=df['sex'].map(cat3)


label=df['charges']
df.drop(['charges'], axis=1, inplace=True)


# Here the best model and his score are saved in a and b respectively and then
# the model is saved into a file named <bestForcastModel.pkl>
a,b=bestForecastModel(df,label)
joblib.dump(a, 'bestForecastModel.pkl')

