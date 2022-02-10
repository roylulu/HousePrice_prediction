import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
raw_data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
raw_data.head()
raw_data.info()
raw_data.describe(include='all')
raw_data.isnull().sum()
data = raw_data.drop(['id','date'],axis = 1)
data.head()

#改变图形大小
plt.subplots(figsize=(15,10))

#绘制热度图
sns.heatmap(data.corr(), annot = True,linewidths=.5,linecolor='black',fmt = '1.1f')
plt.title('correlation heatmap',size = 18)
plt.show()
sns.pairplot(data,vars = ["price","bedrooms","bathrooms","floors"], kind ="reg")

#判断数值是否已使用
data['bedrooms'].unique()

#绘图
plt.subplots(figsize=(12,10))
sns.boxplot(x="bedrooms", y = "price",data= data)
plt.title('price vs no of bedrooms',size = 18)
plt.show()


data['bathrooms'].unique()
plt.subplots(figsize=(12,10))
_=plt.hist(data['bathrooms'],color='salmon')
_=plt.xlabel('no of bathrooms')
_=plt.ylabel('price')
plt.title('price vs no of bathrooms',size = 18)
plt.show()


data['floors'].unique()
plt.subplots(figsize=(15,10))
#seaborn method for plotting a bargraph.
sns.barplot(x="floors",y="price",data=data,palette="Blues_d")
plt.title('price vs no of floors',size = 18)
plt.show()

#改变外观
sns.relplot(x="sqft_living",y="price",hue="waterfront",col="view",palette=["g", "r"],data=data)
sns.pairplot(data, vars = ["sqft_living","sqft_lot","sqft_basement","sqft_living15","sqft_lot15"], kind ="reg")

data['grade'].unique()
plt.subplots(figsize=(12,10))
sns.boxplot(x="grade", y = "price",data= data,palette="Set3")
plt.title('price vs grade',size = 18)
plt.show()

data['condition'].unique()
plt.subplots(figsize=(12,10))
sns.boxplot(x="condition", y = "price",data= data,palette="Set1")
plt.title('price vs condition',size = 18)
plt.show()
plt.subplots(figsize=(15,10))

#分析价格与特征值关系
sns.scatterplot(x="grade",y="price",hue="condition",size="condition",sizes=(20, 200),data=data)
plt.title('relationship between price,grade and condition',size = 18)
plt.show()
plt.subplots(figsize=(15,15))

#matplotlib方法提供50个观测值的计数分布以及年份
data.yr_built.value_counts().head(50).plot.pie(autopct='%1.1f%%')
plt.title('year built pie chart',size = 18)
plt.show()
data.yr_built.value_counts()
plt.subplots(figsize=(12,10))
plt.scatter(data['long'],data['lat'],color="purple")

#设置图表区间
plt.xlim(-180,180)
plt.ylim(-180,180)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('distribution of houses',size = 18)
plt.show()

plt.subplots(figsize=(12,10))
plt.scatter(data['long'],data['lat'],color="purple")

#根据散点图选取坐标
plt.xlim(-121.2,-122.6)
plt.ylim(47,47.9)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('distribution of houses closeup',size = 18)
plt.show()
plt.subplots(figsize=(12,10))
y=data['price']
x=data['sqft_living']
plt.scatter(x,y,color='green')
plt.title('price vs living area',size = 18)
plt.show()
plt.show()
plt.subplots(figsize=(8,8))

sns.distplot(data['price'],color='crimson')
plt.title('pdf of price',size = 18)
plt.show()
plt.subplots(figsize=(8,8))

#建立99%置信区间
q = data['price'].quantile(0.99)

data_1 = data[data['price']<q]
data_1.describe(include = "all")
sns.distplot(data_1['price'],color='crimson')
plt.title('pdf of price less than the 99th percentile',size = 18)
plt.show()

plt.subplots(figsize=(8,8))
sns.distplot(data_1['bedrooms'],color='m')
plt.title('pdf of no of bedrooms',size = 18)
plt.show()

plt.subplots(figsize=(8,8))
p=data_1['bedrooms'].quantile(0.99)
data_2 = data_1[data_1['bedrooms']<p]
sns.distplot(data_2['bedrooms'],color='m')
plt.title('pdf of no of bedrooms less than 99th percentile',size = 18)
plt.show()

plt.subplots(figsize=(12,10))
sns.distplot(data_2['sqft_living'],color='pink')
plt.title('pdf of living area',size = 18)
plt.show()

plt.subplots(figsize=(8,8))
p = data_2['sqft_living'].quantile(0.99)
data_3 = data_2[data_2['sqft_living']<p]
sns.distplot(data_3['sqft_living'],color='pink')
plt.title('pdf of price less than 99th percentile',size = 18)
plt.show()

plt.subplots(figsize=(8,8))
sns.distplot(data_2['sqft_basement'],color='cyan')
plt.title('pdf of basement',size = 18)
plt.show()

plt.subplots(figsize=(8,8))
p = data_3['sqft_basement'].quantile(0.99)
data_4 = data_3[data_3['sqft_basement']<p]
sns.distplot(data_4['sqft_basement'],color='cyan')
plt.title('pdf of basement with 99th percentile',size = 18)
plt.show()

data_model=data_3.reset_index(drop=True)
data_model.describe(include="all")

#转换存储price
log_price=np.log(data_model['price'])
data_model['log_price'] = log_price
data_model.head()
plt.subplots(figsize=(8,8))
y=data_model['log_price']
x=data_model['sqft_living']
plt.scatter(x,y,color='green')
plt.title('log price vs living area',size = 18)
plt.show()

data_model.columns.values

#计算多重影响因素
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_model[[ 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above','sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat','long', 'sqft_living15', 'sqft_lot15']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif

data_cleaned = data_model.drop(['sqft_living15','sqft_lot15','sqft_above','sqft_lot','sqft_basement',],axis = 1)
data_cleaned.head()
data_cleaned.columns.values

x1=data_cleaned[[ 'bedrooms', 'bathrooms', 'sqft_living',
       'floors', 'waterfront', 'view', 'condition', 'grade',
        'yr_built', 'yr_renovated', 'zipcode', 'lat',
       'long']]
y=data_cleaned[['log_price']]
x= sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

data_reg=data_model.drop(['long','yr_renovated'],axis=1)
data_reg.head()
x1=data_cleaned[[ 'bedrooms', 'bathrooms', 'sqft_living',
       'floors', 'waterfront', 'view', 'condition', 'grade',
        'yr_built', 'lat','zipcode']]
y=data_cleaned[['log_price']]
x= sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

x
#新建表格
data_with_predictions = pd.DataFrame({'const':1,'bedrooms':[3,3],'bathrooms':[1,2.25],'sqft_living':[1180,2570],'floors':[1,2],'waterfront':[0,0],'view':[0,0],'condition':[3,3], 'grade':[7,7],'yr_built':[1955,1951],'lat':[47.5112,47.7210],'zipcode':[98103,98002]})
data_with_predictions=data_with_predictions[['const','bedrooms','bathrooms','sqft_living','floors','waterfront','view','condition','grade','yr_built','lat','zipcode']]
data_with_predictions

#预测数值
predictions = results.predict(data_with_predictions)
predictions
data_with_predictions['predictions'] = predictions
data_with_predictions
pred_price=np.exp(data_with_predictions['predictions'])
pred_price
data_with_predictions['predicted_price']=pred_price
data_with_predictions

