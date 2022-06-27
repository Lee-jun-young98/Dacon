# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:57:15 2021

@author: student
"""

## scoring = 'neg_mean_absolute_error' 로 설정해 주십시요. 일반적으로 scoring을 값이 클 수록 모델 성능이 좋은 것으로 사이킷런에서 인식하는데, mae는 값이 클 수록 모델 성능이 저하되는 것이므로 Negative 키워드를 붙여서 사용합니다.
# 앞에 -1을 곱해 양수로 만들어준다
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as LGB
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# %% 전처리
path = "D:/git_project/Parking_demand_prediction/parking_data"


# train = pd.read_csv("C:/dacon/parking_data/train.csv")
# test = pd.read_csv("C:/dacon/parking_data/test.csv")
# gender = pd.read_csv("C:/dacon/parking_data/age_gender_info.csv")
# submission = pd.read_csv("C:/dacon/parking_data/sample_submission.csv")

train = pd.read_csv(path + "/train.csv")
test = pd.read_csv(path + "/test.csv")
gender = pd.read_csv(path + "/age_gender_info.csv")
submission = pd.read_csv(path + "/sample_submission.csv")


gender.shape

train.shape, test.shape

train.head()

test.head()

train.isna().sum()
test.isna().sum()

train.columns

# 컬럼명 변경
train.columns = ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
                 '신분', '임대보증금', '임대료', '지하철', '버스', '단지내주차면수', '등록차량수']


test.columns = ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
                '신분', '임대보증금', '임대료', '지하철', '버스', '단지내주차면수']

gender.loc[:,"지역"] = gender.loc[:,'지역'].astype('category').cat.codes

# 지역명 숫자로 매핑
local_map = {}
for i, loc in enumerate(train['지역'].unique()):
    local_map[loc] = i

train['지역'] = train['지역'].map(local_map)
test['지역'] = test['지역'].map(local_map)


# 전용면적을 5의 배수로 변경
train['전용면적'] = train['전용면적']//5*5
test['전용면적'] = test['전용면적']//5*5


# 전용면적 상/하한 적용
idx = train[train['전용면적'] > 100].index
train.loc[idx, '전용면적'] = 100
idx = test[test['전용면적'] > 100].index
test.loc[idx, '전용면적'] = 1000


idx = train[train['전용면적'] < 15].index
train.loc[idx, '전용면적'] = 15
idx = test[test['전용면적'] < 15].index
test.loc[idx, '전용면적'] = 15

test['전용면적'].unique()

## 단지별 데이터 1차원으로 취합
columns = ['단지코드','총세대수','공가수','지역','단지내주차면수','지하철','버스']
target = '등록차량수'
area_columns = []
for area in train['전용면적'].unique():
    area_columns.append(f'면적_{area}')
    

new_train = pd.DataFrame()
new_test = pd.DataFrame()


for i, code in tqdm(enumerate(train['단지코드'].unique())):
    temp = train[train['단지코드']==code]
    temp.index = range(temp.shape[0])
    for col in columns:
        new_train.loc[i, col] = temp.loc[0, col]
        
    for col in area_columns:
        area = float(col.split('_')[-1])
        new_train.loc[i, col] = temp[temp['전용면적']==area]['전용면적별세대수'].sum()
    
    new_train.loc[i, '등록차량수'] = temp.loc[0, '등록차량수']
    

for i, code in tqdm(enumerate(test['단지코드'].unique())):
    temp = test[test['단지코드']==code]
    temp.index = range(temp.shape[0])
    for col in columns:
        new_test.loc[i, col] = temp.loc[0, col]
        
    for col in area_columns:
        area = float(col.split('_')[-1])
        new_test.loc[i, col] = temp[temp['전용면적']==area]['전용면적별세대수'].sum()

new_train_edited = pd.merge(new_train, gender, left_on = ["지역"], right_on = ["지역"], how = "left")
new_test_edited = pd.merge(new_test, gender, left_on = ["지역"], right_on = ["지역"], how = "left")


train_drop = ["C2085", "C1397", "C2431", "C1649", "C1036", "C1095", "C2051", "C1218", "C1894", "C2483", "C1502", "C1988"]
test_drop = ["C2675", "C2335", "C1327"]

for i in range(len(train_drop)):
    new_train_edited = new_train_edited.drop(new_train[new_train["단지코드"] == train_drop[i]].index)

for i in range(len(test_drop)):
    new_test = new_test.drop(new_test[new_test["단지코드"] == test_drop[i]].index)



#%% 학습
new_train_edited

# 결측치 처리
new_train_edited.isna().sum()

new_train_edited["지하철"].value_counts()

new_train.shape

    
new_train_edited["지하철nan"] = np.where(new_train_edited["지하철"] == "nan", 1, 0)
new_train_edited["지하철nan"]
new_test_edited["지하철nan"] = np.where(new_test_edited["지하철"] == "nan", 1, 0)


new_train = new_train_edited.fillna(0)
new_test = new_test_edited.drop(0)


# 학습
x_train = new_train.drop("등록차량수", axis = 1)
x_train = x_train.drop("단지코드", axis = 1)
y_train = new_train["등록차량수"]
x_test = new_test.iloc[:,1:]


# 표준화
sc = StandardScaler()


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)



#%% 모델링
kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)


models = []
models.append(['Ridge', Ridge()])
models.append(['Lasso', Lasso()])
models.append(['ElasticNet', ElasticNet()])
models.append(['SVR', SVR()])
models.append(['Random Forest', RandomForestRegressor()])
models.append(['XGBoost', XGBRegressor()])
models.append(['LinearRegression', LinearRegression()])
models.append(['CatBoostRegressor', CatBoostRegressor(logging_level=("Silent"))])
models.append(['PLSRegression', PLSRegression()])
models.append(['Lightgbm', LGB.LGBMRegressor()])

list_1 = []

for m in range(len(models)):
    print(models[m])
    model = models[m][1]
    scores = -1 * cross_val_score(model, x_train, y_train, cv=kfold, scoring = "neg_mean_absolute_error")  # 교차검증 MAE
    list_1.append(scores)


df_1 = pd.DataFrame(models)    
df = pd.DataFrame(list_1)
df.index = df_1.iloc[:,0]
df


df['mean'] = df.mean(axis=1) 


#%% CatBoostRegressor 
model = CatBoostRegressor(logging_level=("Silent"),  ## logging_level만 사용했을 때 MAE가 가장 낮게 나왔다.
                          iterations = 10, 
                          learning_rate = 1,
                          depth = 8,
                          eval_metric="MAE")
model.fit(x_train, y_train)


#%% LGBM 하이퍼파라미터 튜닝

gridParams = { 
    'learning_rate': [0.005],
    'n_estimators': [40],
    'num_leaves': [16,32, 64], 
    'random_state' : [501],
    'num_boost_round' : [3000],
    'colsample_bytree' : [0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4], 
    }

lgbm  = LGB.LGBMRegressor(n_estimators=100)

gridcv = GridSearchCV(lgbm, param_grid = gridParams, cv = 3)
gridcv.fit(x_train, y_train, eval_metric = 'mae')


print('Optimized hyperparameters', gridcv.best_params_) # {'colsample_bytree': 0.66, 'learning_rate': 0.005, 'n_estimators': 40, 'num_boost_round': 3000, 'num_leaves': 16, 'random_state': 501, 'reg_alpha': 1.2, 'reg_lambda': 1.4, 'subsample': 0.7}

#%% LGBm 모델설정
model = LGB.LGBMRegressor(colsample_bytree = 0.66,
                           learning_rate = 0.005,
                           n_estimators = 40, 
                           num_boost_round = 3000, 
                           num_leaves = 16,
                           random_state = 501,
                           reg_alpha = 1.2,
                           reg_lambda = 1.4, 
                           subsample = 0.7)

model.fit(x_train, y_train)


#%%
model = CatBoostRegressor(logging_level = ("Silent"))

model.fit(x_train, y_train)
pred = model.predict(x_test)
submission['num'] = pred
submission.to_csv(path + "/sample_submission.csv", index=False)

