# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:49:35 2021

@author: student
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("C:/dacon/parking_data/train.csv")
test = pd.read_csv("C:/dacon/parking_data/test.csv")
age_gender = pd.read_csv("C:/dacon/parking_data/age_gender_info.csv")
submission = pd.read_csv("C:/dacon/parking_data/sample_submission.csv")


## 차트 한글 깨짐 방지
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family = font)

import warnings
warnings.filterwarnings("ignore")
#%%

# 데이터 모양 확인 (2952, 15), (1022, 14), (16, 23)
train.shape, test.shape, age_gender.shape

# train셋과 test셋에 큰 차이가 없는 것을 확인
train.describe().T
test.describe().T


#%% 지역별 라인 차트

ag = age_gender.set_index('지역') # age_gender의 인덱스를 지역 컬럼으로 대체


ag.loc['전체 평균'] = ag.mean()
ag.loc['광역시 평균'] = ag.loc[(ag.index.str.contains('시')) & (~ag.index.str.contains('세종'))].mean()
ag.loc['도 평균'] = ag.loc[ag.index.str.contains('도')].mean()

flt = plt.figure(figsize = (20, 6))
sns.lineplot(data=ag.T,)
plt.title("지역별 세대(양성) 라인 차트")
plt.xticks(rotation=45)
plt.ylim(top=0.13)
plt.show()


fig = plt.figure(figsize = (18,6))
sns.lineplot(data = ag.loc[:,ag.columns.str.contains('여자')].T)
plt.title('지역별 세대(여자) 라인 차트')
plt.ylim(top = 0.13)
plt.show()

fig = plt.figure(figsize = (18,6))
sns.lineplot(data = ag.loc[:,ag.columns.str.contains('남자')].T)
plt.title('지역별 세대(남자) 라인 차트')
plt.ylim(top = 0.13)
plt.show()


plt.figure(figsize = (14,10))
sns.heatmap((ag*100).round(3),
            annot = True, linewidths=0.01)


fig = plt.figure(figsize = (22,18))
plt.xticks(rotation=90)
for i, n in enumerate(list(ag.index)):
    plt.subplot(5,5, i+1)
    plt.subplots_adjust(hspace = 1.0)
    plt.title("{} 인구비중".format(n))
    sns.barplot(data=ag.loc[ag.index==n])
    plt.xticks(rotation=90)
    
    
#%% 컬럼별 밸류 체크
def check_train_test_column_values(train, test, column):
    #함수 정의 : 두 데이터 테이블과 특정 컬럼을 집어 넣으면 value를 비교하여 출력하는 함수
    print("{} Column에 대한 train_test_values_check 입니다 =================================".format(column))
    # Train/Test set의 입력 칼럼의 value를 set으로 받아줍니다.
    train_colset = set(train[column])
    test_colset = set(test[column])
    
    # Train/Test-set 고유한 value의 개수를 구함.
    print("Train-set에 있는 고유한 value 개수 : {}".format(len(train_colset)))
    print("Test-set에 있는 고유한 value 개수 : {}".format(len(test_colset)))
    
    # Train/Test-set 모두에 포함되어있는 value를 구함. intersection은 교집합
    print("="*80)
    common_colset = train_colset.intersection(test_colset)
    print("Train/Test-set에 공통으로 포함되어 있는 value 개수 : {}".format(len(common_colset)))
    if len(common_colset) > 100: # value가 너무 많으면 보기 힘드니 pass
        pass
    else:
        try: # int나 float은 sorted가 되지 않기 때문에 try except로 넣어줌
            print("Train/Test-set에 공통으로 포함되어 있는 value : {}".format(sorted(common_colset)))
        except:
            print("Train/Test-set에 공통으로 포함되어 있는 value : {}".format(common_colset))

    # Train-set에만 있는 value를 구함.
    print('='*80)
    train_only_colset = train_colset.difference(test_colset)
    print("Train-set에만 있는 value는 총 {}개 입니다.".format(len(train_only_colset)))
    if len(train_only_colset) > 100:
        pass
    else:
        try:
            print("Train-set에만 있는 value는 : {}".format(sorted(train_only_colset)))
        except:
            print("Train-set에만 있는 value는 : {}".format(train_only_colset))
    # Test-set에만 있는 value를 구함.
    print('='* 80)
    test_only_colset = test_colset.difference(train_colset)
    print("Test-set에만 있는 value는 총 {} 개 입니다.".format(len(test_only_colset)))
    if len(test_only_colset) > 100:
        pass
    else:
        try:
            print("Test-set에만 있는 value는 : {}".format(sorted(test_only_colset)))
        except:
            print("Test-set에만 있는 value는 : {}".format((test_only_colset)))
    print(" ")

obj_cols = []
for col in train.columns:
    if train[col].dtypes == "object":
        obj_cols.append(col)
        
for col in obj_cols:
    check_train_test_column_values(train, test, col)
    
    
#%%

# 임대건물 구분 아파트
train_apart = train[['단지코드', '임대건물구분', '공급유형', '전용면적', '전용면적별세대수', '자격유형', '임대보증금', '임대료']].loc[train['임대건물구분'] == '아파트']

# 임대건물 구분 상가
train_shop = train[['단지코드', '임대건물구분', '공급유형', '전용면적', '전용면적별세대수', '자격유형']].loc[train['임대건물구분'] == '상가']

# 임대건물 구분 단지
train_danji = train[['단지코드', '총세대수', '지역', '공가수', '도보 10분거리 내 지하철역 수(환승노선 수 반영)', '도보 10분거리 내 버스정류장 수', '단지내주차면수', '등록차량수']].drop_duplicates()

train_apart.shape, train_shop.shape, train_danji.shape

train_apart.head(3)
train_shop.head(3)
train_danji.head(3)

# test
test_apart = test[['단지코드', '임대건물구분', '공급유형', '전용면적', '전용면적별세대수', '자격유형', '임대보증금', '임대료']].loc[test['임대건물구분'] == '아파트']

test_shop = test[['단지코드', '임대건물구분', '공급유형', '전용면적', '전용면적별세대수', '자격유형']].loc[test['임대건물구분'] == '상가']

test_danji = test[['단지코드', '총세대수', '지역', '공가수', '도보 10분거리 내 지하철역 수(환승노선 수 반영)', '도보 10분거리 내 버스정류장 수', '단지내주차면수']].drop_duplicates()

test_danji = test[['단지코드', '총세대수', '지역', '공가수', '도보 10분거리 내 지하철역 수(환승노선 수 반영)', '도보 10분거리 내 버스정류장 수', '단지내주차면수']].drop_duplicates()


test_apart.shape, test_shop.shape, test_danji.shape

test_apart.head(3)
test_shop.head(3)
test_danji.head(3)


#%% 결측치
plt.figure(figsize=(16, 6))
sns.heatmap(train.isnull().T, cbar=False)
plt.show()

train.isnull().sum().to_frame()

plt.figure(figsize=(16, 6))
sns.heatmap(test.isnull().T, cbar=False)
plt.show()

test.isnull().sum().to_frame()

# 지하철 결측치
train.loc[train['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].isnull()].sample(5)

print('전체 단지 수: ', train['단지코드'].nunique())
print('지하철 결측치 단지 수: ', train.loc[train['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].isnull()]['단지코드'].nunique())
print('지하철 결측치 단지: ', train.loc[train['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].isnull()]['단지코드'].unique())
print('지하철 결측치 단지 지역: ', train.loc[train['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].isnull()]['지역'].unique())


# test
test.loc[test['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].isnull()].sample(5)
print('전체 단지 수: ', test['단지코드'].nunique())
print('지하철 결측치 단지 수: ', test.loc[test['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].isnull()]['단지코드'].nunique())
print('지하철 결측치 단지: ', test.loc[test['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].isnull()]['단지코드'].unique())
print('지하철 결측치 단지 지역: ', test.loc[test['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].isnull()]['지역'].unique())


# 버스 결측치
train.loc[train['도보 10분거리 내 버스정류장 수'].isnull()]

train.loc[train['도보 10분거리 내 버스정류장 수'].isnull()]['지역'].unique()

test.loc[test['도보 10분거리 내 버스정류장 수'].isnull()]

# 임대료, 임대보증금 결측치
train.loc[(train['임대건물구분'] != '상가') & (train['임대보증금'].isnull())]

test.loc[(test['임대건물구분'] != '상가') & (test['임대보증금'].isnull())]

# 상가유형
len(train.loc[train['임대건물구분'] == '상가'])

train.loc[train['임대건물구분'] == '상가'].isnull().sum().to_frame()

print('전체 단지 수:', train['단지코드'].nunique())
print('상가 보유 단지 수:', train.loc[train['임대건물구분'] == '상가']['단지코드'].nunique())
print('상가 보유 단지:', train.loc[train['임대건물구분'] == '상가']['단지코드'].unique())
print('상가 보유 단지 지역:', train.loc[train['임대건물구분'] == '상가']['지역'].unique())

# test
len(test.loc[test['임대건물구분'] == '상가'])

test.loc[test['임대건물구분'] == '상가'].isnull().sum().to_frame()

print('전체 단지 수:', test['단지코드'].nunique())
print('상가 보유 단지 수:', test.loc[test['임대건물구분'] == '상가']['단지코드'].nunique())
print('상가 보유 단지:', test.loc[test['임대건물구분'] == '상가']['단지코드'].unique())
print('상가 보유 단지 지역:', test.loc[test['임대건물구분'] == '상가']['지역'].unique())


# 자격유형 결측치
test.loc[test['자격유형'].isnull()]

#%% 변수별 분포확인
numeric_cols = []

for col in train.columns:
    if train[col].dtypes != 'object':
        numeric_cols.append(col)

fig = plt.figure(figsize = (22,22))
for i, n in enumerate(numeric_cols):
    plt.subplot(5, 2, i+1)
    plt.subplots_adjust(hspace = 0.3)
    sns.histplot(train[n])
    

train.loc[train[['단지코드']].drop_duplicates().index][['단지코드','총세대수']].nlargest(3, '총세대수')
train.loc[train[['단지코드']].drop_duplicates().index][['단지코드','총세대수']].nsmallest(3, '총세대수')
train.loc[train[['단지코드']].drop_duplicates().index][['단지코드','등록차량수']].nlargest(3, '등록차량수')
train.loc[train[['단지코드']].drop_duplicates().index][['단지코드','등록차량수']].nsmallest(3, '등록차량수')


# 총 세대수
train.loc[train['단지코드'] == 'C1004']['전용면적별세대수'].sum()

## 총 세대수와 전용면적별 세대수 합이 맞지 않는 단지가 40개가 있다.
(train.groupby(['단지코드'])['전용면적별세대수'].sum() != train.groupby(['단지코드'])['총세대수'].max()).sum()

train_danji.set_index('단지코드').loc[(train.groupby(['단지코드'])['전용면적별세대수'].sum() != train.groupby(['단지코드'])['총세대수'].max())]

unmatch_danji_list = list(train_danji.set_index('단지코드').loc[(train.groupby(['단지코드'])['전용면적별세대수'].sum() != 
 train.groupby(['단지코드'])['총세대수'].max())].index)


train.loc[train['단지코드'].isin(unmatch_danji_list)].head()

# 지역
train_danji['지역'].value_counts()
test_danji['지역'].value_counts()

## 지역별 단지 개수 train
sns.barplot(data=train.groupby(['지역']).nunique().sort_values(by=['단지코드'], ascending=False),
            x=train.groupby(['지역']).nunique().sort_values(by=['단지코드'], ascending=False).index, y='단지코드')
plt.xticks(rotation=90)
plt.title('지역별 단지 개수')
plt.show()

## 지역별 단지 개수 test
sns.barplot(data=test.groupby(['지역']).nunique().sort_values(by=['단지코드'], ascending=False),
            x=test.groupby(['지역']).nunique().sort_values(by=['단지코드'], ascending=False).index, y='단지코드')
plt.xticks(rotation=90)
plt.title('지역별 단지 개수')
plt.show()

## 지역별 단지별 총세대수 train
sns.barplot(data=train_danji.groupby(['지역']).sum().sort_values(by=['총세대수'], ascending=False),
            x=train_danji.groupby(['지역']).sum().sort_values(by=['총세대수'], ascending=False).index, y='총세대수')
plt.xticks(rotation=90)
plt.title('지역별 단지별 총세대수 합계')
plt.show()

## 지역별 단지별 총세대수 test
sns.barplot(data=test_danji.groupby(['지역']).sum().sort_values(by=['총세대수'], ascending=False),
            x=test_danji.groupby(['지역']).sum().sort_values(by=['총세대수'], ascending=False).index, y='총세대수')
plt.xticks(rotation=90)
plt.title('지역별 단지별 총세대수 합계')
plt.show()


# 공급 유형
test[['단지코드','공급유형']].drop_duplicates()['공급유형'].value_counts()
train.loc[train['공급유형'] == '공공분양']
test.loc[test['공급유형'] == '공공분양']


train.loc[train['공급유형'] == '장기전세']
train.groupby(['공급유형']).sum()


## train 공급유형
sns.barplot(data=train.groupby(['공급유형']).sum().sort_values(by=['전용면적별세대수'], ascending=False),
            x=train.groupby(['공급유형']).sum().sort_values(by=['전용면적별세대수'], ascending=False).index, y='전용면적별세대수')
plt.xticks(rotation=90)
plt.title('공급유형별 세대수 합계')
plt.show()


## test 공급유형
sns.barplot(data=test.groupby(['공급유형']).sum().sort_values(by=['전용면적별세대수'], ascending=False),
            x=test.groupby(['공급유형']).sum().sort_values(by=['전용면적별세대수'], ascending=False).index, y='전용면적별세대수')
plt.xticks(rotation=90)
plt.title('공급유형별 세대수 합계')
plt.show()

# 전용면적
train.loc[train['단지코드'] == 'C2612'][['단지코드','전용면적','전용면적별세대수','자격유형','임대보증금','임대료']]

# 자격유형
train['자격유형'].unique() #15가지의 자격유형이 있다.

train.groupby(['단지코드','전용면적','전용면적별세대수'])['자격유형'].nunique().value_counts()

## train 자격유형별 총 세대수
sns.barplot(data=train.groupby(['자격유형']).sum().sort_values(by=['전용면적별세대수'], ascending=False),
            x=train.groupby(['자격유형']).sum().sort_values(by=['전용면적별세대수'], ascending=False).index, y='전용면적별세대수')
plt.title('자격유형별 총세대수 총합')
plt.xticks(rotation=90)
plt.show()

## test 자격유형별 총 세대수
sns.barplot(data=test.groupby(['자격유형']).sum().sort_values(by=['전용면적별세대수'], ascending=False),
            x=test.groupby(['자격유형']).sum().sort_values(by=['전용면적별세대수'], ascending=False).index, y='전용면적별세대수')
plt.title('자격유형별 총세대수 총합')
plt.xticks(rotation=90)
plt.show()

train.loc[train['임대건물구분'] == '상가']['자격유형'].value_counts()

# 지하철
train_danji['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].value_counts()
train_danji.groupby(['지역'])['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].mean().plot(kind='bar')

test_danji.groupby(['지역'])['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].mean().plot(kind='bar')

# 버스
train_danji['도보 10분거리 내 버스정류장 수'].value_counts()
train_danji.groupby(['지역'])['도보 10분거리 내 버스정류장 수'].mean().plot(kind='bar')

test_danji['도보 10분거리 내 버스정류장 수'].value_counts()
test_danji.groupby(['지역'])['도보 10분거리 내 버스정류장 수'].mean().plot(kind='bar')

# 단지내 주차면수
(train_danji['단지내주차면수'] / train_danji['총세대수']).plot(kind='hist', bins=50)
(test_danji['단지내주차면수'] / test_danji['총세대수']).plot(kind='hist', bins=50)

# 등록차량수
(train_danji['등록차량수'] / train_danji['총세대수']).plot(kind='hist', bins=50)

