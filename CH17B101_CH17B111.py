import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from math import cos, asin, sqrt, pi
from sklearn.model_selection import train_test_split

train = pd.read_csv(r"../../Data/train.csv")
test = pd.read_csv(r"../../Data/test.csv")
bikers=pd.read_csv(r"../../Data/bikers.csv")
tours=pd.read_csv(r"../../Data/tours.csv")
bikers_network = pd.read_csv(r"../../Data/bikers_network.csv")
tour_convoy = pd.read_csv(r"../../Data/tour_convoy.csv")
locations = pd.read_csv(r"../../Data/locations.csv")
test_df = pd.read_csv(r"../../Data/test.csv")
locations = locations.rename({'latitude':'biker_latitude','longitude':'biker_longitude'},axis=1)
tours = tours.rename({'biker_id':'biker_org_id'},axis=1)

impute = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df = pd.DataFrame(bikers.area)
impute = impute.fit(df)
df = impute.transform(df)
bikers['area'] = df

impute = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df = pd.DataFrame(bikers.gender)
impute = impute.fit(df)
df = impute.transform(df)
bikers['gender'] = df

impute = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df = pd.DataFrame(bikers.time_zone)
impute = impute.fit(df)
df = impute.transform(df)
bikers['time_zone'] = df

impute = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df = pd.DataFrame(bikers.member_since)
impute = impute.fit(df)
df = impute.transform(df)
bikers['member_since'] = df
bikers.bornIn = bikers.bornIn.replace('None',1988)

tours['city'].fillna(value=str(tours['city'].mode()), inplace=True)
tours['state'].fillna(value=str(tours['state'].mode()), inplace=True)
tours['country'].fillna(value=str(tours['country'].mode()), inplace=True)
tours['latitude'].fillna(value=int(tours['latitude'].mean()),inplace=True)
tours['longitude'].fillna(value=int(tours['longitude'].mean()),inplace=True)
tours['pincode'].fillna(value=str(tours['pincode'].mode()),inplace=True)

locations['biker_latitude'].fillna(value=locations['biker_latitude'].mean(),inplace=True)
locations['biker_longitude'].fillna(value=locations['biker_longitude'].mean(),inplace=True)

tour_convoy = tour_convoy.replace(np.nan,'X1 X2')
popularity_tour = []
for i in range(tour_convoy.shape[0]):
    if tour_convoy.going.iloc[i] != 'X1 X2' and tour_convoy.maybe.iloc[i] != 'X1 X2' and tour_convoy.not_going.iloc[i]!= 'X1 X2':
        popularity_tour.append(len(tour_convoy.going.iloc[i].split(' '))/(len(tour_convoy.going.iloc[i].split(' '))+len(tour_convoy.maybe.iloc[i].split(' '))+len(tour_convoy.not_going.iloc[i].split(' '))))
    else:
        popularity_tour.append(np.nan)

train=pd.merge(train,bikers,how="left",on='biker_id')
train=pd.merge(train,tours,how='left',on='tour_id')
train['timestamp'] = pd.to_datetime(train.timestamp)
train['tour_date'] = pd.to_datetime(train.tour_date)
train['year_tour'] = train.tour_date.dt.year
train['month_tour'] = train.tour_date.dt.month
train['day_tour'] = train.tour_date.dt.day

train.bornIn = train.bornIn.astype(int)
m = list()
for i in range(train.shape[0]):
    m.append(train.year_tour.iloc[i]-train.bornIn.iloc[i])
    
train['biker_age']=m

train = pd.merge(train,bikers_network,how='left',on='biker_id')
train = pd.merge(train,tour_convoy,how='left',on='tour_id')

num_friends_going = []
num_friends_maybe = []
num_friends_not_going = []
for i in range(train.shape[0]):
    a1 = set(train.going.iloc[i].split(' '))
    b1 = set(train.friends.iloc[i].split(' '))
    c1 = set(train.not_going.iloc[i].split(' '))
    d1 = set(train.maybe.iloc[i].split(' '))
    num_friends_going.append(len(b1&a1))
    num_friends_maybe.append(len(b1&d1))
    num_friends_not_going.append(len(b1&c1))
train['num_friends_going'] = num_friends_going
train['num_friends_maybe'] = num_friends_maybe
train['num_friends_not_going'] = num_friends_not_going


train = train.rename({'invited_x':'invited'},axis=1)

train['timestamp'] = train.timestamp.astype(str)
train.tour_date = train.tour_date.astype(str)

train = pd.merge(train,locations,how='left',on='biker_id')

def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a))

tour_distance = []
for i in range(train.shape[0]):
    lat1 = train.latitude.iloc[i]
    lat2 = train.biker_latitude.iloc[i]
    lon1 = train.longitude.iloc[i]
    lon2 = train.biker_longitude.iloc[i]
    tour_distance.append(distance(lat1,lon1,lat2,lon2))

train['tour_distance']=tour_distance

train = train.drop(columns=['area','biker_org_id','city','state','country','pincode','friends','going','maybe','not_going','invited_y'])

X = train.drop(columns=['biker_id','tour_id','like','dislike'])
categorical = ['language_id','location_id','gender','member_since','timestamp','tour_date','bornIn']

Y = train.like

X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.33,random_state=42)

model = CatBoostClassifier(task_type='CPU',iterations=1500,eval_metric='AUC',use_best_model=True)
model_2 = CatBoostClassifier(task_type='CPU',iterations=1500,eval_metric='F1',use_best_model=True)
model.fit(X_train,Y_train,cat_features=categorical,eval_set=(X_val,Y_val))
model_2.fit(X_train,Y_train,cat_features=categorical,eval_set=(X_val,Y_val))
test=pd.merge(test,bikers,how="left",on='biker_id')
test=pd.merge(test,tours,how='left',on='tour_id')
test['timestamp'] = pd.to_datetime(test.timestamp)
test['tour_date'] = pd.to_datetime(test.tour_date)
test['year_tour'] = test.tour_date.dt.year
test['month_tour'] = test.tour_date.dt.month
test['day_tour'] = test.tour_date.dt.day
test.bornIn = test.bornIn.astype(int)
m = list()
for i in range(test.shape[0]):
    m.append(test.year_tour.iloc[i]-test.bornIn.iloc[i])
test['biker_age']=m
test = pd.merge(test,bikers_network,how='left',on='biker_id')
test = pd.merge(test,tour_convoy,how='left',on='tour_id')
num_friends_going = []
num_friends_maybe = []
num_friends_not_going = []
for i in range(test.shape[0]):
    a1 = set(test.going.iloc[i].split(' '))
    b1 = set(test.friends.iloc[i].split(' '))
    c1 = set(test.not_going.iloc[i].split(' '))
    d1 = set(test.maybe.iloc[i].split(' '))
    num_friends_going.append(len(b1&a1))
    num_friends_maybe.append(len(b1&d1))
    num_friends_not_going.append(len(b1&c1))
test['num_friends_going'] = num_friends_going
test['num_friends_maybe'] = num_friends_maybe
test['num_friends_not_going'] = num_friends_not_going
test['timestamp'] = test.timestamp.astype(str)
test.tour_date = test.tour_date.astype(str)
test = pd.merge(test,locations,how='left',on='biker_id')
tour_distance = []
for i in range(test.shape[0]):
    lat1 = test.latitude.iloc[i]
    lat2 = test.biker_latitude.iloc[i]
    lon1 = test.longitude.iloc[i]
    lon2 = test.biker_longitude.iloc[i]
    tour_distance.append(distance(lat1,lon1,lat2,lon2))
test['tour_distance'] = tour_distance

test = test.drop(columns=['area','biker_org_id','city','state','pincode','country','friends','going','maybe','not_going','invited_y'])
test = test.rename({'invited_x':'invited'},axis=1)
test = test.drop(columns=['biker_id','tour_id'])
predict_prob = model.predict_proba(test)
predict_prob_2 = model_2.predict_proba(test)

#1st csv
test_df['prediction'] = predict_prob[:,1]
biker_set = test_df.drop_duplicates(subset="biker_id")
biker_set=biker_set['biker_id']
bikers_set=np.array(biker_set)
bikers = []
tours = []
for biker in bikers_set:
    idx = np.where(biker==test_df["biker_id"]) 
    tour = list(test_df["tour_id"].loc[idx])
    preds = list(test_df['prediction'].loc[idx])
    tour = [x for _,x in sorted(zip(preds,tour),reverse=True)]
    tour = " ".join(tour) # list to space delimited string
    bikers.append(biker)
    tours.append(tour)
submission =pd.DataFrame(columns=["biker_id","tour_id"])
submission["biker_id"] = bikers
submission["tour_id"] = tours
submission.to_csv('CH17B101_CH17B111_1.csv',index=False)

#2nd csv file
test_df['prediction'] = predict_prob_2[:,1]
biker_set = test_df.drop_duplicates(subset="biker_id")
biker_set=biker_set['biker_id']
bikers_set=np.array(biker_set)
bikers = []
tours = []
for biker in bikers_set:
    idx = np.where(biker==test_df["biker_id"]) 
    tour = list(test_df["tour_id"].loc[idx])
    preds = list(test_df['prediction'].loc[idx])
    tour = [x for _,x in sorted(zip(preds,tour),reverse=True)]
    tour = " ".join(tour) # list to space delimited string
    bikers.append(biker)
    tours.append(tour)
submission =pd.DataFrame(columns=["biker_id","tour_id"])
submission["biker_id"] = bikers
submission["tour_id"] = tours
submission.to_csv('CH17B101_CH17B111_2.csv',index=False)
