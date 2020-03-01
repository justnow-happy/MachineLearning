# 파이썬 2와 파이썬 3 지원
# from __future__ import division, print_function, unicode_literals

import os

# 공통
import numpy as np

# 일관된 출력을 위해 유사난수 초기화
np.random.seed(42)

# 맷플롯립 설정
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
matplotlib.rc('font', family='gulim')
plt.rcParams['axes.unicode_minus'] = False

# 그림을 저장할 폴드
PROJECT_ROOT_DIR = "C:\MachineLearning"
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
# 데이터 추출하여 다운로드
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
# 왜인지 모르겠지만
# 내 작업공간(c:\machinelearning\datasets\housing)에
# 파일이 만들어져있음

# 판다스를 이용한 데이터 읽어 들이는 간단한 함수를 만들었다 >> 실패
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing=load_housing_data()
housing.head()  #앞 5개 열만  보여주기

housing.info() #>>null갯수는 빼고 cnt셈
housing["ocean_proximity"].value_counts()
housing.describe() #freq같은 요약정보, 25분위수 중위수, 갯수..

# 대망의 히스토그램 그려보기
# %matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()
#그림 왜 안나오는지 아는사람?열받는다>>다시해보니 열린다 왜 열리는지 모르겠음

# 데이터를 눈여겨보아 특성을 보기 전에 테스트, 즉 검증세트는 떼어놓고 들여댜보지 않기
# 검증 세트 생성
import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]  #땡땡이(:)와 그냥 변수명의 차이는?
    train_indices=shuffled_indices[test_set_size:]  #얘는 땡땡이가 뒤에 있음
    return data.iloc[train_indices], data.iloc[test_indices]

# 검증세트를 0.2개로 따로 떼어놓겠따
train_set,test_set = split_train_test(housing, 0.2)
# 그에따른 검증세트 obs결과 : 16512 train+ 4128 test
print(len(train_set),"train+", len(test_set), "test")
# np.random   하면 난수생성하는 거기 때문에 실행n번 반복하면 n번 모두 다른 결과 생성
# 초기값 지정해서 하는 함수는....np.random.seed(42)

from zlib import  crc32

def test_set_check(identifier, test_ratio) :
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids=data[id_column]
    in_test_set=ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"]=housing["longitude"]*1000+housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

test_set.info()
housing["median_income"].hist()
#>>>여기까지가 무작위 샘플링 방식 허허허....

####################stratified sampling 계층 샘플링####################
# 소득에 대하여 계층을 부여 후, 샘플링 예정
# 1. 소득에 대해 카테고리 부여
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0, inplace=True)
# >>>5이상은 5로 치환

housing["income_cat"].value_counts()
housing["income_cat"].hist()  #여전히 히스토그램 안생성되구요>>팝업이 아니었나봄
save_fig('income_category_hist')     #더 저장안돼>>저장돼

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# >>얘는뭐야 그럼
    strat_test_set["income_cat"].value_counts() / len(strat_test_set)
    # 전체 주택에서 소득카테고리 비율
    housing["income_cat"].value_counts() / len(housing)

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
# 계층샘플링과 순수 무작위 샘플링 편향 비교
compare_props

#       Overall  Stratified    Random  Rand. %error  Strat. %error
# 1.0  0.039826    0.039729  0.040213      0.973236      -0.243309
# 2.0  0.318847    0.318798  0.324370      1.732260      -0.015195
# 3.0  0.350581    0.350533  0.358527      2.266446      -0.013820
# 4.0  0.176308    0.176357  0.167393     -5.056334       0.027480
# 5.0  0.114438    0.114583  0.109496     -4.318374       0.127011

# 데이터를 원래 상태로 되돌리기
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

################탐색한다
housing = strat_train_set.copy()
# import matplotlib.pyplot as plt
ax = housing.plot(kind="scatter", x="longitude", y="latitude")
ax.set(xlabel='경도', ylabel='위도')
save_fig("bad_visualization_plot")


ax=housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
ax.set(xlabel='경도', ylabel='위도')
save_fig("beeter_visualization_plot")

# 인구표시도 더해줌
ax = housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="인구", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
ax.set(xlabel='경도', ylabel='위도')
plt.legend()
save_fig("housing_prices_scatterplot")
# >>>북부쪽은 주택 가격이 낮다

import matplotlib.image as mpimg
california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california_housing_prices_plot.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="인구",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("위도", fontsize=14)
plt.xlabel("경도", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('중간 주택 가격', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()

########################### 2.4.2 상관관계 조사###########################
# 주택가격 중위수 변수와의 상관계수 구하기
corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
save_fig("scatter_matrix_plot") #그림은실패 font 없어서 그런가

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

# dataset안에 변수를 추가

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"]
# longitude                  -0.047432
# latitude                   -0.142724
# housing_median_age          0.114110
# total_rooms                 0.135097
# total_bedrooms              0.047689
# population                 -0.026920
# households                  0.064506
# median_income               0.687160
# median_house_value          1.000000
# rooms_per_household         0.146285(추가됨)
# bedrooms_per_room          -0.259984(추가됨
# population_per_household   -0.021985(추가됨
# Name: median_house_value, dtype: float64

# 2.5 머신러닝 알고리즘을 위한 데이터 준비
# 2.5.1 데이터 클렌징
# 굳이 정제를 함수로 만드는 이유 :
# 1)나중에 새로운 데이터가 들어와도 적용 가능.
# 2)여러 데이터 변환방법 시도가능

housing=strat_test_set.drop("median_house_value", axis=1)  # 훈련 세트를 위해 레이블 삭제
housing_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
# 누락 데이터는 머시러닝이 다루지 못하는 특성이 있음. 아래 처리방법 1~3
housing.dropna(subset=["total_bedrooms"])
housing.drop("total_bedrooms", axis=1)       # 옵션 2
median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True) # 옵션 3

# 이건 미리 만들어져 있는 함수..진즉 알려주지
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# 근데 함정이 텍스트는 적용 불가 > ocean머시기 변수 제외하고 복사본 하나 만들어
housing_num= housing.drop("ocean_proximity", axis=1)
# 대체처리 적용!!
imputer.fit(housing_num)

imputer.statistics_
# Out[163]:
# array([-118.455     ,   34.22      ,   28.        , 2158.5       ,
#         441.        , 1172.        ,  416.        ,    3.51475   ,
#           2.82308399])
housing_num.median().values #이건 수동으로 median 계산한것
# array([-118.455     ,   34.22      ,   28.        , 2158.5       ,
#         441.        , 1172.        ,  416.        ,    3.51475   ,
#           2.82308399])

X=imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))

housing_tr.loc[sample_incomplete_rows.index.values]

# 2.5.2 텍스트와  범주형 특성
housing_cat = housing["ocean_proximity"]
housing_cat.head(10)
 # 머신러닝은 숫자를 다루어서 위에 카테고리 범주형 텍스트를 숫자로 변환한다
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]   #array([0, 0, 0, 1, 1, 1, 1, 2, 1, 1], dtype=int64)앞의 열놈 show

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot

# ????????????????????????????????????????희소행렬임? 2차원 넘파이 배열로 바꾸어줘야함
# 넘파이 배열로 바꾸기위해서 toarry
housing_cat_1hot.toarray()

# array([[1., 0., 0., 0., 0.],
#        [1., 0., 0., 0., 0.],
#        [1., 0., 0., 0., 0.],
#        ...,
#        [1., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0.],
#        [0., 1., 0., 0., 0.]])
# github에서 가져오기 시작
# [PR #9151](https://github.com/scikit-learn/scikit-learn/pull/9151)에서 가져온 CategoricalEncoder 클래스의 정의.
# 이 클래스는 사이킷런 0.20에 포함될 예정입니다.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse


class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):

        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
    # git복사 끝

    # from sklearn.preprocessing import CategoricalEncoder # Scikit-Learn 0.20에서 추가 예정

    cat_encoder = CategoricalEncoder(encoding="onehot-dense")
    housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
    housing_cat_1hot

cat_encoder.categories_

housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)