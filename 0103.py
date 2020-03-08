
# pip3 install --upgrade pip
# 가상환경 만들기
# conda create -n ML_PATH python=3.7
# 만든 가상환경 활성화
# activate  ML_PATH
# jupyter notebook


...

from sklearn.datasets import fetch_openml
mnist= fetch_openml('MNIST original')
mnist
a= fetch_mldata('mnist_789')

X, y=
