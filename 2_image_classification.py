import os
import pickle

import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터 준비
input_dir = "C:/Users/clf-data"
categories = ['empty', 'not_empty']

data = []
labels = []

i = 1
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train/test 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# 모델 훈련
classifier = SVC()

parameters = [
    {
        'gamma': [0.01, 0.001, 0.0001],  # 개별 데이터 포인트의 영향 범위를 조정하여 결정 경계의 복잡성을 제어
        'C': [1, 10, 100, 1000] # 규제 강도를 조절하며, 데이터에 얼마나 잘 맞출지를 제어
    }
]

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# 테스트
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

# 모델 저장
pickle.dump(best_estimator, open('./model.p', 'wb'))