import csv
from sklearn.preprocessing import Normalizer

def proc():
  headings = []
  users = []
  with open('users.csv', 'rb') as f:
    s = csv.reader(f, delimiter=',')
    for row in s:
      users.append(row)
  headings = users[0]
  users = users[1:]
  features = []

  for i in users:
    temp = []
    temp.append(i[3])
    temp.append(i[4])
    temp.append(i[5])
    temp.append(i[6])
    temp.append(i[7])
    features.append(temp)

  transformed_features = Normalizer().fit_transform(features)
  return transformed_features
