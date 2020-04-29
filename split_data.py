import os
import random
import pickle
path_to_json1 = './FakeHealth/dataset/content/HealthStory/'
json_files1 = []
counter = 0
for i, pos_json in enumerate(sorted(os.listdir(path_to_json1))):
    if json_files1 == []:
        json_files1.append((i,pos_json))
        continue
    x1 = json_files1[-1][0]
    x2 = int(pos_json.split('_')[2].split('.')[0].lstrip('0'))
    counter += x2 - (x1 + 1)
    json_files1.append((i+counter,pos_json))

# json_files1 = [(i,pos_json) for i, pos_json in enumerate(sorted(os.listdir(path_to_json1))) if pos_json.endswith('.json')]
print(json_files1)
lsize = len(json_files1)
trainsize = int(lsize*0.8)
testsize = lsize - trainsize
print("Story Size:",lsize)
random.shuffle(json_files1)

trainset = json_files1[:trainsize]
testset = json_files1[trainsize:]

path_to_json1 = './FakeHealth/dataset/content/HealthRelease/'
json_files1 = []
counter = 0
for i, pos_json in enumerate(sorted(os.listdir(path_to_json1))):
    if json_files1 == []:
        json_files1.append((i,pos_json))
        continue
    x1 = json_files1[-1][0]
    x2 = int(pos_json.split('_')[2].split('.')[0].lstrip('0'))
    counter += x2 - (x1 + 1)
    json_files1.append((i+counter,pos_json))
lsize = len(json_files1)
print("Release Size:",lsize)
trainsize = int(lsize*0.8)
random.shuffle(json_files1)

trainset.extend(json_files1[:trainsize])
testset.extend(json_files1[trainsize:])

fopen1 = open("train.pickle",'wb')
fopen2 = open("test.pickle",'wb')
pickle.dump(trainset,fopen1)
pickle.dump(testset,fopen2)

fopen1.close()
fopen2.close()