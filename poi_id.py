
# coding: utf-8

# In[1]:


# %load poi_id


# In[1]:


# %load poi_id
#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','deferral_payments','deferred_income','director_fees',
                'exercised_stock_options','expenses', 
                'loan_advances','long_term_incentive','other','restricted_stock',
                'restricted_stock_deferred','shared_receipt_with_poi',
                'poi_fraction_to_messages', 'poi_fraction_from_messages'] 
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Looking at the orginal data source, I see a spreadsheet quirk. There is a data point with key
# "TOTAL". It is unnecessary data and an outlier compared to the rest of the data points. So I 
# remove it :
data_dict.pop("TOTAL", 0)
# I also removed :
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
# Now the rest of the data points are valid and relevant to our quest.

### Task 3: Create new feature(s)
def fraction_messages(poi_messages, all_messages):
    fraction = 0
    if poi_messages != "NaN" and all_messages != "NaN":
        fraction = poi_messages/float(all_messages)
    return fraction

for name in data_dict:
    data_point = data_dict[name]
    data_point["poi_fraction_to_messages"] = fraction_messages(data_point["from_poi_to_this_person"],                                                             data_point["to_messages"])
    data_point["poi_fraction_from_messages"] = fraction_messages(data_point["from_this_person_to_poi"],                                                               data_point["from_messages"])
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test =\
#    train_test_split(features, labels, test_size=0.3, random_state=42)

#from sklearn.model_selection import StratifiedKFold
#skf = StratifiedKFold(n_splits = 4, shuffle = True)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

clf = Pipeline([('preprocessing', MinMaxScaler()),('pca', PCA(n_components=1, whiten=True)),('svm', LinearSVC(class_weight='balanced'))])
pipe = Pipeline([('preprocessing', MinMaxScaler()),('pca', PCA()),('svm', LinearSVC(class_weight = 'balanced'))])
parameters = {'pca__n_components': (1,2,3,4), 'pca__whiten': (True,False), 'svm__C':(1.0,10.0,100.0),'svm__dual':(True,False)}

precision_list = []
recall_list = []
accuracy_list = []
f1_score_list = []
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits = 3, test_size = 0.2, random_state = 42)
grid_search = GridSearchCV(pipe,parameters,scoring='recall',cv=sss)
for train_index, test_index in sss.split(features, labels):
    #features_train, features_test = features[train_index], features[test_index]
    #labels_train, labels_test = labels[train_index], labels[test_index]
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    for i in train_index:
        features_train.append(features[i])
        labels_train.append(labels[i])
    for j in test_index:
        features_test.append(features[j])
        labels_test.append(labels[j])
    #grid_search.fit(features_train, labels_train)
    #pred = grid_search.predict(features_test)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    
    precision_list.append(precision_score(pred,labels_test))
    recall_list.append(recall_score(pred,labels_test))
    accuracy_list.append(accuracy_score(pred,labels_test))
    f1_score_list.append(f1_score(pred,labels_test))
    
precision = (sum(precision_list))/float(len(precision_list))
recall = (sum(recall_list))/float(len(recall_list))
accuracy = (sum(accuracy_list))/float(len(accuracy_list))
f1_score = (sum(f1_score_list))/float(len(f1_score_list))

print "Precision Score :", precision
print "Recall Score :", recall
print "Accuracy Score :", accuracy
print "F1 Score :", f1_score

print clf.named_steps['pca'].explained_variance_ratio_
print clf.named_steps['pca'].components_
#clf.named_steps['pca'].singular_values_
#print grid_search.best_params_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

