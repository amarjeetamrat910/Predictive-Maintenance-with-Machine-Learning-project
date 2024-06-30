#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")
import scikitplot as skplt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, fbeta_score, matthews_corrcoef, log_loss,
                             confusion_matrix, classification_report, make_scorer,
                             balanced_accuracy_score, accuracy_score, roc_curve,
                             auc, recall_score, roc_auc_score, average_precision_score,
                             precision_score, 
                             precision_recall_curve, multilabel_confusion_matrix)




# In[ ]:





# In[2]:


data=pd.read_csv("predictive_maintenance.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data["Failure Type"]


# In[6]:


data["Failure Type"].value_counts()


# In[7]:


from sklearn.preprocessing import LabelBinarizer
labelbinarizer = LabelBinarizer()
encoded_results_1 = labelbinarizer.fit_transform(data["Failure Type"])
encoded_results_1


# In[ ]:





# In[8]:


df_encoded_1 = pd.DataFrame(encoded_results_1,columns=labelbinarizer.classes_)
df_encoded_1


# In[9]:


data["Type"].value_counts()


# In[10]:


encoded_results_2 = labelbinarizer.fit_transform(data["Type"])
encoded_results_2


# In[11]:


df_encoded_2 = pd.DataFrame(encoded_results_2,columns=labelbinarizer.classes_)
df_encoded_2


# In[12]:


data1 = pd.concat([data, df_encoded_1, df_encoded_2], axis=1)
data1.head()


# In[13]:


data1.columns


# In[14]:


data1 = data1.drop(['Product ID', 'UDI', 'Type', 'Failure Type'], axis=1)
data1.head()


# In[ ]:





# In[15]:


figure = plt.figure(figsize=(12, 10))
sns.heatmap(data1.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# In[16]:


data1.columns


# In[17]:


column_order = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]', 'Tool wear [min]', 'H', 'L', 'M', 'Target', 
                'No Failure', 'Heat Dissipation Failure', 'Overstrain Failure', 
                'Power Failure', 'Tool Wear Failure', 'Random Failures']

data1 = data1.reindex(columns=column_order)


# In[18]:


data1.info()


# In[19]:


data1 = data1.reindex(columns=[
    'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
    'Torque [Nm]', 'Tool wear [min]', 'H', 'L', 'M', 'Target', 
    'No Failure', 'Heat Dissipation Failure', 'Overstrain Failure', 
    'Power Failure', 'Tool Wear Failure', 'Random Failures'
])
data1


# In[20]:


data1.columns=data1.columns.astype("str")
data1


# In[21]:


X = data1.iloc[:, 0:8].values.astype("float")
y = data1.iloc[:, 8:].values.astype("uint8")


# In[22]:


X.shape


# In[23]:


y.shape


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.33)


# In[25]:


scaler = StandardScaler().fit(x_test)
lgbm = LGBMClassifier()


# In[26]:


from sklearn.multioutput import ClassifierChain


# In[27]:


model_lgb = MultiOutputClassifier(estimator=lgbm,n_jobs=None)
chain_lgbm = ClassifierChain(lgbm, order='random', random_state=0)
model_lgb_pred = chain_lgbm.fit(x_train, y_train)
print(model_lgb_pred.score(x_test,y_test)*100)


# In[28]:


yhat_lgb = model_lgb_pred.predict(x_test)
yhat_lgb


# In[29]:


yhat_lgb[:,0]


# In[30]:


y_test[:,0]


# In[31]:


print("F1 Score:", f1_score(y_test[:,0], yhat_lgb[:,0]) * 100)
print("F-beta Score:", fbeta_score(y_test[:,0], yhat_lgb[:,0], beta=2) * 100)
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test[:,0], yhat_lgb[:,0]) * 100)
print("Log Loss:", log_loss(y_test[:,0], yhat_lgb[:,0]) * 100)
print("Confusion Matrix:\n", confusion_matrix(y_test[:,0], yhat_lgb[:,0]) * 100)
print("Classification Report:\n", classification_report(y_test[:,0], yhat_lgb[:,0]))
print("Balanced Accuracy Score:", balanced_accuracy_score(y_test[:,0], yhat_lgb[:,0]) * 100)
print("Accuracy Score:", accuracy_score(y_test[:,0], yhat_lgb[:,0]) * 100)
print("ROC AUC Score:", roc_auc_score(y_test[:,0], yhat_lgb[:,0]) * 100)
print("Average Precision Score:", average_precision_score(y_test[:,0], yhat_lgb[:,0]) * 100)
print("Precision Score:", precision_score(y_test[:,0], yhat_lgb[:,0]) * 100)
print("Multilabel Confusion Matrix:\n", multilabel_confusion_matrix(y_test[:,0], yhat_lgb[:,0]) * 100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[32]:


print("Classification Report for column 0:\n", classification_report(y_test[:,0], yhat_lgb[:,0]))
tn, fp, fn, tp = confusion_matrix(y_test[:,0], yhat_lgb[:,0]).ravel()
skplt.metrics.plot_confusion_matrix(y_test[:,0], yhat_lgb[:,0], normalize=False)
plt.show()

print("Classification Report for column 1:\n", classification_report(y_test[:,1], yhat_lgb[:,1]))
tn, fp, fn, tp = confusion_matrix(y_test[:,1], yhat_lgb[:,1]).ravel()
skplt.metrics.plot_confusion_matrix(y_test[:,1], yhat_lgb[:,1], normalize=False)
plt.show()

print("Classification Report for column 2:\n", classification_report(y_test[:,2], yhat_lgb[:,2]))
tn, fp, fn, tp = confusion_matrix(y_test[:,2], yhat_lgb[:,2]).ravel()
skplt.metrics.plot_confusion_matrix(y_test[:,2], yhat_lgb[:,2], normalize=False)
plt.show()

print("Classification Report for column 3:\n", classification_report(y_test[:,3], yhat_lgb[:,3]))
tn, fp, fn, tp = confusion_matrix(y_test[:,3], yhat_lgb[:,3]).ravel()
skplt.metrics.plot_confusion_matrix(y_test[:,3], yhat_lgb[:,3], normalize=False)
plt.show()

print("Classification Report for column 4:\n", classification_report(y_test[:,4], yhat_lgb[:,4]))
tn, fp, fn, tp = confusion_matrix(y_test[:,4], yhat_lgb[:,4]).ravel()
skplt.metrics.plot_confusion_matrix(y_test[:,4], yhat_lgb[:,4], normalize=False)
plt.show()

print("Classification Report for column 5:\n", classification_report(y_test[:,5], yhat_lgb[:,5]))
tn, fp, fn, tp = confusion_matrix(y_test[:,5], yhat_lgb[:,5]).ravel()
skplt.metrics.plot_confusion_matrix(y_test[:,5], yhat_lgb[:,5], normalize=False)
plt.show()

print("Classification Report for column 6:\n", classification_report(y_test[:,6], yhat_lgb[:,6]))
tn, fp, fn, tp = confusion_matrix(y_test[:,6], yhat_lgb[:,6]).ravel()
skplt.metrics.plot_confusion_matrix(y_test[:,6], yhat_lgb[:,6], normalize=False)
plt.show()


# In[33]:


precision = dict()
recall = dict()
n_classes = 7
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:,i],
                                                        yhat_lgb[:,i])
    plt.plot(recall[i], precision[i], lw=2, label='Failure Type {}'.format(i))
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()


# In[34]:


ranforest = RandomForestClassifier()
moc = MultiOutputClassifier(estimator=ranforest)
moc.fit(x_train,y_train)


# In[35]:


print(round(moc.score(x_test,y_test)*100))


# In[36]:


y_pred_moc=moc.predict(x_test)
y_pred_moc


# In[37]:


for i in range(y_test.shape[1]):
    print("Column %d:" % i)
    print("  Accuracy Score: %.4f" % (accuracy_score(y_test[:, i], y_pred_moc[:, i])*100))
    print("  Average Precision Score: %.4f" % (average_precision_score(y_test[:, i], y_pred_moc[:, i])*100))
    print("  Matthews correlation coefficient: %.4f" % (matthews_corrcoef(y_test[:, i], y_pred_moc[:, i])*100))



# In[38]:


precision = dict()
recall = dict()
n_classes = 6
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred_moc[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='Failure Type {}'.format(i))

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.title("Precision vs. Recall Curve")
plt.show()


# In[39]:


for i in range(y_test.shape[1]):
    print("Classification Report for column %d:\n" % i, classification_report(y_test[:, i], y_pred_moc[:, i]))
    tn, fp, fn, tp = confusion_matrix(y_test[:, i], y_pred_moc[:, i]).ravel()
    skplt.metrics.plot_confusion_matrix(y_test[:, i], y_pred_moc[:, i], normalize=False)
    plt.show()


# In[40]:


catb = CatBoostClassifier()
model_catb = MultiOutputClassifier(estimator=catb)


# In[41]:


model_catb.fit(x_train, y_train)
print(model_catb.score(x_test, y_test)*100)


# In[42]:


y_pred_catb = model_catb.predict(x_test)
y_pred_catb


# In[43]:


for i in range(y_test.shape[1]):
    print("Column %d:" % i)
    print("  Accuracy Score: %.4f" % (accuracy_score(y_test[:, i], y_pred_catb[:, i])*100))
    print("  Average Precision Score: %.4f" % (average_precision_score(y_test[:, i], y_pred_catb[:, i])*100))
    print("  Matthews correlation coefficient: %.4f" % (matthews_corrcoef(y_test[:, i], y_pred_catb[:, i])*100))


# In[44]:


precision = dict()
recall = dict()
n_classes = 6
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred_catb[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='Failure Type {}'.format(i))

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.title("Precision vs. Recall Curve")
plt.show()


# In[45]:


for i in range(y_test.shape[1]):
    print("Classification Report for column %d:\n" % i, classification_report(y_test[:, i], y_pred_catb[:, i]))
    tn, fp, fn, tp = confusion_matrix(y_test[:, i], y_pred_catb[:, i]).ravel()
    skplt.metrics.plot_confusion_matrix(y_test[:, i], y_pred_catb[:, i], normalize=False)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




