
# coding: utf-8

# In[5]:

from io import StringIO
from io import BytesIO

import astropy.io.fits as pf
import numpy as np
import pylab as pl
import pandas as pd

import itertools
import timeit

from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier

from SciServer import CasJobs

NOBJECTS = 100000

GAL_COLORS_DTYPE = [('u', float),
                    ('g', float),
                    ('r', float),
                    ('i', float),
                    ('z', float),
                    ('class', unicode),
                    ('subclass', unicode),
                    ('redshift', float),
                    ('redshift_err', float),
                    ('redshift_warning', int)]

# get data from CasJobs
query_text = ('\n'.join(
    ("SELECT TOP %i" % NOBJECTS,
    "   p.u, p.g, p.r, p.i, p.z, s.class, s.subclass, s.z, s.zerr, s.zwarning",
    "FROM PhotoObj AS p",
    "   JOIN SpecObj AS s ON s.bestobjid = p.objid",
    "WHERE ",
    "   p.u BETWEEN 0 AND 20.0",
    "   AND p.g BETWEEN 0 AND 21.0",
    "   AND s.zwarning = 0",
    "   AND (s.class = 'STAR' OR s.class = 'GALAXY' OR s.class = 'QSO')")))

response = CasJobs.executeQuery(query_text, "DR14")
output = StringIO(response.read().decode('utf-8'))
data = np.loadtxt(output, delimiter=',',skiprows=1, dtype=GAL_COLORS_DTYPE)

#len(data)

#data

df = CasJobs.getPandasDataFrameFromQuery(query_text, "DR14")
 
df

#out = CasJobs.getFitsFileFromQuery("out.fits", query_text, "DR14")
#out
#fp=pf.open('out.fits')
#dp=pd.DataFrame(fp[1].data)



# In[6]:

u = df['u']
g = df['g']
r = df['r']
i = df['i']
z = df['z']
class_in = df['class']
subclass_in = df['subclass']

class_in=df['class']

class_in


# In[7]:

x = np.vstack([np.array(r), np.array(u) - np.array(g), np.array(g) - np.array(r), np.array(r) - np.array(i), np.array(i) - np.array(z)]).T

y = pd.factorize(class_in)[0]

indices = np.arange(x.shape[0])

x_train, x_test, y_train, y_test, i_train, i_test = train_test_split(x, y, indices, test_size=0.20)

y_predict = []

start_time = timeit.default_timer()

k=10
ks = KNeighborsClassifier(n_neighbors=k)#RandomForestClassifier()
ks.fit(x_train, y_train)  
y_ks = ks.predict(x_test)
elapsed = timeit.default_timer() - start_time
proba = ks.predict_proba(x_test)
y_predict.append(y_ks)
acu = acc(y_test, y_ks)


# In[8]:

print('Elapsed time for knn: {} seconds'.format(elapsed))
print(len(y_ks))
print('Accuracy for kNN for k={:} is: {}'.format(k,acu))
print(metrics.classification_report(y_test, y_ks,target_names=['SFGs', 'STARS', 'QSOs'], digits=4))


# In[ ]:




# In[ ]:



