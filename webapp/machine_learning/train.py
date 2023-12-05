from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def RF(x_train, y_train, x_test, y_test):
    rdecision_tree = RandomForestClassifier(n_estimators=100)
    rf = rdecision_tree.fit(x_train, y_train)
    y_pred = rf.predict(x_test)


    acc = accuracy_score(y_test, y_pred)
    acc = acc*100
    print(f"Accuracy : ", acc)

    return rf, acc

def knn(x_train, y_train, x_test, y_test):
    knn =KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    acc = acc*100
    print(f"Accuracy : ", acc)

    return knn, acc

def DT(x_train, y_train, x_test, y_test):
    classifier = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
    clf = classifier.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    acc = acc*100
    print(f"Accuracy : ", acc)

    return clf, acc
def adb(x_train, y_train, x_test, y_test):
    classifier = AdaBoostClassifier()
    adb_model = classifier.fit(x_train, y_train)
    y_pred = adb_model.predict(x_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    acc = acc * 100
    print(f"Accuracy: {acc}")

    return adb_model, acc

def NB(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    acc = acc*100
    print(f"Accuracy : ", acc)
    
    return gnb, acc




