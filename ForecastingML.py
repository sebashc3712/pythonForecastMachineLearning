
'''
This code finds the best algorithm to forecast whatever numeric data you want.

The algorithms used in the model are linear regression, neural network, support vector regressor, decision tree
and random forest

This model returns the best algorithm into a file with .pkl extension wich you can use then with the joblib library from
sklearn
'''



# Here all the used libraries are imported
from sklearn.linear_model import LinearRegression # linear regression algorithm
from sklearn.model_selection import train_test_split # datasplit algorithm to train your data
from sklearn.preprocessing import StandardScaler, Normalizer # Normalizer
from sklearn.decomposition import PCA # principal component analysis algorithm
from sklearn import tree # decision tree algorithm
from sklearn.ensemble import RandomForestRegressor # random forest algorithm
from sklearn.svm import SVR # support vector algorithm
from sklearn.neural_network import MLPRegressor # neural network algorithm
from sklearn import manifold # isomap algorithm



#With this function we find the best neural network for the given dataset
def bestMLP(X_train,X_test,y_train,y_test,n_components,MaxLayers):
    number_Layers=1
    best_score=0
    best_PCA=0
    best_Layers=0
    best_Neurons=0
    best_model=[] # Empty list for the best model
    # Loop for number of layers
    for number_Layers in range(1,MaxLayers+1):
        #Loop for neurons for each layer
        print("Layers: ",number_Layers)
        number_Neurons=n_components
        for number_Neurons in range(1,number_Layers+1):
            print("Neurons: ",number_Neurons)
            model2=MLPRegressor(hidden_layer_sizes=(number_Layers,number_Neurons ))
            model2.fit(X_train,y_train)
            score=model2.score(X_test,y_test)
            if score > best_score:
                best_score=score
                best_PCA=n_components
                best_Layers=number_Layers
                best_Neurons=number_Neurons
                best_model=model2
    return best_score,best_PCA,best_Layers,best_Neurons,best_model
            


#With this function we find the best random forest for the given dataset
def bestRF(X_train,X_test,y_train,y_test,n_components,MaxDepth):
    max_depth=1
    best_score=0
    best_depth=0
    best_randomState=0
    best_PCA=0
    best_model=[] # Empty list for the best model
    # Loop for the deep of the trees of the random forest
    for max_depth in range(1,MaxDepth+1):
        random_state=1
        # Loop for values of the random state
        for random_state in range(1,11):
            model4=RandomForestRegressor(max_depth = max_depth, random_state = random_state)
            model4.fit(X_train,y_train)
            score=model4.score(X_test,y_test)
            if score > best_score:
                best_score=score
                best_depth=max_depth
                best_randomState=random_state
                best_PCA=n_components
                best_model=model4
    return best_score,best_depth,best_randomState,best_PCA,best_model
    


#With this function we find the best decision tree for the given dataset
def bestDT(X_train,X_test,y_train,y_test,n_components,MaxDepth):
    max_depth=1
    best_score=0
    best_depth=0
    best_randomState=0
    best_PCA=0
    best_model=[] # Empty list for the bes model
    # Loop for the deep of the trees of the decision tree
    for max_depth in range(1,MaxDepth+1):
        # Loop for values of the random state
        random_state=1
        for random_state in range(1,11):
            model5=tree.DecisionTreeRegressor(max_depth = max_depth, random_state = random_state)
            model5.fit(X_train,y_train)
            score=model5.score(X_test,y_test)
            if score > best_score:
                best_score=score
                best_depth=max_depth
                best_randomState=random_state
                best_PCA=n_components
                best_model=model5
    return best_score,best_depth,best_randomState,best_PCA,best_model



#This function uses the three functions from above to find the best model for each PCA result
def forecastingModel(Dataset,Label,MLP=False,MaxLayers=100,MaxDepth=10):
    best_model_all=[]
    best_score_all=0
    intial_components=len(Dataset.columns)
    #Here you set the PCA and try different values for n_components
    n_components=1
    best_MLP=0
    best_PCA_MLP=0
    best_Layers_MLP=0
    best_Neurons_MLP=0
    best_SVR=0
    best_PCA_SVR=0
    best_RandomForest=0
    max_depth_RF=0
    random_state_RF=0
    best_PCA_RF=0
    best_DecisionTree=0
    max_depth_DT=0
    random_state_DT=0
    best_PCA_DT=0
    best_LinearRegression=0
    best_PCA_LinearRegression=0
    # Without PCA
    #Here you split the data
    X_train, X_test, y_train, y_test = train_test_split(Dataset, Label, test_size=0.33)
    #MLP
    if MLP==True:
        best_score,best_PCA,best_Layers,best_Neurons,best_model_MLP=bestMLP(X_train,X_test,y_train,y_test,intial_components,MaxLayers)
        if best_score>best_MLP:
            best_MLP=best_score
            best_PCA_MLP=best_PCA
            best_Layers_MLP=best_Layers
            best_Neurons_MLP=best_Neurons
        if best_score>best_score_all:
            best_model_all=best_model_MLP
            best_score_all=best_score
    #SVR
    model3=SVR()
    model3.fit(X_train,y_train)
    score=model3.score(X_test,y_test)
    if score > best_SVR:
        best_SVR=score
        best_PCA_SVR=intial_components
    if score > best_score_all:
        best_model_all=model3
        best_score_all=score
    #RandomForestRegressor
    best_score,best_depth,best_randomState,best_PCA, best_model_RF=bestRF(X_train,X_test,y_train,y_test,intial_components,MaxDepth)
    if best_score>best_RandomForest:
        best_RandomForest=best_score
        best_PCA_RF=best_PCA
        max_depth_RF=best_depth
        random_state_RF=best_randomState
    if best_score>best_score_all:
        best_model_all=best_model_RF
        best_score_all=best_score
    #DecisionTree
    best_score,best_depth,best_randomState,best_PCA, best_model_DT=bestDT(X_train,X_test,y_train,y_test,intial_components,MaxDepth)
    if best_score>best_DecisionTree:
        best_DecisionTree=best_score
        best_PCA_DT=best_PCA
        max_depth_DT=best_depth
        random_state_DT=best_randomState
    if best_score>best_score_all:
        best_model_all=best_model_DT
        best_score_all=best_score
    #LinearRregression
    model6=LinearRegression()
    model6.fit(X_train,y_train)
    score=model6.score(X_test,y_test)
    if score > best_LinearRegression:
        best_LinearRegression=score
        best_PCA_LinearRegression=intial_components
    if score > best_score_all:
        best_model_all=model6
        best_score_all=score
    # With PCA
    for n_components in range(1,len(Dataset.columns)):
        model=PCA(n_components=n_components)
        model.fit(Dataset)
        T=model.transform(Dataset)
        #Here you split the data
        X_train, X_test, y_train, y_test = train_test_split(T, Label, test_size=0.33)
        #MLP
        if MLP==True:
            best_score,best_PCA,best_Layers,best_Neurons,best_model_MLP=bestMLP(X_train,X_test,y_train,y_test,n_components,MaxLayers)
            if best_score>best_MLP:
                best_MLP=best_score
                best_PCA_MLP=best_PCA
                best_Layers_MLP=best_Layers
                best_Neurons_MLP=best_Neurons
            if best_score>best_score_all:
                best_model_all=best_model_MLP
                best_score_all=best_score
        #SVR
        model3=SVR()
        model3.fit(X_train,y_train)
        score=model3.score(X_test,y_test)
        if score > best_SVR:
            best_SVR=score
            best_PCA_SVR=n_components
        if score > best_score_all:
            best_model_all=model3
            best_score_all=score
        #RandomForestRegressor
        best_score,best_depth,best_randomState,best_PCA, best_model_RF=bestRF(X_train,X_test,y_train,y_test,n_components,MaxDepth)
        if best_score>best_RandomForest:
            best_RandomForest=best_score
            best_PCA_RF=best_PCA
            max_depth_RF=best_depth
            random_state_RF=best_randomState
        if best_score>best_score_all:
            best_model_all=best_model_RF
            best_score_all=best_score
        #DecisionTree
        best_score,best_depth,best_randomState,best_PCA, best_model_DT=bestDT(X_train,X_test,y_train,y_test,n_components,MaxDepth)
        if best_score>best_DecisionTree:
            best_DecisionTree=best_score
            best_PCA_DT=best_PCA
            max_depth_DT=best_depth
            random_state_DT=best_randomState
        if best_score>best_score_all:
            best_model_all=best_model_DT
            best_score_all=best_score
        #LinearRregression
        model6=LinearRegression()
        model6.fit(X_train,y_train)
        score=model6.score(X_test,y_test)
        if score > best_LinearRegression:
            best_LinearRegression=score
            best_PCA_LinearRegression=n_components
        if score > best_score_all:
            best_model_all=model6
            best_score_all=score
            
    print('MLP:',best_MLP,'with ',best_PCA_MLP,' components, number of hidden layers: ',best_Layers_MLP,' and number of neurons: ',best_Neurons_MLP)
    print('SVR:',best_SVR,'with ',best_PCA_SVR,' components')
    print('Random forest:',best_RandomForest,' max_depth=',max_depth_RF,' random_state:',random_state_RF,'with ',best_PCA_RF,' components')
    print('Decision tree:',best_DecisionTree,' max_depth=',max_depth_DT,' random_state:',random_state_DT,'with ',best_PCA_DT,' components')
    print('Linear regression:',best_LinearRegression,'with ',best_PCA_LinearRegression,' components')
    print(best_score_all)
    return best_model_all,best_score_all



#This function uses the three functions from above to find the best model for each Isomap result
def forecastingModelISO(Dataset,Label,MLP=False,MaxLayers=100,neighbors=20,MaxDepth=10):
    best_model_all=[]
    best_score_all=0
    intial_components=len(Dataset.columns)
    #Here you set the Isomap and try different values for n_components
    n_components=1
    best_MLP=0
    best_ISO_MLP=0
    best_Layers_MLP=0
    best_Neurons_MLP=0
    best_neighbor_MLP=0
    best_SVR=0
    best_ISO_SVR=0
    best_neighbor_SVR=0
    best_RandomForest=0
    max_depth_RF=0
    random_state_RF=0
    best_ISO_RF=0
    best_neighbor_RF=0
    best_DecisionTree=0
    max_depth_DT=0
    random_state_DT=0
    best_ISO_DT=0
    best_neighbor_DT=0
    best_LinearRegression=0
    best_ISO_LinearRegression=0
    best_neighbor_LinearRregression=0
    # Without Isomap
    #Here you split the data
    X_train, X_test, y_train, y_test = train_test_split(Dataset, Label, test_size=0.33)
    #MLP
    if MLP==True:
        best_score,best_ISO,best_Layers,best_Neurons,best_model_MLP=bestMLP(X_train,X_test,y_train,y_test,intial_components,MaxLayers)
        if best_score>best_MLP:
            best_MLP=best_score
            best_ISO_MLP=best_ISO
            best_Layers_MLP=best_Layers
            best_Neurons_MLP=best_Neurons
        if best_score>best_score_all:
            best_model_all=best_model_MLP
            best_score_all=best_score
    #SVR
    model3=SVR()
    model3.fit(X_train,y_train)
    score=model3.score(X_test,y_test)
    if score > best_SVR:
        best_SVR=score
        best_ISO_SVR=intial_components
    if score > best_score_all:
        best_model_all=model3
        best_score_all=score
    #RandomForestRegressor
    best_score,best_depth,best_randomState,best_ISO, best_model_RF=bestRF(X_train,X_test,y_train,y_test,intial_components,MaxDepth)
    if best_score>best_RandomForest:
        best_RandomForest=best_score
        best_ISO_RF=best_ISO
        max_depth_RF=best_depth
        random_state_RF=best_randomState
    if best_score>best_score_all:
        best_model_all=best_model_RF
        best_score_all=best_score
    #DecisionTree
    best_score,best_depth,best_randomState,best_ISO, best_model_DT=bestDT(X_train,X_test,y_train,y_test,intial_components,MaxDepth)
    if best_score>best_DecisionTree:
        best_DecisionTree=best_score
        best_ISO_DT=best_ISO
        max_depth_DT=best_depth
        random_state_DT=best_randomState
    if best_score>best_score_all:
        best_model_all=best_model_DT
        best_score_all=best_score
    #LinearRregression
    model6=LinearRegression()
    model6.fit(X_train,y_train)
    score=model6.score(X_test,y_test)
    if score > best_LinearRegression:
        best_LinearRegression=score
        best_ISO_LinearRegression=intial_components
    if score > best_score_all:
        best_model_all=model6
        best_score_all=score
    # With Isomap
    for n_components in range(1,len(Dataset.columns)):
        neighbor=1
        for neighbor in range(1,neighbors+1):
            model=manifold.Isomap(n_neighbors=neighbor, n_components=n_components)
            model.fit(Dataset)
            T=model.transform(Dataset)
            #Here you split the data
            X_train, X_test, y_train, y_test = train_test_split(T, Label, test_size=0.33)
            #MLP
            if MLP==True:
                best_score,best_ISO,best_Layers,best_Neurons,best_model_MLP=bestMLP(X_train,X_test,y_train,y_test,n_components,MaxLayers)
                if best_score>best_MLP:
                    best_MLP=best_score
                    best_ISO_MLP=best_ISO
                    best_Layers_MLP=best_Layers
                    best_Neurons_MLP=best_Neurons
                    best_neighbor_MLP=neighbor
                if best_score>best_score_all:
                    best_model_all=best_model_MLP
                    best_score_all=best_score
            #SVR
            model3=SVR()
            model3.fit(X_train,y_train)
            score=model3.score(X_test,y_test)
            if score > best_SVR:
                best_SVR=score
                best_ISO_SVR=n_components
                best_neighbor_SVR=neighbor
            if score > best_score_all:
                best_model_all=model3
                best_score_all=score
            #RandomForestRegressor
            best_score,best_depth,best_randomState,best_ISO,best_model_RF=bestRF(X_train,X_test,y_train,y_test,n_components,MaxDepth)
            if best_score>best_RandomForest:
                best_RandomForest=best_score
                best_ISO_RF=best_ISO
                max_depth_RF=best_depth
                random_state_RF=best_randomState
                best_neighbor_RF=neighbor
            if best_score>best_score_all:
                best_model_all=best_model_RF
                best_score_all=best_score
            #DecisionTree
            best_score,best_depth,best_randomState,best_ISO,best_model_DT=bestDT(X_train,X_test,y_train,y_test,n_components,MaxDepth)
            if best_score>best_DecisionTree:
                best_DecisionTree=best_score
                best_ISO_DT=best_ISO
                max_depth_DT=best_depth
                random_state_DT=best_randomState
                best_neighbor_DT=neighbor
            if best_score>best_score_all:
                best_model_all=best_model_DT
                best_score_all=best_score
            #LinearRregression
            model6=LinearRegression()
            model6.fit(X_train,y_train)
            score=model6.score(X_test,y_test)
            if score > best_LinearRegression:
                best_LinearRegression=score
                best_ISO_LinearRegression=n_components
                best_neighbor_LinearRregression=neighbor
            if score>best_score_all:
                best_model_all=model6
                best_score_all=score
                
    print('MLP:',best_MLP,'with ',best_ISO_MLP,' components, neighbors: ',best_neighbor_MLP,'number of hidden layers: ',best_Layers_MLP,' and number of neurons: ',best_Neurons_MLP)
    print('SVR:',best_SVR,'with ',best_ISO_SVR,' components and neighbors: ',best_neighbor_SVR)
    print('Random forest:',best_RandomForest,' max_depth=',max_depth_RF,' random_state:',random_state_RF,'with ',best_ISO_RF,' components and neighbors: ',best_neighbor_RF)
    print('Decision tree:',best_DecisionTree,' max_depth=',max_depth_DT,' random_state:',random_state_DT,'with ',best_ISO_DT,' components and neighbors: ', best_neighbor_DT)
    print('Linear regression:',best_LinearRegression,'with ',best_ISO_LinearRegression,' components and neighbors: ',best_neighbor_LinearRregression)
    print(best_score_all)
    return best_model_all, best_score_all



# This function uses the two functions from above and return the model with best score
def bestForecastModel(Dataset,label,neighbors=20,MLP=False,MaxLayers=100,MaxDepth=10):
    bestModel=[]
    bestScore=0
    model1,score1=forecastingModel(Dataset,label,MLP,MaxLayers,MaxDepth)
    model2,score2=forecastingModelISO(Dataset,label,MLP,MaxLayers,neighbors,MaxDepth)
    if score1>score2:
        bestModel=model1
        bestScore=score1
    else:
        bestModel=model2
        bestScore=score2
    return bestModel,bestScore



