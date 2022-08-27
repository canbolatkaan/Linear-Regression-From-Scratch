from typing import List

class LinearRegression:
    def __init__(self, learning_rate: float, epoch: int):
        self.learning_rate= learning_rate
        self.epoch=epoch
        pass

    def fit(self, boy: List[int], kilo: List[int], target:List[int]):
        
        global newm1
        global newm2
        global newbias
        myepoch=[]
        myloss=[]
        myRsquaredAccuracy=[]
        mylossTest=[]
        myRsquaredAccuracyTest=[]
        
        m1list=[]
        m2list=[]
        biaslist=[]
        m1=1
        m2=2
        b=0
        m1list.append(m1)
        m2list.append(m2)
        biaslist.append(b)
        lossm1,lossm2,lossbias=self.calculateLoss(boy,kilo,target,m1,m2,b)
        newm1=m1-self.learning_rate*lossm1
        newm2=m2-self.learning_rate*lossm2
        newbias=b-self.learning_rate*lossbias
        for i in range (self.epoch-1):
            lossm1,lossm2,lossbias=self.calculateLoss(boy,kilo,target,newm1,newm2,newbias)
            newm1=newm1-self.learning_rate*lossm1
            newm2=newm2-self.learning_rate*lossm2
            newbias=newbias-self.learning_rate*lossbias
            xhat=self.predict(boy,kilo)
            loss=0
            
            
            m1list.append(newm1)
            m2list.append(newm2)
            biaslist.append(newbias)
            
            for j in range(len(xhat)):
                loss+=(xhat[j]-target[j])**2
                
            mean=sum(target)/len(target)
            sumofsquares = 0
            sumofresiduals = 0
            for j in range(len(target)) :
                sumofsquares += (target[j] - mean) ** 2
                sumofresiduals += (target[j] - xhat[j]) **2
                
            score  = 1 - (sumofresiduals/sumofsquares)
            myRsquaredAccuracy.append(score)
            loss=loss/len(xhat)
            myepoch.append(i)
            myloss.append(loss)
            #########################
            
                         
            #print(acc)
            #print(i)
    
        return myepoch,myloss,myRsquaredAccuracy,newm1,newm2,newbias,m1list,m2list,biaslist

    def predict(self, boy: List[int], kilo: List[int]):
        #print(newm1,newm2,newbias)
        myTarget=[]
        for i in range (len(boy)):
            myTarget.append(newm1*boy[i]+newm2*kilo[i]+newbias)
        
                    
        return myTarget
    def calculateLoss(self, boy: List[int], kilo: List[int], target, m1:int, m2:int, bias:int ):
        #derivative m1,sm2,bias loss
            oldm1=0
            oldm2=0
            oldBias=0
            for i in range (len(target)):
                oldm1=oldm1+ (m1*boy[i]+m2*kilo[i]+bias-target[i]) * boy[i]
                oldm2=oldm2+ (m1*boy[i]+m2*kilo[i]+bias-target[i]) * kilo[i]
                oldBias=oldBias+ (m1*boy[i]+m2*kilo[i]+bias-target[i])
            
            oldm1=2*oldm1/len(target)
            oldm2=2*oldm2/len(target)
            oldBias=2*oldBias/len(target)
            return oldm1,oldm2,oldBias
   
                   
# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    

