import r0817044
import pandas as pd


if __name__ == "__main__":
    
    
    # bestObjectives = []
    # meanObjectives = []
    # for i in range(2):
    a = r0817044.r0817044()
    bestObjective,meanObjective = a.optimize("./tour29.csv")
        # bestObjectives.append(bestObjective)
        # meanObjectives.append(meanObjective)
        
        # lists =zip(bestObjectives,meanObjectives)
        # name=['bestObjective','meanObjectives']
        # df=pd.DataFrame(columns=name,data=lists)       
        # df.to_csv('records.csv',encoding='utf-8',index=None)
    
    
    