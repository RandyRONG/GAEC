import numpy as np
import random
import time
import pandas as pd
import math
import matplotlib.pyplot as plt


class SimpleGeneticAlgorithm():
    def __init__(self,DMat,start_point):
        self.start_point = start_point
        self.item_num = len(DMat[0])
        self.left_parts = [i for i in range(self.item_num)]
        self.left_parts.remove(self.start_point)
        self.DMat = DMat
        self.seed_num = 1000
        self.offspring_num = int(1*self.seed_num)
        self.macro_alpha = 0.7
        self.micro_alpha = 0.16
        self.numIters = 1000
        self.tournament_times = self.seed_num
        self.tournament_num = 5

    def InitializeProcess(self):
        population = []
        for idx in range(self.seed_num):
            temp_list = self.left_parts[:]
            temp_occu = []
            while len(temp_list)>0:
                t_num = random.choice(temp_list)
                temp_occu.append(t_num)
                temp_list.remove(t_num)
            population.append(temp_occu)
        return population
        
    def Fitness(self,population,record_values_dict):
        record_lists = []
        for individual in population:
            key_ = ' '.join([str(i) for i in [self.start_point]+individual[:]+[self.start_point]])
            if key_ in list(record_values_dict.keys()):
                value = record_values_dict[key_]
            else:  
                value = self.DMat[self.start_point][individual[0]]
                for idx,item in enumerate(individual[:-1]):
                    value += self.DMat[individual[idx]][individual[idx+1]]
                value += self.DMat[individual[-1]][self.start_point]
                # print (key_,value)
                record_values_dict[key_] = value
            record_lists.append([individual,1/(value)])
        # print (record_lists)
        sum_value = sum([i[1] for i in record_lists])
        roulette_wheel_list = []
        cumm_prob = 0
        for item in record_lists:
            cumm_prob_0 = cumm_prob
            prob_ = item[1]/sum_value
            cumm_prob += prob_
            roulette_wheel_list.append([item[0],[cumm_prob_0,cumm_prob]])
        return roulette_wheel_list,record_values_dict

    ### selecting parents
    def RouletteCrossOver(self,roulette_wheel_list):
        offsprings = []
        for turn in range(self.offspring_num):
            select_items = []
            for double in range(2):
                decision_prob = random.uniform(0,1)
                for item in roulette_wheel_list:
                    if decision_prob>=item[1][0] and decision_prob<item[1][1]:
                        select_items.append(item[0])
            decision_orders = random.sample(range(0,self.item_num), 2)
            decision_orders.sort()
            while (decision_orders[0] == decision_orders[1]) or (decision_orders[0] <= 1 and  decision_orders[1] >= self.item_num-2):
                decision_orders = random.sample(range(0,self.item_num), 2)
                decision_orders.sort()
            new_items = []
            # print ('parents: ',select_items,'\n','decision_orders:',decision_orders)
            for item_idx,new_item in enumerate(select_items):
                stable_part = new_item[decision_orders[0]:decision_orders[1]+1]
                original_length = len(stable_part)
                # non_stable_part = new_item[:decision_orders[0]][:]+ new_item[decision_orders[1]+1:][:]  
                pointer = decision_orders[1]+1
                if pointer <= len(self.left_parts[:])-1 or len(stable_part) == original_length:
                    acco_index = select_items[1-item_idx].index(stable_part[-1])
                else:
                    acco_index = select_items[1-item_idx].index(stable_part[0])
                while len(stable_part) != len(self.left_parts[:]):
                    # time.sleep(1)
                    # print ('child:',stable_part,)
                    # print ('parent1:',pointer,new_item)
                    # print ('parent2:',select_items[1-item_idx],acco_index)
                    if acco_index == len(new_item)-1:
                        acco_index_next = 0
                    else:
                        acco_index_next = acco_index + 1
                    if select_items[1-item_idx][acco_index_next] in stable_part:
                        acco_index = acco_index_next
                        continue
                    else:
                        if pointer <= len(self.left_parts[:])-1:
                            stable_part = stable_part[:] + [select_items[1-item_idx][acco_index_next]]
                            pointer += 1
                        else:
                            stable_part = [select_items[1-item_idx][acco_index_next]]+stable_part[:]
                new_prex = stable_part[:decision_orders[0]]
                new_prex.reverse()
                stable_part = new_prex[:] + stable_part[decision_orders[0]:]
                # print ('final child:',stable_part)
                new_items.append(stable_part)
                if len(set(new_items[item_idx])) != len(new_items[item_idx]):
                    print (new_items[item_idx])
                    print ('wrong in path!')
                    exit()
            # print ('children: ',new_items)
            offsprings.extend(new_items)
        return offsprings
    
    # def DecayingCrossOver(self,roulette_wheel_list):

    #     return offsprings
                    

    # def Mutation(self,population):
    #     new_population = []
    #     for individual in population:
    #         decision_prob = random.uniform(0,1)
    #         if decision_prob >= self.alpha:
    #             continue
    #         new_population.append([1-i for i in individual])
    #     return new_population

    def Mutation(self,population):
        new_population = []
        for individual in population:
            decision_prob = random.uniform(0,1)
            if decision_prob >= self.macro_alpha:
                continue
            new_population.append(individual[::-1])
        for individual in population:
            new_individual = individual[:]
            for sub_ in new_individual:
                decision_prob = random.uniform(0,1)
                if decision_prob >= self.micro_alpha:
                    continue
                i=new_individual.index(sub_)
                j=i
                while j == i:
                    j=new_individual.index(random.choice(new_individual))
                new_individual[i],new_individual[j]=new_individual[j],new_individual[i]
            if new_individual == individual:
                continue
            new_population.append(new_individual)  
        new_population.extend(population)  
        return new_population

    ### cost; the selection method
    
    def Elimination(self,population,record_values_dict):
        selected_population = []
        competing_individuals = random.sample(population, min(self.tournament_num*self.tournament_times,len(population)))
        total_record_list = []
        total_values = []
        for time in range(self.tournament_times):
            record_lists = []
            competing_group = competing_individuals[time*self.tournament_num:time*self.tournament_num+self.tournament_num]
            if len(competing_group) == 0:
                # print (self.tournament_num*self.tournament_times,len(competing_individuals))
                break
            for individual in competing_group:
                key_ = ' '.join([str(i) for i in [self.start_point]+individual[:]+[self.start_point]])
                if key_ in record_values_dict.keys():
                    value = record_values_dict[key_]
                else:
                    value = self.DMat[self.start_point][individual[0]]
                    for idx,item in enumerate(individual[:-1]):
                        value += self.DMat[individual[idx]][individual[idx+1]]
                    value += self.DMat[individual[-1]][self.start_point]
                    record_values_dict[key_] = value
                record_lists.append([individual,value])
            record_lists.sort(key=lambda x:x[1])
            total_record_list.append(record_lists[0])
            selected_population.append(record_lists[0][0])
            total_values.append(record_lists[0][1])
        # print (record_lists[:10])
        diversities = []
        for i in selected_population:
            if i not in diversities:
                diversities.append(i)
            else:
                continue
        diversity_num = len(diversities)
        set_record_lists = []
        total_record_list.sort(key=lambda x:x[1])
        for i in total_record_list[:self.seed_num]:
            i = [self.start_point]+i[0][:]+[self.start_point],i[1]
            if i in set_record_lists:
                continue
            set_record_lists.append(i)
        print (set_record_lists[:10])
        # print (selected_population[:10])
        return selected_population,record_values_dict,set_record_lists[0],round(np.mean(total_values),4),diversity_num

    # def Selection(self,population):
    #     record_lists = []
    #     for individual in population:
    #         value = self.DMat[self.start_point][individual[0]]
    #         for idx,item in enumerate(individual[:-1]):
    #             value += self.DMat[individual[idx]][individual[idx+1]]
    #         value += self.DMat[individual[-1]][self.start_point]
    #         record_lists.append([individual,value])
    #     record_lists.sort(key=lambda x:x[1])
    #     # print (record_lists[:10])
    #     set_record_lists = []
    #     for i in record_lists[:self.seed_num]:
    #         i = [self.start_point]+i[0][:]+[self.start_point],i[1]
    #         if i in set_record_lists:
    #             continue
    #         set_record_lists.append(i)
    #     print (set_record_lists)
    #     selected_population = [i[0] for i in record_lists[:self.seed_num]]
    #     return selected_population

    def Iteration(self):
        population = self.InitializeProcess()
        final_results = []
        mean_results = []
        diversity_nums = []
        record_values_dict = {}
        time_start=time.time()
        for iteration in range(self.numIters):
            roulette_wheel_list,record_values_dict = self.Fitness(population,record_values_dict)
            # offsprings = self.RouletteCrossOver(roulette_wheel_list)
            offsprings = self.RouletteCrossOver(roulette_wheel_list)
            population = self.Mutation(population)
            population.extend(offsprings)
            population,record_values_dict,best_result,mean_result,diversity_num = self.Elimination(population,record_values_dict)
            diversity_nums.append(diversity_num)
            final_results.append(round(best_result[1],4))
            mean_results.append(mean_result)
            time_end=time.time()
            if time_end-time_start >= 5*60:
                print('time cost',time_end-time_start,'s')
                print ('time is up!')
                print (best_result)
                print (final_results)
                plt.figure(figsize=(7,4)) 
                plt.plot(final_results,'b',lw = 1.5,label = 'best ojective value')
                plt.plot(final_results,'ro') 
                plt.plot(mean_results,'g',lw = 1.5,label = 'mean ojective value')
                plt.plot(mean_results,'ro') 
                plt.grid(axis='y')
                for a,b,c in zip(range(len(mean_results)),mean_results,diversity_nums):
                    plt.text(a, b+2, c, ha='center', va= 'bottom',fontsize=9)
                plt.legend(loc = 0) 
                plt.axis('tight')
                plt.xlabel('turns')
                plt.ylabel('total distance')
                plt.title('the convergence trend; best result: '+str(round(best_result[1],4)))
                plt.show()
                print (round(best_result[1],4))
                exit()
            

if __name__ == "__main__":
    # capacity = int(input())
    # DistanceMatrix=[
    # [ 0, 3, 6, 7, 13, 2, 4, 9],
    # [ 5, 0, 2, 3, 23, 5, 7, 1],
    # [ 6, 4, 0, 2, 4, 8, 19, 1],
    # [ 3, 7, 5, 0, 2, 3, 4, 7],
    # [ 3, 7, 15, 20, 0, 4, 4, 2],
    # [ 5, 2, 5, 3, 2, 0, 4, 7],
    # [ 6, 7, 1, 8, 5, 4, 0, 2],
    # [ 3, 4, 5, 4, 2, 4, 10, 0]]
    
    df = pd.read_csv('tour250.csv',header=None)
     # df = pd.read_csv(filename,header=None)
    DistanceMatrix = []
    for i in range(df.shape[0]):
        row = []
        for j in range(df.shape[1]):
            if df.iloc[i,j] == float("inf"):
                row.append(1e10)
            else:
                row.append(df.iloc[i,j])
            # row.append(df.iloc[i,j])
        DistanceMatrix.append(row)
    # print (DistanceMatrix)
    # print (DistanceMatrix)

    # DistanceMatrix=[
    # [-1, 3, 6, 7],
    # [ 5,-1, 2, 3],
    # [ 6, 4,-1, 2],
    # [ 3, 7, 5,-1]]
    start_point = 0
    EA = SimpleGeneticAlgorithm(DistanceMatrix,start_point)
    SimpleGeneticAlgorithm.Iteration(EA)
    
