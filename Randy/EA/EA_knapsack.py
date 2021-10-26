import numpy as np
import random
import time

def DictProcess(weights):
    weights_dict = {}
    for i in range(len(weights)):
        weights_dict[i] = weights[i]
    return weights_dict

class SimpleGeneticAlgorithm():
    def __init__(self,weights,values,capacity):
        self.capacity = capacity
        self.weights = weights
        self.weights_dict = DictProcess(weights)
        self.values = values
        self.values_dict = DictProcess(values)
        self.item_num = len(self.weights)
        self.seed_num = 1000
        self.offspring_num = self.seed_num
        self.macro_alpha = 0.5
        self.micro_alpha = 0.2
        self.numIters = 100

    def InitializeProcess(self):
        population = []
        for idx in range(self.seed_num):
            population.append([1 if random.uniform(0,1) >= 0.5 else 0 for i in [0]*self.item_num])
        return population
        
    def Fitness(self,population):
        record_lists = []
        for individual in population:
            value = 0
            weight = 0
            for idx,item in enumerate(individual):
                if item == 0:
                    continue
                value += self.values_dict[idx]
                weight += self.weights_dict[idx]
                if weight > self.capacity:
                    break
            if weight > self.capacity:
                continue
            record_lists.append([individual,value])
        # print (record_lists)
        sum_value = sum([i[1] for i in record_lists])
        roulette_wheel_list = []
        cumm_prob = 0
        for item in record_lists:
            cumm_prob_0 = cumm_prob
            prob_ = item[1]/sum_value
            cumm_prob += prob_
            roulette_wheel_list.append([item[0],[cumm_prob_0,cumm_prob]])
        return roulette_wheel_list
    
    def CrossOver(self,roulette_wheel_list):
        offsprings = []
        for turn in range(self.offspring_num):
            select_items = []
            for double in range(2):
                decision_prob = random.uniform(0,1)
                for item in roulette_wheel_list:
                    if decision_prob>=item[1][0] and decision_prob<item[1][1]:
                        select_items.append(item[0])
            decision_order = random.choice(range(1,self.item_num))
            new_items = [select_items[0][:decision_order]+select_items[1][decision_order:],\
                            select_items[1][:decision_order]+select_items[0][decision_order:]]
            offsprings.extend(new_items)
        return offsprings
                    

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
            new_population.append([1-i for i in individual])
        for individual in population:
            new_individual = []
            for sub_ in individual:
                decision_prob = random.uniform(0,1)
                if decision_prob >= self.micro_alpha:
                    new_individual.append(sub_)
                    continue
                new_individual.append(1-sub_)
            if new_individual == individual:
                continue
            new_population.append(new_individual)  
        new_population.extend(population)  
        return new_population

    def Selection(self,population):
        record_lists = []
        for individual in population:
            value = 0
            weight = 0
            for idx,item in enumerate(individual):
                if item == 0:
                    continue
                value += self.values_dict[idx]
                weight += self.weights_dict[idx]
                if weight > self.capacity:
                    break
            if weight > self.capacity:
                continue
            record_lists.append([individual,value])
        record_lists.sort(key=lambda x:x[1],reverse=True)
        # print (record_lists[:10])
        set_record_lists = []
        for i in record_lists[:self.seed_num]:
            if i in set_record_lists:
                continue
            set_record_lists.append(i)
        print (set_record_lists)
        selected_population = [i[0] for i in record_lists[:self.seed_num]]
        return selected_population
    
    def Iteration(self):
        population = self.InitializeProcess()
        for iteration in range(self.numIters):
            roulette_wheel_list = self.Fitness(population)
            offsprings = self.CrossOver(roulette_wheel_list)
            population = self.Mutation(population)
            population.extend(offsprings)
            population = self.Selection(population)

if __name__ == "__main__":
    # capacity = int(input())
    capacity = 12
    weights = [4,6,2,2,5,1]
    values = [8,10,6,3,7,2]
    EA = SimpleGeneticAlgorithm(weights,values,capacity)
    SimpleGeneticAlgorithm.Iteration(EA)
