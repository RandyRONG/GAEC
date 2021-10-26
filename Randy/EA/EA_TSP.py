import numpy as np
import random
import time


class SimpleGeneticAlgorithm():
    def __init__(self,DMat,start_point):
        self.start_point = start_point
        self.item_num = len(DMat[0])
        self.left_parts = [i for i in range(self.item_num)]
        self.left_parts.remove(self.start_point)
        self.DMat = DMat
        self.seed_num = 1000
        self.offspring_num = self.seed_num
        self.macro_alpha = 0.5
        self.micro_alpha = 0.2
        self.numIters = 100

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
        
    def Fitness(self,population):
        record_lists = []
        for individual in population:
            value = self.DMat[self.start_point][individual[0]]
            for idx,item in enumerate(individual[:-1]):
                value += self.DMat[individual[idx]][individual[idx+1]]
            value += self.DMat[individual[-1]][self.start_point]
            record_lists.append([individual,1/value])
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
    ### selecting parents
    def CrossOver(self,roulette_wheel_list):
        offsprings = []
        for turn in range(self.offspring_num):
            select_items = []
            for double in range(2):
                decision_prob = random.uniform(0,1)
                for item in roulette_wheel_list:
                    if decision_prob>=item[1][0] and decision_prob<item[1][1]:
                        select_items.append(item[0])
            decision_order = random.choice(range(1,self.item_num-1))
            new_items = [select_items[0][:decision_order]+select_items[1][decision_order:],\
                            select_items[1][:decision_order]+select_items[0][decision_order:]]
            for item_idx,new_item in enumerate(new_items):
                stable_part = new_item[decision_order:]
                non_stable_part = new_item[:decision_order]        
                diff_part=list(set(self.left_parts).difference(set(stable_part)))
                for sub_idx,sub_ in enumerate(non_stable_part):
                    if sub_ not in stable_part and sub_ not in non_stable_part:
                        continue
                    t_num = random.choice(diff_part)
                    non_stable_part[sub_idx] = t_num
                    diff_part.remove(t_num)
                    if len(diff_part) == 0:
                        break
                new_items[item_idx] = non_stable_part[:] + stable_part[:]
                if len(set(new_items[item_idx])) != len(new_items[item_idx]):
                    print (new_items[item_idx])
                    print ('wrong in path!')
                    exit()
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
    def Selection(self,population):
        record_lists = []
        for individual in population:
            value = self.DMat[self.start_point][individual[0]]
            for idx,item in enumerate(individual[:-1]):
                value += self.DMat[individual[idx]][individual[idx+1]]
            value += self.DMat[individual[-1]][self.start_point]
            record_lists.append([individual,value])
        record_lists.sort(key=lambda x:x[1])
        # print (record_lists[:10])
        set_record_lists = []
        for i in record_lists[:self.seed_num]:
            i = [self.start_point]+i[0][:]+[self.start_point],i[1]
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
    DistanceMatrix=[
    [ 0, 3, 6, 7, 13, 2, 4, 9],
    [ 5, 0, 2, 3, 23, 5, 7, 1],
    [ 6, 4, 0, 2, 4, 8, 19, 1],
    [ 3, 7, 5, 0, 2, 3, 4, 7],
    [ 3, 7, 15, 20, 0, 4, 4, 2],
    [ 5, 2, 5, 3, 2, 0, 4, 7],
    [ 6, 7, 1, 8, 5, 4, 0, 2],
    [ 3, 4, 5, 4, 2, 4, 10, 0]]

    # DistanceMatrix=[
    # [-1, 3, 6, 7],
    # [ 5,-1, 2, 3],
    # [ 6, 4,-1, 2],
    # [ 3, 7, 5,-1]]
    start_point = 5
    EA = SimpleGeneticAlgorithm(DistanceMatrix,start_point)
    SimpleGeneticAlgorithm.Iteration(EA)