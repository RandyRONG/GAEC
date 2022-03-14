import numpy as np
import random
# random.seed( 10 )
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import Reporter
import Levenshtein
from operator import itemgetter
# import line_profiler as lp


class r0817044():
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        # self.start_point = start_point
        # self.item_num = len(DMat[0])
        
        # self.DMat = DMat
        
        self.seed_num = 50
        self.offspring_num = int(1*self.seed_num)
        
        self.macro_alpha = 0.3
        self.micro_alpha = 0.4
        self.micro_alpha2 = 0.01
        self.micro_alpha3 = 0.5
        
        self.numIters = 1000
        self.tournament_times = self.seed_num
        self.tournament_num = 10
        self.retain_prob_1 = 0.1
        self.retain_prob_2 = 0.2
        
        self.new_init_ratio = 2.0
        self.top_ratio = 0.2
        self.worst_ratio = 0.5*(self.top_ratio)
        self.top_ratio_2 = 0.3
        self.permuation_skip = 0.0
        
        
        self.top_cities = 0.3
        
        self.test_br_len = 10
        
        self.LSOMin = 3
        self.DiverseMin = 4
        
        self.ComplTurn = 10
        self.ComplSameTurn = int(2*self.ComplTurn)
        
    
    def PerIndexInitialization(self,DMat, tour_length, start_city,start_point):
        individual = np.empty(tour_length, dtype=np.int32)
        if start_city == None:
            start_city = np.random.randint(0, tour_length)
        else:
            if start_city >= tour_length or start_city < 0:
                raise IndexError("start city out of range!")
        individual[0] = start_point
        individual[1] = start_city
        visited = np.full(tour_length, False, dtype=bool)
        visited[start_city] = True
        visited[start_point] = True
        for individual_index in range(2, tour_length):
            current_city = individual[individual_index - 1]
            min_distance = float('inf')
            candidate = None
            for next_city in range(tour_length):
                if not visited[next_city]:
                    if DMat.item((current_city, next_city)) <= min_distance:
                        min_distance = DMat.item((current_city, next_city))
                        candidate = next_city
            individual[individual_index] = candidate
            visited[candidate] = True
        individual = list(individual)
        while start_point in individual:
            individual.remove(start_point)
        return individual

    def InitializeProcess(self,seed_num,start_point,left_parts,DMat,record_values_dict,not_allowed_cities,picked_cities,PermList,StarPermu,PerAvail,meanObjective,bestObjective, bestSolution):
        old_population = []
        for idx in range(int(self.seed_num/4)):
            temp_list = left_parts[:]
            temp_occu = []
            while len(temp_list)>0:
                t_num = random.choice(temp_list)
                temp_occu.append(t_num)
                temp_list.remove(t_num)
            old_population.append(temp_occu)
        print ('old population: ',len(old_population))
        new_population = []
        if PerAvail:
            ### Good Initialization
            for idx in range(int(self.seed_num/4)):
                temp_list = left_parts[:]+[start_point]
                temp_occu = [start_point]
                temp_count = 0  
                while len(temp_list)>0:
                    
                    pass_sign = 0
                    candidates = []
                    for sub_1 in picked_cities:
                        if temp_occu[-1] == sub_1[0]:
                            candidates.append(sub_1[1])
                    for candidate in candidates:
                        if candidate not in temp_list:
                            continue
                        if candidate == start_point and len(temp_list)!= 1:
                            continue
                        # print (candidate)
                        temp_occu.append(candidate)
                        temp_list.remove(candidate)
                        pass_sign = 1
                        break
                    if pass_sign == 1:
                        continue
                    
                    if len(temp_list) == 0:
                        break
                    if len(temp_list) == 1:
                        temp_occu.append(temp_list[0])
                        temp_list.remove(temp_list[0])
                        break  
                    if temp_count > len(temp_list)+2:
                        break
                    while True:
                        # print (temp_list,temp_count)
                        t_num = random.choice(temp_list)
                        temp_count += 1
                        if len(temp_list) == 0:
                            break
                        if len(temp_list) == 1:
                            temp_occu.append(temp_list[0])
                            temp_list.remove(temp_list[0])
                            break  
                        if temp_count > len(temp_list)+2:
                            break
                        
                        if [temp_occu[-1],t_num] in not_allowed_cities:
                            # print (t_num,'*')
                            
                            continue
                        elif t_num == start_point and len(temp_list)!= 1:
                            continue
                        else:
                            temp_occu.append(t_num)
                            temp_list.remove(t_num)
                            temp_count = 0
                        
                    
                while start_point in temp_occu:
                    temp_occu.remove(start_point)
                if len(set(temp_occu)) == len(temp_occu) and len(temp_occu)+1 == len(DMat[0]):        
                    # print (temp_occu)
                    new_population.append(temp_occu)
            
        print ('new_population: ',len(new_population))  
        # exit()
        new_population_2_,record_values_dict,PermList = self.LSOPermutation(start_point,DMat,new_population,record_values_dict,PermList,self.top_ratio_2,self.worst_ratio,meanObjective,bestObjective, bestSolution)            
        new_population.extend(new_population_2_)   
        
        PerIndex_population = []
        if StarPermu:
            left_cities_chosen = left_parts[:]
            if len(left_cities_chosen) > 500:
                left_cities_chosen = random.sample(left_cities_chosen,500)
            start_time = time.time()
            for startcity in left_cities_chosen:
                if time.time() - start_time > 3*60:
                    break
                individual_PerIndex = self.PerIndexInitialization(DMat,len(DMat[0]),startcity,start_point)         
                PerIndex_population.append(individual_PerIndex)
    
        print ('PerIndex_population num: ',len(PerIndex_population))
            
            
        population = []
        if PerAvail:
            ### no impossible
            for idx in range(int(self.seed_num/4)):
                temp_list = left_parts[:]+[start_point]
                temp_occu = [start_point]
                temp_count = 0  
                while len(temp_list)>0:
                    # print (temp_list,temp_count)
                    t_num = random.choice(temp_list)
                    temp_count += 1
                    if len(temp_list) == 0:
                        break
                    if len(temp_list) == 1:
                        temp_occu.append(temp_list[0])
                        temp_list.remove(temp_list[0])
                        break  
                    if temp_count > len(temp_list)+2:
                        break
                    
                    if [temp_occu[-1],t_num] in not_allowed_cities:
                        # print (t_num,'*')
                        continue
                    elif t_num == start_point and len(temp_list)!= 1:
                        continue
                    else:
                        temp_occu.append(t_num)
                        temp_list.remove(t_num)
                        temp_count = 0
                while start_point in temp_occu:
                    temp_occu.remove(start_point)
                # print (len(set(temp_occu)),len(temp_occu),len(DMat[0]))
                if len(set(temp_occu)) == len(temp_occu) and len(temp_occu)+1 == len(DMat[0]):
                    population.append(temp_occu)
        # print ('original population:',len(population))
        population.extend(old_population)
        population.extend(new_population)
        population.extend(PerIndex_population)
        
        another_population = []
        ### no impossible
        for idx in range(int(self.seed_num/4)):
            temp_list = left_parts[:]+[start_point]
            temp_occu = [start_point]
            temp_count = 0
            orginal_len = len(temp_list)
            while len(temp_list)>int(orginal_len/2):
                t_num = random.choice(temp_list)
                if DMat[temp_occu[-1],t_num] == float("inf"):
                    temp_count += 1
                    if temp_count > len(temp_list)+10:
                        break
                    continue
                temp_occu.append(t_num)
                temp_list.remove(t_num)
                temp_count = 0
            
            temp_occu_reve = [start_point]
            while len(temp_list)>0:
                t_num = random.choice(temp_list)
                if DMat[temp_occu_reve[-1],t_num] == float("inf"):
                    temp_count += 1
                    if temp_count > len(temp_list)+10:
                        break
                    continue
                temp_occu_reve.append(t_num)
                temp_list.remove(t_num)
                temp_count = 0
            temp_occu.extend(temp_occu_reve[::-1])
            while start_point in temp_occu:
                temp_occu.remove(start_point)
            # print (len(set(temp_occu)),len(temp_occu),len(DMat[0]))
            if len(set(temp_occu)) == len(temp_occu) and len(temp_occu)+1 == len(DMat[0]):
                # print (temp_occu)
                another_population.append(temp_occu)
        population.extend(another_population)  
        population_2 = []
        for individual in population:
            key_ = ' '.join([str(i) for i in [start_point]+individual[:]+[start_point]])  
            if key_ in list(record_values_dict.keys()):
                decision_prob = random.uniform(0,1)
                if record_values_dict[key_] <= meanObjective:
                    retain_prob_1 = min(2 * self.retain_prob_1,1.0)
                else:
                    retain_prob_1 = self.retain_prob_1
                if decision_prob > retain_prob_1:
                    continue
                else:
                    population_2.append(individual)
            else:
                population_2.append(individual)
        print ('population num: ',len(population_2))        
        return population_2,record_values_dict,PermList
    
    
    def DiversityConsider(self,population,bigbetter,edit_distance_dict,tourlength,meanObjective,bestObjective, bestSolution):
        diversity_indicators = {}
        unique_population = []
        for i in population:
            if i  in unique_population:
                continue
            unique_population.append(i)
        max_mean_edit_distance = 0
        min_mean_edit_distance = 1e8
        interval = ''
        for idx,individual in enumerate(unique_population):
            if self.reporter.report(meanObjective, bestObjective, bestSolution) <= 20:
                diversity_indicators[interval.join([str(i) for i in individual])] =  0.5*(max_mean_edit_distance+min_mean_edit_distance)
                continue
            edit_distances= []
            # str1 = interval.join([str(i) for i in individual])
            
            if tourlength < 700:
                str1 = interval.join(map(str, individual))
            for idx_2,other_individual in enumerate([i for i in unique_population if i != individual]):
                if tourlength >= 700:
                    edit_distance = len([i for i in range(len(individual)) if individual[i] != other_individual[i]])
                else:
                    str2 = interval.join(map(str, other_individual))
                    key_ = str1 + interval + str2
                    if key_ not in edit_distance_dict:
                        if str2 + interval + str1 not in edit_distance_dict:
                            edit_distance = Levenshtein.distance(str1,str2)
                            edit_distance_dict[key_] = edit_distance
                        else:
                            edit_distance = edit_distance_dict[str2 + interval + str1]
                    else:
                        edit_distance = edit_distance_dict[key_]
                if bigbetter:
                    edit_distances.append(edit_distance+1)
                else:
                    edit_distances.append(1/(edit_distance+1))
            diversity_indicators[interval.join([str(i) for i in individual])] =  np.mean(edit_distances)
            max_mean_edit_distance = max(max_mean_edit_distance,np.mean(edit_distances))
            min_mean_edit_distance = min(min_mean_edit_distance,np.mean(edit_distances))
        
        for key_ in diversity_indicators:
            diversity_indicators[key_] = (diversity_indicators[key_] - min_mean_edit_distance)/(max_mean_edit_distance - min_mean_edit_distance)
        
        return diversity_indicators,edit_distance_dict,interval
       
    def Fitness(self,start_point,DMat,population,record_values_dict,diversity_coefficient,edit_distance_dict,if_go_diverse,diversity_fold,tourlength,meanObjective,bestObjective, bestSolution):
        record_lists = []
        max_dist = 1e8
        
        if if_go_diverse:
            diversity_indicators,edit_distance_dict,interval = self.DiversityConsider(population,True,edit_distance_dict,tourlength,meanObjective,bestObjective, bestSolution)
        
        for individual in population:
            key_ = ' '.join([str(i) for i in [start_point]+individual[:]+[start_point]])
            if key_ in list(record_values_dict.keys()):
                value = record_values_dict[key_]
            else:  
                value = DMat[start_point][individual[0]]
                for idx,item in enumerate(individual[:-1]):
                    value += DMat[individual[idx]][individual[idx+1]]
                value += DMat[individual[-1]][start_point]
                # print (key_,value)
                record_values_dict[key_] = value
            
            
            if value != float("inf") and value  > max_dist:
                max_dist = value
            if value  == float("inf"):
                if if_go_diverse:
                    record_lists.append([individual,1/(max_dist*2)+diversity_coefficient*diversity_indicators[interval.join([str(i) for i in individual])]])
                    # print (1/(max_dist ** 2))
                    # print (diversity_coefficient*diversity_indicators[interval.join([str(i) for i in individual])])
                    # print ('*'*10)
                else:
                    record_lists.append([individual,1/(max_dist*2)])
            else:
                # print (1/(value),diversity_coefficient*diversity_indicators[interval.join([str(i) for i in individual])])
                if if_go_diverse:
                    record_lists.append([individual,1/(value)+diversity_coefficient*diversity_indicators[interval.join([str(i) for i in individual])]])
                    # print (1/(value))
                    # print (diversity_coefficient*diversity_indicators[interval.join([str(i) for i in individual])])
                    # print ('*'*10)
                else:
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
        return roulette_wheel_list,record_values_dict,edit_distance_dict

    ### selecting parents
    def RouletteCrossOver(self,item_num,left_parts,start_point,DMat,roulette_wheel_list,record_values_dict,meanObjective):
        offsprings = []
        for turn in range(self.offspring_num):
            select_items = []
            for double in range(2):
                decision_prob = random.uniform(0,1)
                for item in roulette_wheel_list:
                    if decision_prob>=item[1][0] and decision_prob<item[1][1]:
                        select_items.append(item[0])
            decision_orders = random.sample(range(0,item_num), 2)
            decision_orders.sort()
            while (decision_orders[0] == decision_orders[1]) or (decision_orders[0] <= 1 and  decision_orders[1] >= item_num-2):
                decision_orders = random.sample(range(0,item_num), 2)
                decision_orders.sort()
            new_items = []
            # print ('parents: ',select_items,'\n','decision_orders:',decision_orders)
            for item_idx,new_item in enumerate(select_items):
                stable_part = new_item[decision_orders[0]:decision_orders[1]+1]
                original_length = len(stable_part)
                # non_stable_part = new_item[:decision_orders[0]][:]+ new_item[decision_orders[1]+1:][:]  
                pointer = decision_orders[1]+1
                if pointer <= len(left_parts[:])-1 or len(stable_part) == original_length:
                    acco_index = select_items[1-item_idx].index(stable_part[-1])
                else:
                    acco_index = select_items[1-item_idx].index(stable_part[0])
                while len(stable_part) != len(left_parts[:]):
                    # time.sleep(1)
                    # print ('child:',stable_part,)
                    # print ('parent1:',pointer,new_item)
                    # print ('parent2:',select_items[1-item_idx],acco_index)
                    if acco_index == len(new_item)-1:
                        acco_index_next = 0
                    else:
                        acco_index_next = acco_index + 1
                    # print (item_idx,acco_index_next,acco_index,len(new_item),len(select_items[1-item_idx]))
                    if select_items[1-item_idx][acco_index_next] in stable_part:
                        acco_index = acco_index_next
                        continue
                    else:
                        # print ('pointer',pointer,len(left_parts[:]))
                        if pointer <= len(left_parts[:])-1:
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
        # sured_offsprings,record_values_dict = self.SureOffspring(offsprings,start_point,DMat,record_values_dict)
        # exit()
        
        
        offsprings_2 = []
        current_ = record_values_dict.keys()
        for new_ind in offsprings:
            key_ = ' '.join([str(i) for i in [0]+new_ind[:]+[0]])
            if key_ not in current_:
                offsprings_2.append(new_ind)
            else:
                decision_prob = random.uniform(0,1)
                if record_values_dict[key_] <= meanObjective:
                    retain_prob_1 = min(2 * self.retain_prob_1,1.0)
                else:
                    retain_prob_1 = self.retain_prob_1
                if decision_prob > retain_prob_1:
                    continue
                else:
                    offsprings_2.append(new_ind)
        offsprings = offsprings_2
        
        return offsprings,record_values_dict
    
    
    def KTournamentCrossOver(self,item_num,left_parts,start_point,DMat,roulette_wheel_list,record_values_dict):
        population = roulette_wheel_list
        # print (population[0])
        tournament_num = max(self.tournament_num,int(len(population)/self.tournament_num))
        # tournament_num = 20
        competing_individuals = random.sample(population, min(tournament_num*self.tournament_times,len(population)))
        offsprings = []
        for turn in range(self.offspring_num):
            select_items = []
            competing_group = competing_individuals[turn*tournament_num:turn*tournament_num+tournament_num]
            if len(competing_group) == 0:
                # print (tournament_num*tournament_times,len(competing_individuals))
                break
            # print (competing_group[0])
            select_in = [[i[0],i[1][0]] for i in competing_group]
            select_sub = [i[1][0] for i in competing_group]
            if len(select_sub) < 2:
                break
            if len(list(set(select_sub))) == 1:
                select_items.append(random.sample([i[0] for i in select_in],min(2,len(select_sub))))
            else:
                select_in.sort(key=lambda x:x[1])
                select_items.append(select_in[0][0])
                select_items.append(select_in[1][0])
            decision_orders = random.sample(range(0,item_num), 2)
            decision_orders.sort()
            while (decision_orders[0] == decision_orders[1]) or (decision_orders[0] <= 1 and  decision_orders[1] >= item_num-2):
                decision_orders = random.sample(range(0,item_num), 2)
                decision_orders.sort()
            new_items = []
            # print ('parents: ',select_items,'\n','decision_orders:',decision_orders)
            for item_idx,new_item in enumerate(select_items):
                # print (new_item,len(new_item),decision_orders[0],decision_orders[1]+1)
                stable_part = new_item[decision_orders[0]:decision_orders[1]+1]
                original_length = len(stable_part)
                # non_stable_part = new_item[:decision_orders[0]][:]+ new_item[decision_orders[1]+1:][:]  
                pointer = decision_orders[1]+1
                if pointer <= len(left_parts[:])-1 or len(stable_part) == original_length:
                    acco_index = select_items[1-item_idx].index(stable_part[-1])
                else:
                    acco_index = select_items[1-item_idx].index(stable_part[0])
                while len(stable_part) != len(left_parts[:]):
                    # time.sleep(1)
                    # print ('child:',stable_part,)
                    # print ('parent1:',pointer,new_item)
                    # print ('parent2:',select_items[1-item_idx],acco_index)
                    if acco_index == len(new_item)-1:
                        acco_index_next = 0
                    else:
                        acco_index_next = acco_index + 1
                    # print (item_idx,acco_index_next,acco_index,len(new_item),len(select_items[1-item_idx]))
                    if select_items[1-item_idx][acco_index_next] in stable_part:
                        acco_index = acco_index_next
                        continue
                    else:
                        # print ('pointer',pointer,len(left_parts[:]))
                        if pointer <= len(left_parts[:])-1:
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
        # sured_offsprings,record_values_dict = self.SureOffspring(offsprings,start_point,DMat,record_values_dict)
        # exit()
        return offsprings,record_values_dict
    
    def SureOffspring(self,offsprings,start_point,DMat,record_values_dict):
        sured_offsprings = []
        for idx,offspring in enumerate(offsprings):
            temp_occu = [start_point]
            temp_list = offspring[:]+[start_point]
            temp_temp_list = temp_list[:]
            
            for sub_idx,sub_item in enumerate(temp_list[:]):
                if sub_item in temp_occu:
                    continue
                
                if DMat[temp_occu[-1],sub_item] != float("inf") and sub_item not in temp_occu:
                    temp_occu.append(sub_item)
                    temp_temp_list.remove(sub_item)
                else:
                    temp_temp_list_2 = temp_temp_list[:]
                    sub_add_idx = 0
                    while sub_add_idx<len(temp_temp_list_2):
                        if DMat[temp_occu[-1],temp_temp_list_2[sub_add_idx]] == float("inf"):
                            sub_add_idx += 1
                            continue
                        else:
                            # print (sub_add_idx)
                            temp_occu.append(temp_temp_list_2[sub_add_idx])
                            temp_temp_list.remove(temp_temp_list_2[sub_add_idx])
                            break
            while start_point in temp_occu:
                temp_occu.remove(start_point)
            if len(set(temp_occu)) != len(temp_occu):
                print (temp_occu)
                print ('wrong in path!')
                exit()
            if len(offspring) == len(temp_occu):
                sured_offsprings.append(temp_occu)
        print ('sured_offsprings:',len(offsprings),len(sured_offsprings))
        
        # for il in sured_offsprings:
        #     key_ = ' '.join([str(i) for i in [0]+il[:]+[0]])
        #     if key_ in record_values_dict.keys():
        #         print ('sured:',il,record_values_dict[key_])
        #         if record_values_dict[key_] == float("inf"):
        #             li = [0]+il[:]+[0]
        #             for i in range(len(li)-1):
        #                 print (li[i],li[i+1],DMat[li[i]][li[i+1]])
        #             exit()
        
        sured_offsprings.extend(offsprings)
        return sured_offsprings,record_values_dict

    def Mutation(self,DMat,population,start_point,record_values_dict,meanObjective):
        population_2 = []
        for ind_1 in population:
            if ind_1 in population_2:
                continue
            else:
                population_2.append(ind_1)
        population = population_2
        new_population = []
        
        # inversion
        for individual in population:
            decision_prob = random.uniform(0,1)
            if decision_prob >= self.macro_alpha:
                continue
            new_population.append(individual[::-1])
            
        # for individual in population:
        #     decision_prob = random.uniform(0,1)
        #     if decision_prob >= self.macro_alpha2:
        #         continue
        #     new_population.append(random.sample(individual, len(individual)))
            
        # swap-more than one times
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
        # swap - one time
        for individual in population:
            new_individual = individual[:]
            for sub_ in new_individual:
                decision_prob = random.uniform(0,1)
                if decision_prob >= self.micro_alpha2:
                    continue
                i=new_individual.index(sub_)
                j=i
                while j == i:
                    j=new_individual.index(random.choice(new_individual))
                new_individual[i],new_individual[j]=new_individual[j],new_individual[i]
                break
            if new_individual == individual:
                continue
            # else:
            new_population.append(new_individual)  
        #  Scramble - sub disconnected sequence   
        for individual in population:
            new_individual = individual[:]
            changing_units = []
            changing_idxs = []
            for n_i,sub_ in enumerate(new_individual):
                decision_prob = random.uniform(0,1)
                if decision_prob >= self.micro_alpha3:
                    continue
                changing_idxs.append(n_i)
                changing_units.append(sub_)
            for n_i,sub_ in enumerate(new_individual):
                if n_i in changing_idxs:
                    new_sub = random.choice(changing_units)
                    new_individual[n_i] = new_sub
                    changing_units.remove(new_sub)
                if len(changing_units) == 0:
                    break
                
            if new_individual == individual:
                continue
            # else:
            new_population.append(new_individual)  
        # Scramble - sub-sequence
        for individual in population:
            new_individual = individual[:]
            city_chos = random.sample(range(0,len(new_individual)),2)
            city_chos.sort()
            the_chos_part = new_individual[city_chos[0]:city_chos[1]] 
            the_new_chos_part = random.sample(the_chos_part,len(the_chos_part))
            new_individual[city_chos[0]:city_chos[1]]  = the_new_chos_part
            if new_individual == individual:
                continue
            # else:
            new_population.append(new_individual)  
            
        current_ = record_values_dict.keys()
        new_population_2 = []
        for new_ind in new_population:
            key_ = ' '.join([str(i) for i in [0]+new_ind[:]+[0]])
            if key_ not in current_:
                new_population_2.append(new_ind)
            else:
                if record_values_dict[key_] <= meanObjective:
                    retain_prob_1 = min(2 * self.retain_prob_1,1.0)
                else:
                    retain_prob_1 = self.retain_prob_1
                decision_prob = random.uniform(0,1)
                if decision_prob > retain_prob_1:
                    continue
                else: 
                    new_population_2.append(new_ind)
        new_population = new_population_2
        
        new_population.extend(population)
        # sured_new_population,record_values_dict = self.SureOffspring(new_population,start_point,DMat,record_values_dict)  
        return new_population,record_values_dict

    def LSOPermutation(self,start_point,DMat,population,record_values_dict,PermList,top_rate,worst_ratio,meanObjective,bestObjective, bestSolution):
        population_value_list = []
        for indi in population:
            key_ = ' '.join([str(i) for i in [start_point]+indi[:]+[start_point]])
            if key_ in record_values_dict.keys():
                value_ = record_values_dict[key_]
            else:
                continue
            if value_ == float('inf'):
                continue
            if [indi,value_] in population_value_list:
                continue
            population_value_list.append([indi,value_])
        population_value_list.sort(key=lambda x:x[1])
        top_population_value_list = population_value_list[:min(int(len(population_value_list)*top_rate),int(self.seed_num*top_rate))]
        
        worst_population_value_list = population_value_list[-min(int(len(population_value_list)*worst_ratio),int(self.seed_num*worst_ratio)):]
        
        top_population_value_list_2 = []
        for value_list in [top_population_value_list,worst_population_value_list]:
            for i in value_list:
                if i not in top_population_value_list_2:
                    top_population_value_list_2.append(i)
                    
        top_population_value_list = top_population_value_list_2
        
        print ('permutation num:',len(top_population_value_list))
        top_population_0 = [i[0] for i in top_population_value_list]
        top_population = [[start_point]+i+[start_point] for i in top_population_0 if i not in PermList]
        PermList.extend(top_population_0)
        topper_population = []
        for individual_idx,individual in enumerate(top_population):
            if self.reporter.report(meanObjective, bestObjective, bestSolution) <= 10:
                break
            for idx,city in enumerate(individual[1:-1]):
                for idx_2,city_2 in enumerate(individual[idx+1:-1]):
                    # decision_prob = random.uniform(0,1)
                    # if decision_prob <= self.permuation_skip:
                    #     continue
                    new_individual = individual[:]
                    # print (idx+idx_2+1)
                    new_individual[idx+idx_2+1+1] = individual[idx+1]
                    new_individual[idx+1] = individual[idx+idx_2+1+1]
                    # print (individual)
                    # print (idx+1,idx+idx_2+1+1)
                    # print (new_individual)
                    # print ('-----'*5)
                    sign = 1
                    for test_idx in [idx+1,idx+idx_2+1+1]:
                        if test_idx == 0:
                            if DMat[new_individual[test_idx]][new_individual[test_idx+1]] == float('inf'):
                                sign = 0
                        elif test_idx == len(new_individual)-1:
                            if DMat[new_individual[test_idx-1]][new_individual[test_idx]] == float('inf'):
                                sign = 0
                        else:
                            if DMat[new_individual[test_idx]][new_individual[test_idx+1]] == float('inf') or \
                                DMat[new_individual[test_idx-1]][new_individual[test_idx]] == float('inf'):
                                sign = 0
                    if sign == 0:
                        # print ('888')
                        continue
                    sign = 1
                    for test_idx in [idx+1,idx+idx_2+1+1]:
                        if test_idx == 0:
                            if DMat[new_individual[test_idx]][new_individual[test_idx+1]] >= \
                                DMat[individual[test_idx]][individual[test_idx+1]]:
                                # print (DMat[new_individual[test_idx]][new_individual[test_idx+1]],DMat[individual[test_idx]][individual[test_idx+1]])
                                sign = 0
                        elif test_idx == len(new_individual)-1:
                            if DMat[new_individual[test_idx-1]][new_individual[test_idx]] >= \
                                DMat[individual[test_idx-1]][individual[test_idx]]:
                                # print (DMat[new_individual[test_idx-1]][new_individual[test_idx]],DMat[individual[test_idx-1]][individual[test_idx]])
                                sign = 0
                        else:
                            if DMat[new_individual[test_idx]][new_individual[test_idx+1]] >= \
                                DMat[individual[test_idx]][individual[test_idx+1]]  or \
                                DMat[new_individual[test_idx-1]][new_individual[test_idx]] >= \
                                DMat[individual[test_idx-1]][individual[test_idx]]:
                                # print ('*'*10)
                                # print (DMat[new_individual[test_idx]][new_individual[test_idx+1]],DMat[individual[test_idx]][individual[test_idx+1]])
                                # print (DMat[new_individual[test_idx-1]][new_individual[test_idx]],DMat[individual[test_idx-1]][individual[test_idx]])
                                # print ('*'*10)
                                sign = 0
                    if sign == 0:
                        continue
                    while start_point in new_individual:
                        new_individual.remove(start_point)
                    key_ = ' '.join([str(i) for i in [start_point]+new_individual[:]+[start_point]])
                    if key_ in list(record_values_dict.keys()):
                        value = record_values_dict[key_]
                    else:  
                        value = DMat[start_point][new_individual[0]]
                        for idx_3,item in enumerate(new_individual[:-1]):
                            value += DMat[new_individual[idx_3]][new_individual[idx_3+1]]
                        value += DMat[new_individual[-1]][start_point]
                        # print (key_,value)
                        record_values_dict[key_] = value
                    # print (new_individual)
                    # exit()
                    if value >= top_population_value_list[individual_idx][1]:
                        decision_prob = random.uniform(0,1)
                        if record_values_dict[key_] <= meanObjective:
                            retain_prob_2 = min(1 * self.retain_prob_2,1.0)
                        else:
                            retain_prob_2 = self.retain_prob_2
                        if decision_prob >= retain_prob_2:
                            continue
                        else:
                            population.append(new_individual)
                    else:
                        # print ('new LSO:')
                        # print (new_individual)
                        # exit()
                        population.append(new_individual)
                
        
        
        return population,record_values_dict,PermList
        
    
    ### cost; the selection method
    
    def Elimination(self,start_point,DMat,population,record_values_dict,diversity_coefficient,edit_distance_dict,if_go_diverse,diversity_fold,tourlength,meanObjective,bestObjective, bestSolution):
        
        if if_go_diverse:
            diversity_indicators,edit_distance_dict,interval = self.DiversityConsider(population,False,edit_distance_dict,tourlength,meanObjective,bestObjective, bestSolution)
            
        selected_population = []
        tournament_num = max(self.tournament_num,int(len(population)/self.tournament_times))
        competing_individuals = random.sample(population, min(tournament_num*self.tournament_times,len(population)))
        total_record_list = []
        total_values = []
        for time in range(self.tournament_times):
            record_lists = []
            competing_group = competing_individuals[time*tournament_num:time*tournament_num+tournament_num]
            if len(competing_group) == 0:
                # print (tournament_num*self.tournament_times,len(competing_individuals))
                break
            for individual in competing_group:
                key_ = ' '.join([str(i) for i in [start_point]+individual[:]+[start_point]])
                # diversity_coefficient*diversity_indicators[interval.join([str(i) for i in individual])]
                if key_ in record_values_dict.keys():
                    if if_go_diverse:
                        value = record_values_dict[key_] 
                        value_diverse = value + ((diversity_fold*2))*diversity_indicators[interval.join([str(i) for i in individual])]
                    else:
                        value = record_values_dict[key_]
                        value_diverse = value
                else:
                    value = DMat[start_point][individual[0]]
                    for idx,item in enumerate(individual[:-1]):
                        value += DMat[individual[idx]][individual[idx+1]]
                    value += DMat[individual[-1]][start_point]
                    record_values_dict[key_] = value
                    if if_go_diverse:
                        value_diverse = value + diversity_coefficient*diversity_indicators[interval.join([str(i) for i in individual])]
                    else:
                        value_diverse = value
                record_lists.append([individual,value_diverse,value,value_diverse-value])
            # record_lists.sort(key=lambda x:x[1])
            select_individual,value_diverse_f,value_f,value_diverse_offset = min(record_lists, key=lambda x:x[1])
            total_record_list.append([select_individual,value_diverse_f,value_f,value_diverse_offset])
            selected_population.append(select_individual)
            # total_record_list.append([record_lists[0][0],value,value_diverse-value])
            # selected_population.append(record_lists[0][0])
            if record_lists[0][1] != float('inf'):
                total_values.append(record_lists[0][2])
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
            i = [start_point]+i[0][:]+[start_point],i[1],i[2],i[3]
            if i in set_record_lists:
                continue
            set_record_lists.append(i)
        print ([[i[1],i[2],i[3]] for i in set_record_lists[:10]])
        # print (selected_population[:10])
        return selected_population,record_values_dict,set_record_lists[0],round(np.mean(total_values),4),diversity_num,edit_distance_dict

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
    
    def FindSub(self,DMat):
        not_allowed_cities = []
        calculated_cities = []
        for i, city_1 in enumerate(DMat):
            for j, city_2 in enumerate(city_1):
                if i == j:
                    continue
                if DMat[i][j] == float('inf'):
                    not_allowed_cities.append([i,j])
                else:
                    calculated_cities.append([[i,j],DMat[i][j]])
        calculated_cities.sort(key=lambda x:x[1])
        picked_cities = [i[0] for i in calculated_cities[:int(self.top_cities*len(calculated_cities))]]
        return not_allowed_cities,picked_cities
            
        

    def optimize(self,filename):
        
        tourlength = int(filename.replace('./','').split('.')[0].replace('tour',''))
        
        file = open(filename)
        
        if tourlength <= 200:
            PerAvail = True
        else:
            PerAvail = False
        distanceMatrix = np.loadtxt(file, delimiter=",")
        DMat = distanceMatrix
        not_allowed_cities,picked_cities = self.FindSub(DMat)
        start_point = 0
        item_num = len(DMat[0])
        left_parts = [i for i in range(item_num)]
        left_parts.remove(start_point)
        start_point = 0
        file.close()
        record_values_dict = {}
        final_results = []
        mean_results = []
        diversity_nums = []
        PermList = []
        edit_distance_dict = {}
        
        # profile = lp.LineProfiler(self.optimize)
        # profile.enable()
        
        meanObjective = 0.0
        bestObjective = 0.0
        bestSolution = np.array([1,2,3,4,5])
        
        population,record_values_dict,PermList = self.InitializeProcess(self.seed_num,start_point,left_parts,DMat,record_values_dict,not_allowed_cities,picked_cities,PermList, True,PerAvail,0.0,bestObjective, bestSolution)
        
        # time_start=time.time()
        yourConvergenceTestsHere = True
        
        if_go_diverse = False
        record_best_results = []
        if len(left_parts) < 100:
            diversity_fold = 4e1
        else:
            diversity_fold = 1e1
        rounds_num = 0
        
        while (yourConvergenceTestsHere):
            rounds_num += 1
            diversity_coefficient = 1/(diversity_fold*max(bestObjective,1))
            roulette_wheel_list,record_values_dict,edit_distance_dict = self.Fitness(start_point,DMat,population,record_values_dict,diversity_coefficient,edit_distance_dict,if_go_diverse,diversity_fold,tourlength,meanObjective,bestObjective, bestSolution)
            ### RouletteCrossOver  KTournamentCrossOver
            offsprings,record_values_dict = self.RouletteCrossOver(item_num,left_parts,start_point,DMat,roulette_wheel_list,record_values_dict,meanObjective)
            population.extend(offsprings)
            new_population_final,record_values_dict = self.Mutation(DMat,population,start_point,record_values_dict,meanObjective)
            population.extend(new_population_final)
            roulette_wheel_list,record_values_dict,edit_distance_dict = self.Fitness(start_point,DMat,population,record_values_dict,diversity_coefficient,edit_distance_dict,if_go_diverse,diversity_fold,tourlength,meanObjective,bestObjective, bestSolution)
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if (timeLeft <= self.LSOMin*60  and rounds_num > self.ComplTurn) or  (rounds_num > self.ComplSameTurn and len(set(record_best_results[-self.ComplSameTurn:])) == 1):
                population,record_values_dict,PermList = self.LSOPermutation(start_point,DMat,population,record_values_dict,PermList,self.top_ratio,self.worst_ratio,meanObjective,bestObjective, bestSolution)
                # print (PermList[:2],len(PermList))
            population,record_values_dict,best_result,mean_result,diversity_num,edit_distance_dict = self.Elimination(start_point,DMat,population,record_values_dict,diversity_coefficient,edit_distance_dict,if_go_diverse,diversity_fold,tourlength,meanObjective,bestObjective, bestSolution)
            if (timeLeft <= self.DiverseMin*60 and rounds_num > self.ComplTurn) or (rounds_num > int(self.ComplSameTurn) and len(set(record_best_results[-self.ComplSameTurn:])) == 1):
                if_go_diverse = True
            new_init_population,record_values_dict,PermList = self.InitializeProcess(int(self.seed_num*self.new_init_ratio),start_point,left_parts,DMat,record_values_dict,not_allowed_cities,picked_cities,PermList,False,PerAvail,meanObjective,bestObjective, bestSolution)
            population.extend(new_init_population)
            bestObjective = best_result[2]
            
            bestSolution = np.array(best_result[0][:-1])
            record_best_results.append(bestObjective)
            if len(record_best_results) > self.test_br_len:
                if len(set(record_best_results[-self.test_br_len:])) == 1:
                    self.macro_alpha += 0.01
                    self.macro_alpha = min(self.macro_alpha,1.0)
                    self.micro_alpha += 0.01
                    self.micro_alpha = min(self.micro_alpha,0.8)
                    
                    self.micro_alpha2 = random.random()
                    self.micro_alpha3 = random.random()
                     
                    self.new_init_ratio += 0.1
                    self.new_init_ratio = min(self.new_init_ratio,3.0)
                    if timeLeft <= self.LSOMin*60  and rounds_num > self.ComplTurn:
                        self.top_ratio += 0.01
                        self.top_ratio = min(self.top_ratio,1.0)
                        self.top_ratio_2 += 0.01
                        self.top_ratio_2 = min(self.top_ratio_2,1.0)
                    if if_go_diverse and timeLeft <= self.DiverseMin*60  and rounds_num > self.ComplTurn:
                        diversity_fold -= 0.1
                        diversity_fold = max(1,diversity_fold)
                    print (self.macro_alpha, self.micro_alpha, self.micro_alpha2, self.new_init_ratio, diversity_fold, self.top_ratio, self.top_ratio_2)
            meanObjective = mean_result
            print (round(bestObjective,2),round(meanObjective,2))
            diversity_nums.append(diversity_num)
            final_results.append(round(best_result[2],4))
            mean_results.append(mean_result)
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
            
        
        
        # profile.disable()
        # profile.print_stats()
        # print ('time is up!')
        # print (bestSolution,round(bestObjective,4),meanObjective)
        # plt.figure(figsize=(7,4)) 
        # plt.plot(final_results,'b',lw = 1.5,label = 'best ojective value')
        # plt.plot(final_results,'ro') 
        # plt.plot(mean_results,'g',lw = 1.5,label = 'mean ojective value')
        # plt.plot(mean_results,'ro') 
        # plt.grid(axis='y')
        # for a,b,c in zip(range(len(mean_results)),mean_results,diversity_nums):
        #     plt.text(a, b+2, c, ha='center', va= 'bottom',fontsize=9)
        # plt.legend(loc = 0) 
        # plt.axis('tight')
        # plt.xlabel('turns')
        # plt.ylabel('total distance')
        # # plt.title('the convergence trend; best result: '+str(round(best_result[2],4)))
        # plt.show()
        
        return round(bestObjective,2),round(meanObjective,2)
            

