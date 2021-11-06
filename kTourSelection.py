import numpy as np
import random
import time
import pandas as pd


class SimpleGeneticAlgorithm():
    def __init__(self, DMat, start_point,
        seed_num = 1000,
        macro_alpha = 0.5,
        micro_alpha = 0.2,
        tournament_num = 10,
        tournament_times = 1000
    ):
        self.start_point = start_point
        self.item_num = len(DMat[0])
        self.left_parts = [i for i in range(self.item_num)]
        self.left_parts.remove(self.start_point)
        self.DMat = DMat
        self.seed_num = seed_num
        self.offspring_num = self.seed_num
        self.macro_alpha = macro_alpha # Reverse
        self.micro_alpha = micro_alpha #Reverses
        self.numIters = 1000
        self.tournament_times = tournament_times
        self.tournament_num = tournament_num

    def InitializeProcess(self):
        population = []
        for idx in range(self.seed_num):
            temp_list = self.left_parts[:]
            temp_occu = []
            while len(temp_list) > 0:
                t_num = random.choice(temp_list)
                temp_occu.append(t_num)
                temp_list.remove(t_num)
            population.append(temp_occu)
        return population

    def Fitness(self, population, record_values_dict):
        record_lists = []
        for individual in population:
            key_ = ' '.join([str(i) for i in [self.start_point] + individual[:] + [self.start_point]])
            if key_ in list(record_values_dict.keys()):
                value = record_values_dict[key_]
            else:
                value = self.DMat[self.start_point][individual[0]]
                for idx, item in enumerate(individual[:-1]):
                    value += self.DMat[individual[idx]][individual[idx + 1]]
                value += self.DMat[individual[-1]][self.start_point]
                # print (key_,value)
                record_values_dict[key_] = value
            record_lists.append([individual, 1 / value])
        # print (record_lists)
        sum_value = sum([i[1] for i in record_lists])
        roulette_wheel_list = []
        cumm_prob = 0
        for item in record_lists:
            cumm_prob_0 = cumm_prob
            prob_ = item[1] / sum_value
            cumm_prob += prob_
            roulette_wheel_list.append([item[0], [cumm_prob_0, cumm_prob]])
        return roulette_wheel_list, record_values_dict

    ### selecting parents
    def RouletteCrossOver(self, roulette_wheel_list):
        offsprings = []
        for turn in range(self.offspring_num):
            select_items = []
            for double in range(2):
                decision_prob = random.uniform(0, 1)
                for item in roulette_wheel_list:
                    if decision_prob >= item[1][0] and decision_prob < item[1][1]:
                        select_items.append(item[0])
            decision_orders = random.sample(range(0, self.item_num), 2)
            decision_orders.sort()
            while (decision_orders[0] == decision_orders[1]) or (
                    decision_orders[0] <= 1 and decision_orders[1] >= self.item_num - 2):
                decision_orders = random.sample(range(0, self.item_num), 2)
                decision_orders.sort()
            new_items = []
            # print ('parents: ',select_items,'\n','decision_orders:',decision_orders)
            for item_idx, new_item in enumerate(select_items):
                stable_part = new_item[decision_orders[0]:decision_orders[1] + 1]
                original_length = len(stable_part)
                # non_stable_part = new_item[:decision_orders[0]][:]+ new_item[decision_orders[1]+1:][:]
                pointer = decision_orders[1] + 1
                if pointer <= len(self.left_parts[:]) - 1 or len(stable_part) == original_length:
                    acco_index = select_items[1 - item_idx].index(stable_part[-1])
                else:
                    acco_index = select_items[1 - item_idx].index(stable_part[0])
                while len(stable_part) != len(self.left_parts[:]):
                    # time.sleep(1)
                    # print ('child:',stable_part,)
                    # print ('parent1:',pointer,new_item)
                    # print ('parent2:',select_items[1-item_idx],acco_index)
                    if acco_index == len(new_item) - 1:
                        acco_index_next = 0
                    else:
                        acco_index_next = acco_index + 1
                    if select_items[1 - item_idx][acco_index_next] in stable_part:
                        acco_index = acco_index_next
                        continue
                    else:
                        if pointer <= len(self.left_parts[:]) - 1:
                            stable_part = stable_part[:] + [select_items[1 - item_idx][acco_index_next]]
                            pointer += 1
                        else:
                            stable_part = [select_items[1 - item_idx][acco_index_next]] + stable_part[:]
                new_prex = stable_part[:decision_orders[0]]
                new_prex.reverse()
                stable_part = new_prex[:] + stable_part[decision_orders[0]:]
                # print ('final child:',stable_part)
                new_items.append(stable_part)
                if len(set(new_items[item_idx])) != len(new_items[item_idx]):
                    print(new_items[item_idx])
                    print('wrong in path!')
                    exit()
            # print ('children: ',new_items)
            offsprings.extend(new_items)
        return offsprings

    def KTourCrossOver(self, population, record_values_dict):
        # select parents
        parents = [[None, None] for i in range(self.tournament_times)]
        for i in range(self.tournament_times):
            for j in range(2):
                parents[i][j] = self.KTour(population, record_values_dict)

        # create offspring
        offsprings = [[None for j in range(self.item_num)] for i in range(self.offspring_num)]
        rng = np.random.default_rng(3876)
        for i in range(0, self.offspring_num, 2):
            parent1, parent2 = parents[i][:]
            cross1 = rng.integers(0, self.item_num - 2)
            cross2 = rng.integers(cross1 + 1, self.item_num - 1)

            offspring = [None for s in range(self.item_num - 1)]
            offspring[cross1:cross2] = parent1[cross1:cross2]
            idxStart = np.where(parent2 == parent1[cross2 - 1])[0][0] + 1 # parent2.index(parent1[cross2 - 1]) + 1
            idx = idxStart
            pos = cross2
            for idx in range(0, self.item_num):
                idxN = (idx + idxStart) % (self.item_num - 1)
                if idxN not in range(cross1, cross2):
                    if parent2[idxN] not in offspring:
                        offspring[pos] = parent2[idxN]
                        pos = (pos + 1) % (self.item_num - 1)
            for idxN in range(cross1, cross2 + 1):
                if parent2[idxN] not in offspring:
                    offspring[pos] = parent2[idxN]
                    pos = (pos + 1) % (self.item_num - 1)
            offsprings[i][:] = offspring

            offspring = [None for s in range(self.item_num - 1)]
            offspring[cross1:cross2] = parent2[cross1:cross2]
            idxStart = np.where(parent1 == parent2[cross2 - 1])[0][0] + 1
            idx = idxStart
            pos = cross2
            for idx in range(0, self.item_num):
                idxN = (idx + idxStart) % (self.item_num - 1)
                if idxN not in range(cross1, cross2):
                    if parent1[idxN] not in offspring:
                        offspring[pos] = parent1[idxN]
                        pos = (pos + 1) % (self.item_num - 1)
            for idxN in range(cross1, cross2 + 1):
                if parent1[idxN] not in offspring:
                    offspring[pos] = parent1[idxN]
                    pos = (pos + 1) % (self.item_num - 1)
            offsprings[i+1][:] = offspring
        return offsprings

    def KTour(self, population, record_values_dict):
        k = self.tournament_num
        rng = np.random.default_rng()
        choiceIndividuals = rng.choice(population, size= k)
        fittestIndividual = 0
        fittestValue = 0
        for individual in choiceIndividuals:
            key_ = ' '.join([str(i) for i in [self.start_point] + individual[:] + [self.start_point]])
            if key_ in record_values_dict.keys():
                fitness = record_values_dict[key_]
            else:
                fitness = self.DMat[self.start_point][individual[0]]
                for idx, item in enumerate(individual[:-1]):
                    fitness += self.DMat[individual[idx]][individual[idx + 1]]
                fitness += self.DMat[individual[-1]][self.start_point]
                record_values_dict[key_] = fitness
            if fitness > fittestValue:
                fittestValue = fitness
                fittestIndividual = individual
        return fittestIndividual

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

    def Mutation(self, population):
        new_population = []
        # For each individual, select if they will be in the new population
        # with switched direction
        # Probability of macro_alpha
        for individual in population:
            decision_prob = random.uniform(0, 1)
            if decision_prob >= self.macro_alpha:
                continue
            new_population.append(individual[::-1])
        # For each individual, each element has a chance of being swapped 
        for individual in population:
            new_individual = individual[:]
            # Each element in the individual has a micro_alpha chance of being swapped
            for sub_ in new_individual:
                decision_prob = random.uniform(0, 1)
                if decision_prob >= self.micro_alpha:
                    continue
                i = new_individual.index(sub_)
                j = i
                # Choose a different index than i
                while j == i:
                    j = new_individual.index(random.choice(new_individual))
                # Swap the indexes
                new_individual[i], new_individual[j] = new_individual[j], new_individual[i]
            if new_individual == individual:
                continue
            new_population.append(new_individual)
        new_population.extend(population)
        return new_population

    ### cost; the selection method

    def Elimination(self, population, record_values_dict):
        selected_population = []
        competing_individuals = random.sample(population,
                                              min(self.tournament_num * self.tournament_times, len(population)))
        total_record_list = []
        for time in range(self.tournament_times):
            record_lists = []
            competing_group = competing_individuals[
                              time * self.tournament_num:time * self.tournament_num + self.tournament_num]
            if len(competing_group) == 0:
                # print (self.tournament_num*self.tournament_times,len(competing_individuals))
                break
            for individual in competing_group:
                key_ = ' '.join([str(i) for i in [self.start_point] + individual[:] + [self.start_point]])
                if key_ in record_values_dict.keys():
                    value = record_values_dict[key_]
                else:
                    value = self.DMat[self.start_point][individual[0]]
                    for idx, item in enumerate(individual[:-1]):
                        value += self.DMat[individual[idx]][individual[idx + 1]]
                    value += self.DMat[individual[-1]][self.start_point]
                    record_values_dict[key_] = value
                record_lists.append([individual, value])
            record_lists.sort(key=lambda x: x[1])
            total_record_list.append(record_lists[0])
            selected_population.append(record_lists[0][0])
        # print (record_lists[:10])
        set_record_lists = []
        total_record_list.sort(key=lambda x: x[1])
        for i in total_record_list[:self.seed_num]:
            i = [self.start_point] + i[0][:] + [self.start_point], i[1]
            if i in set_record_lists:
                continue
            set_record_lists.append(i)
        #print(set_record_lists[:10])
        # print (selected_population[:10])
        return selected_population, record_values_dict, set_record_lists[0]

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
        record_values_dict = {}
        time_start = time.time()
        for iteration in range(self.numIters):
            # roulette_wheel_list, record_values_dict = self.Fitness(population, record_values_dict)
            # offsprings = self.RouletteCrossOver(roulette_wheel_list)
            offsprings = self.KTourCrossOver(population, record_values_dict)
            population = self.Mutation(population)
            population.extend(offsprings)
            population, record_values_dict, best_result = self.Elimination(population, record_values_dict)
            time_end = time.time()
            if time_end - time_start >= 5 * 60:
                #print('time cost', time_end - time_start, 's')
                #print('time is up!')
                print(best_result)
                return best_result


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
    random.seed(11051421)

    df = pd.read_csv('tour29.csv', header=None)
    DistanceMatrix = []
    for i in range(df.shape[0]):
        row = []
        for j in range(df.shape[1]):
            row.append(df.iloc[i, j])
        DistanceMatrix.append(row)
    # print (DistanceMatrix)

    # DistanceMatrix=[
    # [-1, 3, 6, 7],
    # [ 5,-1, 2, 3],
    # [ 6, 4,-1, 2],
    # [ 3, 7, 5,-1]]
    start_point = 0
    EA = SimpleGeneticAlgorithm(DistanceMatrix, start_point)
    SimpleGeneticAlgorithm.Iteration(EA)