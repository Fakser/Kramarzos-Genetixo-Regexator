import re
import numpy as np
from copy import deepcopy
import warnings
from nltk import edit_distance
from iteration_utilities import unique_everseen
import json

warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_REGEX_SIZE = 50
possible_regex_parts = ['.', '+', '?', '*', '|', '\w', '\W', 's', '\S', '\d', '\D', '^', '$', '\A', '\z'] #+ [str(chr(i)) for i in range(32,123)] 
#possible_chars = [str(chr(i)) for i in range(32,123)] 

def load_json(path):
    with open(path, 'r') as json_file:
        json_dict = json.load(json_file)
    X = [x['inputData'] for x in json_dict]
    y = [[x['inputData'][interval['start']:interval['end']] for interval in x['selectedSubStrings']] for x in json_dict]
    return X,y


class KramarzosGenetixoRegexator(object):

    def __init__(self, X, y, population_size = 50, set_chance = 0.05):
        self.X = deepcopy(X)
        self.y = deepcopy(y)
        self.population = self.generate_initial_population(population_size=population_size, set_chance=set_chance)
        self.population_size = population_size
        self.set_chance = set_chance

    def generate_random_regex(self, set_chance, max_length = MAX_REGEX_SIZE):
        if int(max_length) < 2:
            return ''
        size = np.random.randint(2, int(max_length))
        regex = []
        for part_index in range(size):
            chance = abs(np.random.normal())
            
            if chance < 1*set_chance:
                # add [...] to the regex
                regex.append(self.generate_random_regex(set_chance = set_chance/2, max_length = max_length*0.1))
            elif chance < 2*set_chance:
                # add {start, end} to the regex
                start = np.random.randint(0, 10)
                end = np.random.randint(start, 20)
                regex.append({'start': str(start), 'end': str(end)})
            # elif chance < 4*set_chance:
            #     new_part = np.random.choice(possible_chars)
            #     regex.append(str(new_part))                
            else:
                new_part = np.random.choice(possible_regex_parts)
                regex.append(str(new_part))
        return regex

    def regex_to_string(self, regex):
        string_regex = ''
        for regex_part in regex:
            if type(regex_part) == str:
                string_regex += regex_part
            elif type(regex_part) == dict:
                string_regex += '{' + regex_part['start'] + ',' + regex_part['end'] + '}'
            elif type(regex_part) == list:
                string_regex += '[' + self.regex_to_string(regex_part) + ']'
        return string_regex

    def calculate_regex_fitness(self, regex, X, y):
        regular_expression = self.regex_to_string(regex)
        n = 0
        fitness = 0
        all_matches = []
        for index, string in enumerate(X):
            string_fitness = 0
            try:
                matches = re.findall(regular_expression, string)
                all_matches.append(deepcopy(matches))
            except:
                continue
            matches_copied = deepcopy(matches)
            for target in y[index]:
                n += 1
                if len(matches_copied) >= 1:
                    ratios = [[match, 1/(1 + edit_distance(target, match))] for match in matches_copied]
                    ratios_sorted = [i for i in sorted(ratios, key = lambda x: x[1])]
                    matches_copied.remove(ratios_sorted[-1][0])
                    string_fitness += ratios_sorted[-1][1]

            if len(y[index]) > len(matches):
                fitness += string_fitness * len(matches)/len(y[index])
            else:
                fitness += string_fitness * len(y[index])/len(matches)

        if n == 1:
            return fitness/(n), all_matches
        else:
            return fitness/(n+1), all_matches

    def generate_initial_population(self, population_size, set_chance):
        population = []
        for regex_index in range(population_size):
            new_regex = self.generate_random_regex(set_chance = set_chance)
            new_regex_fitness, matches = self.calculate_regex_fitness(new_regex, self.X, self.y)
            population.append({'regex': new_regex, 'fitness': new_regex_fitness, 'matches': matches})
        return population
    
    def generate_kid_regex(self, parents, max_mutation_chance = 0.1):

        # crossover
        parent_chance = abs(np.random.normal())
        tournament_population = [regex for regex in sorted(np.random.choice(self.population, size = 10), key = lambda x: x['fitness'])]
        tournament_parents = [regex for regex in sorted(np.random.choice(parents, size = 10), key = lambda x: x['fitness'])]
        if parent_chance < 0.2:
            parent1 = tournament_population[-1]
            parent2 = tournament_population[-2]

        if parent_chance < 0.4 or len(parents) < 2:
            parent1 = tournament_parents[-1]
            parent2 = tournament_population[-1]
        
        if parent_chance < 0.6:
            parent1 = tournament_population[-1]
            parent2 = tournament_parents[-1]
        
        else:
            parent1 = tournament_parents[-1]
            parent2 = tournament_parents[-2]
        
        if len(parent1['regex']) > 2:
            split_parent_1 = np.random.randint(1, len(parent1['regex']) - 1)
        else:
            split_parent_1 = len(parent1['regex']) - 1

        if len(parent2['regex']) > 2:
            split_parent_2 = np.random.randint(1, len(parent2['regex']) - 1)
        else:
            split_parent_2 = len(parent2['regex']) - 1

        kid_regex = deepcopy(parent1['regex'][:split_parent_1]) + deepcopy(parent2['regex'][split_parent_2:])

        # mutations
        to_be_deleted = []
        for regex_part_index in range(len(kid_regex)):
            regex_part_mutation_chance = abs(np.random.normal())        
            if regex_part_mutation_chance < max_mutation_chance:
                mutation_type = abs(np.random.normal())

                if mutation_type < 0.05:
                    # delete part
                    to_be_deleted.append(regex_part_index)

                elif mutation_type < 0.5:
                    # replace part

                    if type(kid_regex[regex_part_index]) == list and abs(np.random.normal()) < 0.5:
                        kid_regex[regex_part_index] = self.generate_kid_regex(parents, max_mutation_chance*max_mutation_chance)
                    else:
                        if abs(np.random.normal()) < 0.1:
                            start = np.random.randint(0, 10)
                            end = np.random.randint(start, 20)
                            kid_regex[regex_part_index] = {'start': str(start), 'end': str(end)}
                        
                        # elif abs(np.random.normal()) < 0.4:
                        #     kid_regex[regex_part_index] = str(np.random.choice(possible_chars))
                        else:
                            kid_regex[regex_part_index] = str(np.random.choice(possible_regex_parts))

                elif mutation_type < 1:
                    # add new part

                    chance = abs(np.random.normal())            
                    if chance < 1*max_mutation_chance:
                        # add [...] to the regex
                        if abs(np.random.normal()) < 0.5:
                            new_part = self.generate_kid_regex(parents, max_mutation_chance*max_mutation_chance)
                        else:
                            new_part = self.generate_random_regex(set_chance = self.set_chance, max_length = MAX_REGEX_SIZE)
                    elif chance < 2*max_mutation_chance:
                        # add {start, end} to the regex
                        start = np.random.randint(0, 10)
                        end = np.random.randint(start, 20)
                        new_part = {'start': str(start), 'end': str(end)}
                    # elif chance < 4*max_mutation_chance:
                    #     new_part = np.random.choice(possi
                    # ble_chars)
                    #     new_part = str(new_part)                
                    else:
                        new_part = np.random.choice(possible_regex_parts)
                        new_part = str(new_part)
                    
                    if regex_part_index == len(kid_regex) - 1:
                        kid_regex.append(new_part)
                    else:
                        kid_regex = kid_regex[:regex_part_index] + [new_part] + kid_regex[regex_part_index:]

        for i, regex_part_index in enumerate(to_be_deleted):
            if regex_part_index - i == len(kid_regex) - 1:
                kid_regex = kid_regex[:-1]
            else:
                kid_regex = kid_regex[:regex_part_index - i] + kid_regex[regex_part_index - i + 1:]
        return kid_regex

    def fit(self, n_iterations = 50, wanted_regex_quality = 0.9):
        for iteration in range(n_iterations):
            parents = [regex for regex in self.population if regex['fitness'] > 0]
            parents = [regex for regex in sorted(parents, key = lambda x: x['fitness'])]
            parents = list(unique_everseen(parents))
            # parents = parents[-10:]
            if len(parents) < 1:
                print('not enough parents in population')
                population = self.generate_initial_population()
                continue
            
            elif parents[-1]['fitness'] >= wanted_regex_quality:
                print('algorithm finished in iteration {} finding regex {} with max fitness of {} matching {}'.format(iteration, self.regex_to_string(parents[-1]['regex']) , parents[-1]['fitness'], parents[-1]['matches']))
                return parents[-1]
            
            else:
                print('iteration {}, current best regex: {} of fitness {} matching {}'.format(iteration, self.regex_to_string(parents[-1]['regex']), parents[-1]['fitness'], parents[-1]['matches']))
            new_population = []   
            for new_kid_index in range(self.population_size - len(parents[-int(self.population_size*0.1):])):
                kid_regex = self.generate_kid_regex(parents)
                kid_regex_fitness, kid_regex_matches = self.calculate_regex_fitness(kid_regex, self.X, self.y)
                new_population.append({'regex': kid_regex, 'fitness': kid_regex_fitness, 'matches': kid_regex_matches})
            
            self.population = new_population + parents[-int(self.population_size*0.1):]

        parents = [regex for regex in self.population if regex['fitness'] > 0]
        parents = [regex for regex in sorted(parents, key = lambda x: x['fitness'])]   
        print('algorithm finished in iteration {} finding regex {} with max fitness of {} matching {}'.format(n_iterations, self.regex_to_string(parents[-1]['regex']) , parents[-1]['fitness'], parents[-1]['matches']))
        return parents[-1]





X, y = load_json('regos/regos1.json')
print( y)
regexator = KramarzosGenetixoRegexator(X, y, population_size=10000)
regex = regexator.fit()
print([regex['fitness'], regexator.regex_to_string(regex['regex']), regex['matches']])
# print(len([[regex['fitness'], regexator.regex_to_string(regex['regex']), regex['matches']] for regex in regexator.population if regex['fitness'] > 0]))