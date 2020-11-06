import re
import numpy as np
from copy import deepcopy
import warnings
from nltk import edit_distance
from iteration_utilities import unique_everseen
import json
import string
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

max_regex_size = 20

def load_json(path):
    """load data for the algorithm in form of json data

    Args:
        path (string): path to the json file

    Returns:
        tuple: first element of the tuple is list of strings, second is list of lists of all wanted string parts
    """
    with open(path, 'r') as json_file:
        json_dict = json.load(json_file)
    X = [x['inputData'] for x in json_dict]
    y = [[x['inputData'][interval['start']:interval['end']] for interval in x['selectedSubStrings']] for x in json_dict]
    # left_expressions = 
    return X,y


class KramarzosGenetixoRegexator(object):
    """

        Genetic algorithm for regular expression evolution.

    """
    def __init__(self, X, y, population_size = 1000, set_chance = 0.05, max_regex_size = 50, max_size_increase = 0.1, mutation_chance = 0.1, only_true_matches = True):
        """initialize KramarzosGenetixoRegexator object

        Args:
            X (list): List of strings
            y ([type]): List of lists of search strings-regex matches
            population_size (int, optional): Size of genetic algorithm population. Defaults to 50.
            set_chance (float, optional): Parameter used in regex generation. Describes how often regex will have inside regexes. Defaults to 0.05.
            max_regex_size ([type], optional): Maximum regex size that each regular expression can have in the beginning of the algorithm. Maximum size can change during the evolution procedure. Defaults to max_regex_size.
            max_size_increase (float, optional): If max_size_increase > 0, max_regex_size can increase by max_size_increase * max_regex_size. Defaults to 0.1.
            mutation_chance (float, optional): Chance that regex will mutate. Defaults to 0.1.
            only_true_matches (bool, optional): If True, fitness will be scaled according to the number of matches that regex found for each string. Defaults to True.
        """
        self.X = deepcopy(X)
        self.y = deepcopy(y)
        self.max_regex_size = max_regex_size
        self.max_size_increase = max_size_increase
        self.only_true_matches = only_true_matches
        self.possible_regex_parts = ['.', '+', '?', '*', '|', '\w', '\W', 's', '\S', '\d', '\D', '^', '&']
        self.possible_chars = list(string.ascii_letters) + [str(i) for i in range(10)]
        self.population = self.generate_initial_population(population_size=population_size, set_chance=set_chance)
        self.population_size = population_size
        self.set_chance = set_chance
        self.mutation_chance = mutation_chance

    def generate_random_regex(self, set_chance, max_length):
        """ Generate random regular expression in the form of list. 
            Resul can contain othe regular expressions in form of lists. 

        Args:
            set_chance (float): Parameter that controlls how many nested regexes will be created
            max_length (int): Max regex length. Defaults to max_regex_size.

        Returns:
            List: Regular expression in the form of list
        """
        if int(max_length) <= 2:
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
            elif chance < 3*set_chance:
                new_part = np.random.choice(self.possible_chars)
                regex.append(str(new_part))         
            else:
                new_part = np.random.choice(self.possible_regex_parts)
                regex.append(str(new_part))
        return regex

    def regex_to_string(self, regex):
        """ Method changing regex in form of list, to Regular Expression in form of string

        Args:
            regex (list):regex in the form of list

        Returns:
            string: Regular Expression in form of string
        """
        string_regex = ''
        for regex_part in regex:
            if type(regex_part) == str:
                string_regex += regex_part
            elif type(regex_part) == dict:
                string_regex += '{' + regex_part['start'] + ',' + regex_part['end'] + '}'
            elif type(regex_part) == list:
                string_regex += '[' + self.regex_to_string(regex_part) + ']'
        return string_regex

    def calculate_regex_fitness(self, regex, X, y, only_true_matches = True):
        """ canlculate fitness of the regex

        Args:
            regex (list): regular expression in the form of list
            X (list): Strings to be searched for targets
            y (list): Target strings/matches
            only_true_matches (bool, optional): If True, fitness will be scaled according to the number of matches that regex found for each string. Defaults to True.

        Returns:
            float: fitness of string. Higher fitness means better regex
        """
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

            if len(y[index]) > len(matches) and only_true_matches == True:
                fitness += string_fitness * len(matches)/len(y[index])
            elif only_true_matches == True:
                fitness += string_fitness * len(y[index])/len(matches)
            else:
                fitness += string_fitness

        if len(regex) > self.max_regex_size:
            fitness *= self.max_regex_size/len(regex)
        if n == 0:
            return pow(fitness/(n+1), 2), all_matches
        else:
            return pow(fitness/(n), 2), all_matches

    def generate_initial_population(self, population_size, set_chance):
        """ Method generating random population of regexes

        Args:
            population_size (int): Size of genetic algorithm population. 
            set_chance (float): Parameter that controlls how many nested regexes will be created

        Returns:
            list: population of regexes in form of list of dicts
        """
        population = []
        for regex_index in range(population_size):
            new_regex = self.generate_random_regex(set_chance = set_chance, max_length = self.max_regex_size)
            new_regex_fitness, matches = self.calculate_regex_fitness(new_regex, self.X, self.y, only_true_matches=self.only_true_matches)
            population.append({'regex': new_regex, 'fitness': new_regex_fitness, 'matches': matches})
        return population
    
    def generate_kid_regex(self, parents, max_mutation_chance = 0.1):
        """ Method generating new regex based on parents.
            Generation 

        Args:
            parents (list): list of parent regexes
            max_mutation_chance (float, optional): maximum chance for mutating part of regex. Defaults to 0.1.

        Returns:
            list: new regex in form of list
        """

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

                if mutation_type < 0.3:
                    # delete part
                    to_be_deleted.append(regex_part_index)

                elif mutation_type < 0.6:
                    # replace part

                    if type(kid_regex[regex_part_index]) == list and abs(np.random.normal()) < 0.5:
                        kid_regex[regex_part_index] = self.generate_kid_regex(parents, max_mutation_chance/2)
                    else:
                        if abs(np.random.normal()) < 0.1:
                            start = np.random.randint(0, 10)
                            end = np.random.randint(start, 20)
                            kid_regex[regex_part_index] = {'start': str(start), 'end': str(end)}
                        
                        elif abs(np.random.normal()) < 0.4:
                            kid_regex[regex_part_index] = str(np.random.choice(self.possible_chars))
                        else:
                            kid_regex[regex_part_index] = str(np.random.choice(self.possible_regex_parts))

                elif mutation_type < 1:
                    # add new part

                    chance = abs(np.random.normal())            
                    if chance < 0.2:
                        # add [...] to the regex
                        if abs(np.random.normal()) < 0.5:
                            new_part = self.generate_kid_regex(parents, max_mutation_chance/2)
                        else:
                            new_part = self.generate_random_regex(set_chance = self.set_chance, max_length = max_regex_size)
                    elif chance < 0.4:
                        # add {start, end} to the regex
                        start = np.random.randint(0, 10)
                        end = np.random.randint(start, 20)
                        new_part = {'start': str(start), 'end': str(end)}     
                    elif chance < 0.6:
                        new_part = np.random.choice(self.possible_chars)
                        new_part = str(new_part)      
                    else:
                        new_part = np.random.choice(self.possible_regex_parts)
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

    def fit(self, n_iterations = 50, wanted_regex_quality = 1.0):
        """ find the best regular expression with the usage of genetic algorithm

        Args:
            n_iterations (int, optional): number of max akgorithm iterations. Defaults to 50.
            wanted_regex_quality (float, optional): How good our regex should be. Defaults to 1.0 - 100% good.

        Returns:
            dict: best regular expression in form of dict {'regex': regex : list, 'fitness': fitness : float, 'matches': matches of regex : list}
        """
        best_fitnesses = []
        for iteration in range(n_iterations):
            parents = [regex for regex in self.population if regex['fitness'] > 0]
            parents = [regex for regex in sorted(parents, key = lambda x: x['fitness'])]
            parents = list(unique_everseen(parents))
            if len(parents) < 1:
                print('not enough parents in population')
                population = self.generate_initial_population()
                continue
            
            elif parents[-1]['fitness'] >= wanted_regex_quality:
                print('algorithm finished in iteration {} finding regex {} with max fitness of {} matching {}'.format(iteration, self.regex_to_string(parents[-1]['regex']) , parents[-1]['fitness'], parents[-1]['matches']))
                return parents[-1]
            
            else:
                print('iteration {}, current best regex: {} of fitness {} matching {}'.format(iteration, self.regex_to_string(parents[-1]['regex']), parents[-1]['fitness'], parents[-1]['matches']))

            best_fitnesses.append(parents[-1]['fitness'])
            new_population = []   
            for new_kid_index in tqdm(range(self.population_size - len(parents[-int(self.population_size*0.1):]))):
                kid_regex = self.generate_kid_regex(parents, max_mutation_chance=self.mutation_chance*best_fitnesses.count(best_fitnesses[-1]))
                kid_regex_fitness, kid_regex_matches = self.calculate_regex_fitness(kid_regex, self.X, self.y, only_true_matches=self.only_true_matches)
                new_population.append({'regex': kid_regex, 'fitness': kid_regex_fitness, 'matches': kid_regex_matches})
            
            self.max_regex_size += int(self.max_size_increase * self.max_regex_size)
            self.population = new_population + parents[-int(self.population_size*0.1):]

        parents = [regex for regex in self.population if regex['fitness'] > 0]
        parents = [regex for regex in sorted(parents, key = lambda x: x['fitness'])]   
        print('algorithm finished in iteration {} finding regex {} with max fitness of {} matching {}'.format(n_iterations, self.regex_to_string(parents[-1]['regex']) , parents[-1]['fitness'], parents[-1]['matches']))
        return parents[-1]





X, y = load_json('regos/show_inventory_vid.json')
print(y)
regexator = KramarzosGenetixoRegexator(X, y, population_size=2000, only_true_matches = False)
regex = regexator.fit()
print([regex['fitness'], regexator.regex_to_string(regex['regex']), regex['matches']])
# print(len([[regex['fitness'], regexator.regex_to_string(regex['regex']), regex['matches']] for regex in regexator.population if regex['fitness'] > 0]))