import re
import numpy as np
from difflib import SequenceMatcher
from copy import deepcopy
import warnings
from nltk import edit_distance

warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_REGEX_SIZE = 500
possible_regex_parts = ['.', '+', '?', '*', '|', '\w', '\W', 's', '\S', '\d', '\D', '^', '$', '\A', '\z'] #+ [str(chr(i)) for i in range(32,123)] 
possible_chars = [str(chr(i)) for i in range(32,123)] 


class KramarzosGenetixoRegexator(object):

    def __init__(self, X, y, population_size = 50, set_chance = 0.05):
        self.X = deepcopy(X)
        self.y = deepcopy(y)
        self.population = self.generate_initial_population(population_size=population_size, set_chance=set_chance)

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
            elif chance < 4*set_chance:
                new_part = np.random.choice(possible_chars)
                regex.append(str(new_part))                
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
                fitness += string_fitness * len(matches)/len(y)
            else:
                fitness += string_fitness * len(y)/len(matches)
        if n > 0:
            return fitness/n, all_matches
        else:
            return 0, all_matches

    def generate_initial_population(self, population_size, set_chance):
        population = []
        for regex_index in range(population_size):
            new_regex = self.generate_random_regex(set_chance = set_chance)
            new_regex_fitness, matches = self.calculate_regex_fitness(new_regex, self.X, self.y)
            population.append({'regex': new_regex, 'fitness': new_regex_fitness, 'matches': matches})
        return population
        
regexator = KramarzosGenetixoRegexator(['yes i am', 'yes mam dude'], [['yes'], ['yes']], population_size=5000)
print([[regex['fitness'], regexator.regex_to_string(regex['regex']), regex['matches']] for regex in regexator.population if regex['fitness'] > 0])