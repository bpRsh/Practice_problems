'''
Created on Jul 10, 2017

@author: mouna
'''
import json
from pprint import pprint

# read and retrieve the content of the json file
with open("/home/mouna/Documents/Coveo challenge/city_search.json") as data_file:    
    data = json.load(data_file)

# This function takes as input a city, calculates all probabilities of next cities and returns the most likely next city
def nextCity(city):
# city_dict is a dictionary which will contain the neighboring cities (of the input city) each with its corresponding frequency 
    city_dict = dict()
    total_cities = 0
    for i in range(len(data)):
        # cities is a list of cities of an entry search data[i]
        cities = data[i]["cities"][0].split(', ')
        if city in cities:
            for c in cities:
                if c != city:
                    total_cities += 1
                    if c in city_dict:
                        city_dict[c][0] += 1 
                    else:
                        city_dict[c] = [1]

#     print total_cities
    for values in city_dict.values():
        city_prob = round(float(values[0])/total_cities,5)
        values.append(city_prob)
# print statics (probabilities of all next cities)    
#     for keys,values in city_dict.items():
#         print(keys)
#         print(values)

# now city_dict is a dictionary that contains the neighboring cities, each one has a list a value that contains the frequency and the corresponding probability
    m = max(city_dict.values())[1]
    Bool = True
    # next_city_list will contain the most likely next citi(es) of the input city 
    next_city_list = []
    for keys,values in city_dict.items():
        if values[1] == m:
            
#             if Bool == True:
#                 print '\nThe most likely next city of'+' '+city+' '+'is'+'...'+'\n'+'\n', keys
#                 Bool = False
#             else:
#                 print '\n',keys
            next_city_list.append(keys)
    return next_city_list

# This function return the most likely next city of all cities of the json file and also calculate the probability of having next city as New York NY 
def nextCityforAllCities():
# citiesDict is a dictionary that will contain for each city (a key) the list of all most likely next citi(es)
    citiesDict = dict()
    NewYorknumber = 0
    for i in range(len(data)): 
        cities = data[i]["cities"][0].split(', ')
        for city in cities:
            if not (city in citiesDict):
                nextC = nextCity(city)
                citiesDict[city] = nextC
                if 'New York NY' in nextC:
                    NewYorknumber += 1
    stat = float(NewYorknumber)/len(citiesDict)          
    return citiesDict, stat

# This function will print all the searches that has a country as an empty field                
def emptyCountry():
    # l will contain all the searches that has a country as an empty field
    l = []
    tot = 0
    for i in range(len(data)):
        if data[i]["user"][0][0]["country"] == "":
            tot += 1
            l.append(data[i])
            
    pprint(l)
    print '\n',tot

# This function takes as a parameter a field (it can be country or joining_date) and predict the most likely city to be searched 
def likelyCity(field):
# countryDict is a dictionary of dictionary containing for each logged country (for example) the list of cities searched for and their frequencies
    countryDict = dict()
    for i in range(len(data)):
        f = data[i]["user"][0][0][field]
        if not (f in countryDict):
            countryDict[f] = dict()
        cities = data[i]["cities"][0].split(', ')
        for city in cities:
            if not(city in countryDict[f]):
                countryDict[f][city] = 1
            else:
                countryDict[f][city] += 1
#     pprint(countryDict)
    l = dict()
    for c in countryDict:
        m = max(countryDict[c].values())
        for keys, values in countryDict[c].items():
            if values == m:
                l[c] = keys
    pprint(l)
            


if __name__ == '__main__':
    
 #   print nextCity('Newark NJ')
    #pprint(nextCityforAllCities())
    
  #  emptyCountry()
    
    likelyCity("country")
    
    
