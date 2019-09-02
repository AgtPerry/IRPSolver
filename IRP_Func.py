import csv
import math
import numpy
import os
import time
import re
import string
import gc
import numpy as np
from geopy.distance import great_circle as gc
from subprocess import run
import subprocess
import matplotlib.pyplot as plt

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(FILE_DIR)
#os.chdir('D:/CS/workspace/DissertationCode')

start_time = time.time()

DATA_FILE = 'Sample300B.CSV'
TIME_DELAY = 1
SCANNING_INTERVAL = 0.2

#==================
# LS_re: local search reallocation
# LS_sw: local search swap
# set to True to use them, otherwise the method is not used
LS_re = True
LS_sw = True

#==================
draw_route = True
#==================


FRIDGE_CAP = 60 
VEHICLE_CAP = 3000
VehicleUsage = 0.6
VEHICLE_ACAP = VEHICLE_CAP * VehicleUsage
PERIOD = 6

# define a Class for customers and depot
class Customer:
    def __init__(self, no, lat, lon, demand):
        self.no = int(no)
        self.lat = float(lat)
        self.lon = float(lon)
        self.demand = float(demand)
        
        self.schedule = [0 for i in range(PERIOD)]

        #p for CW algorithm--
        self.p=2

        #--------------------

    def __repr__(self):
        return "".join(["\nCustomer ", str(self.no), \
                        ", Coordinate [", str(self.lat),",",str(self.lon), \
                        "], Demand ", str(self.demand)]) 
#end

#input data into list customer_list
customer_list=[]
customer_dict={}

with open(DATA_FILE,newline='') as mydata:
    reader = csv.DictReader(mydata)
    #cc = [row for row in reader] #COMMENT OUT
    for index, row in enumerate(reader):
        
        #This is Dict of Dict
        customer_dict[int(row['ACCOUNT NO'])] = \
            {'Latitude':    float(row['LATITUDE']), \
             'Longitude':   float(row['LONGITUDE']), \
             'Demand':      float(row['DEMAND'])}
#end

input_break = time.time()

#================================================
#1 Cluster
#================================================
#Sort by Demand in Descend

#This List is not changed after this
large_demand_list=[]
for i in customer_dict:
    while customer_dict[i]['Demand'] > VEHICLE_ACAP*PERIOD:
        customer_dict[i]['Demand'] -= VEHICLE_ACAP*PERIOD
        large_demand_list.append(i)


#making a dict to record the status of customer
#to support CW and other algorithms
def create_status_dict():
    new_dict = {}
    for i in customer_dict:
        new_dict[i] = {'assigned': False, 'cluster':-1}
    #new_dict[0]['assigned'] = True
    new_dict.pop(0)
    return new_dict

#------------------------------------
#Clarke Wright Algorithm
#Scheduling of vehicles from a central depot to a number of delivery points
#1964

# merge two customers and their clusters into one cluster
def CW_merge(accA, accB):   #merge B to A, with condition check
    clusA = customer_status[accA]['cluster']
    clusB = customer_status[accB]['cluster']
    # the 3 conditions are refered to CW algorithm P573
    # condition(II)
    if clusA == clusB:
        return False
    
    # condition(I)
    if not ((cluster_dict[clusA]['Customer'][0] == accA or \
             cluster_dict[clusA]['Customer'][-1] == accA) and \
            (cluster_dict[clusB]['Customer'][0] == accB or \
             cluster_dict[clusB]['Customer'][-1] == accB)):
        #print ('Error! Account A or B is in the middle!')
        return False
    
    # condition(III)
    # check load capacity
    if cluster_dict[clusA]['Load'] + cluster_dict[clusB]['Load'] \
       > VEHICLE_ACAP*PERIOD:
        return False
    
    # make two lists: [---A],[B---]
    if cluster_dict[clusA]['Customer'][0] == accA:
        cluster_dict[clusA]['Customer'].reverse()
    if cluster_dict[clusB]['Customer'][-1] == accB:
        cluster_dict[clusB]['Customer'].reverse()

    cluster_dict[clusA]['Load'] += cluster_dict[clusB]['Load']
    cluster_dict[clusA]['Load'] = round(cluster_dict[clusA]['Load'],2)
    cluster_dict[clusA]['Customer'].extend(cluster_dict[clusB]['Customer'])

    # update cluster info for all account in cluster B
    for accNo in cluster_dict[clusB]['Customer']:
        customer_status[accNo]['cluster'] = clusA

    del cluster_dict[clusB]
    #cluster_dict.pop(clusB)
    return True

#------------------------------------
#Clustering Algorithm. Assign customer to Cluster, a simple bad way
#Clarke Wright algorithm
def distance(accA, accB):   #accA = account No A
    dis = round(gc((customer_dict[accA]['Latitude'],customer_dict[accA]['Longitude']),
              (customer_dict[accB]['Latitude'],customer_dict[accB]['Longitude'])).meters,2)
    return dis

def saving(accA,accB):
    sav = round(distance(accA,0) + distance(accB,0) - distance(accA,accB),2)
    return sav

# make a saving list
# in CW, it is a half matrix

def create_saving_list():
    n = len(customer_dict)-1
    new = np.zeros((n*(n-1)//2, 3))
    row = 0
    for i in customer_dict:
        for j in customer_dict:
            if j>i and i!=0 and j!=0:
                new[row] = [int(saving(i,j)),i,j]
                row+=1
    #new_list[i][0] is saving
    #new_list[i][1] and [i][2] are two customer accounts   
    return new

sav_list = create_saving_list()
#sorted by the first value (saving between 2 customer)
sav_list = sorted(sav_list, key=(lambda x:x[0]), reverse = True)

#customer_status[account_no]['assigned','t','cluster','']
# initialize cluster info in customer_status
customer_status = create_status_dict()
cluster_dict = {}
def CW_initial():
    counter = 1
    for i in customer_status:
        customer_status[i]['cluster'] = counter
        cluster_dict[counter] = {'Load': customer_dict[i]['Demand'],
                                 'Customer': [i]}
        counter += 1

CW_initial()

def check_cluster():
    for i in customer_status:
        if cluster_dict[customer_status[i]['cluster']]['Customer'][0] != i:
            print('Cluster info in two lists are not match!')
            return
check_cluster()

# iteration until there is no more new link
for i in range(len(sav_list)):
    CW_merge(sav_list[i][1],sav_list[i][2])
    
# rename the keys of cluster list-------
def rename_cluster(cluster_dict):
    keys = list(range(1,len(cluster_dict)+1))
    values = cluster_dict.values()
    cluster_dict = dict(zip(keys, values))
    for i in cluster_dict:
        cluster_dict[i]['Load'] = round(cluster_dict[i]['Load'],2)
    
    return cluster_dict
    
cluster_dict = rename_cluster(cluster_dict)
'''
counter = 1
for i in cluster_dict:
    if i != counter:
        cluster_dict[counter] = cluster_dict.pop(i)
    counter += 1
    '''
#---------------------------------------   

print('cluster: ',cluster_dict)

#----------
del sav_list
#----------

cluster_break = time.time()

#==============================================
#2 Scheduling
#==============================================
#Scheduling List

#List of Scheduling?????

# numpy approach -------------

# scheduling_dict[clusterNo][index][0] is AccNo
# scheduling_dict[clusterNo][index][1-6] is delivery amount every day

# ----------------------------

# Scheduling for one cluster ------
# North west as simple constructive method
def scheduling_iteration(date, index ,remain_cap ,s_matrix):

    # stop until last acc demand is met
    while s_matrix[0][-1] > 0:

        # if a vehicle still has cap to deliver
        if s_matrix[0][index] <= remain_cap:
            s_matrix[date][index] += s_matrix[0][index]
            remain_cap -= s_matrix[0][index]
            s_matrix[0][index] = 0
            scheduling_iteration(date, index+1 ,remain_cap ,s_matrix)

        # if a vehicle has no cap to deliver 
        elif s_matrix[0][index] > remain_cap:
            s_matrix[date][index] = remain_cap
            s_matrix[0][index] -= remain_cap
            scheduling_iteration(date+1, index ,VEHICLE_ACAP ,s_matrix)
            
def scheduling(cluster):

    #Scheduling Matrix Initialize
    s_matrix = np.zeros((PERIOD + 1, len(cluster_dict[cluster]['Customer'])))
    for i in range(s_matrix.shape[1]):
        s_matrix[0][i] = customer_dict[cluster_dict[cluster]['Customer'][i]]['Demand']
    #first row for delivery left
    # s_matrix[date][index of AccNo]
    
    if sum(s_matrix[0]) > VEHICLE_ACAP*PERIOD:
        print(s_matrix[0])
        print('Error in Cluster: Too much load or too small vehicle capacity')
        return 0

    scheduling_iteration(1, 0 , VEHICLE_ACAP, s_matrix)

    s_matrix = np.delete(s_matrix, 0, axis=0)
    return s_matrix

scheduling_dict = {}
for i in cluster_dict:
    scheduling_dict[i] = scheduling(i)
    
print('scheduling_dict =',scheduling_dict)

scheduling_break = time.time()

#==========================================
#3 Routing
#==========================================

#fleet_list[Date][Cluster][indexAccNo]=AccNo

# counter for LKH-2.exe
solver_counter = 0

#rebuild a fleet list. must have a better idea here, maybe use Mask Array
fleet_dict = {i: [[0] for j in range(PERIOD)]
              for i in scheduling_dict}

#write fleet dict
def input_fleet(fleet_dict):
    for i in scheduling_dict:
        for j in range(PERIOD):
            k = 0
            while k < len(scheduling_dict[i][j]):
                if scheduling_dict[i][j][k] > 0:
                    fleet_dict[i][j].append(cluster_dict[i]['Customer'][k])

                k+=1

input_fleet(fleet_dict)

#fleet_dict[cluster][date,0 ~ period-1][k]

print('fleet list =',fleet_dict)
#print(fleet_list[1][1])
def distance_m(accA, accB):
    d = round(gc((customer_dict[accA]['Latitude'],
                  customer_dict[accA]['Longitude']),
                 (customer_dict[accB]['Latitude'],
                  customer_dict[accB]['Longitude'])).meters,2)
    return d

def distance_matrix(acc_list,customer_dict):
    mx = [[round(gc((customer_dict[i]['Latitude'],customer_dict[i]['Longitude']),
              (customer_dict[j]['Latitude'],customer_dict[j]['Longitude'])).meters,2)
           for i in acc_list] for j in acc_list]
    return mx



'''solve a TSP using LKH solver ==================


'''
fname_tsp = 'tmp'
user_comment = 'a comment by user'

lkh_cmd = 'LKH-2.exe'
pwd = os.path.dirname(os.path.abspath(__file__))+'\\'
lkh_dir = pwd
tsplib_dir = pwd

#write *.tsp and *. par files for LKH solver
def write_TSP_file(fname_tsp, a_distance_matrix, user_comment):
    dims_tsp = len(a_distance_matrix)
    name_line = 'NAME : ' + fname_tsp + '\n'
    type_line = 'TYPE: TSP' + '\n'
    comment_line = 'COMMENT : ' + user_comment + '\n'
    tsp_line = 'TYPE : ' + 'TSP' + '\n'
    dimension_line = 'DIMENSION : ' + str(dims_tsp) + '\n'
    edge_weight_type_line = 'EDGE_WEIGHT_TYPE : ' + 'EXPLICIT' + '\n'
    edge_weight_format_line = 'EDGE_WEIGHT_FORMAT: ' + 'FULL_MATRIX' + '\n'
    display_data_type_line ='DISPLAY_DATA_TYPE: ' + 'NO_DISPLAY' + '\n'
    edge_weight_section_line = 'EDGE_WEIGHT_SECTION' + '\n'
    eof_line = 'EOF\n'

    Cost_Matrix_STRline = []
    for i in range(0,dims_tsp):
        cost_matrix_strline = ''
        for j in range(0,dims_tsp-1):
            cost_matrix_strline = cost_matrix_strline + str(int(a_distance_matrix[i][j])) + '\t'
            
        j = dims_tsp-1
        cost_matrix_strline = cost_matrix_strline + str(int(a_distance_matrix[i][j]))
        cost_matrix_strline = cost_matrix_strline + '\n'
        Cost_Matrix_STRline.append(cost_matrix_strline)

    #write tsp file        
    file_tsp = open((tsplib_dir + fname_tsp + '.tsp'), "w")
    #print (name_line)
    file_tsp.write(name_line)
    file_tsp.write(comment_line)
    file_tsp.write(tsp_line)
    file_tsp.write(dimension_line)
    file_tsp.write(edge_weight_type_line)
    file_tsp.write(edge_weight_format_line)
    file_tsp.write(edge_weight_section_line)

    for i in range(0,len(Cost_Matrix_STRline)):
        file_tsp.write(Cost_Matrix_STRline[i])

    file_tsp.write(eof_line)
    file_tsp.close()

    #write par file
    file_par = open((tsplib_dir + fname_tsp + '.par'), "w")

    problem_file_line = 'PROBLEM_FILE = ' + fname_tsp + '.tsp' + '\n'
    optimum_line = 'OPTIMUM = 99999999' + '\n'
    move_type_line = 'MOVE_TYPE = 5' + '\n'
    patching_c_line = 'PATCHING_C = 3' + '\n'
    patching_a_line = 'PATCHING_A = 2' + '\n'
    runs_line = 'RUNS = 3' + '\n'
    tour_file_line = 'OUTPUT_TOUR_FILE = ' + fname_tsp + '.txt' + '\n'

    file_par.write(problem_file_line)
    file_par.write(optimum_line)
    file_par.write(move_type_line)
    file_par.write(patching_c_line)
    file_par.write(patching_a_line)
    file_par.write(runs_line)
    file_par.write(tour_file_line)
    file_par.close()

    return file_tsp, file_par

def run_LKH_solver(fname_basis):

    #subprocess.Popen([lkh_cmd,(fname_basis+'.par')])
    proc = subprocess.Popen([lkh_cmd,(fname_basis+'.par')])

    fname_txt = fname_basis + '.txt'

    while os.path.exists(fname_txt) == False:
        time.sleep(SCANNING_INTERVAL)

    pattern = 'EOF'
    found = False
    with open(fname_txt,'r') as f:
        for line in f.readlines():
            if pattern in line:
                found = True
                break
    if found == False:
        time.sleep(TIME_DELAY)

# close LKH-2.exe, does not work 
    #os.system('taskkill /f /im LKH-2.exe')
    
def if_exist_del(file, filetype):
    delfile = file + filetype
    if os.path.isfile(delfile):
        os.remove(delfile) 

def TSP_solver(distance_matrix,fname,cluster,date):
#solve problem
    global solver_counter
    
    if_exist_del(fname, '.par')
    if_exist_del(fname, '.tsp')
    if_exist_del(fname, '.txt')
    
    fname_txt = fname+'.txt'
    [file_tsp, file_par] = write_TSP_file(fname,distance_matrix,user_comment)
    run_LKH_solver(fname)

    solver_counter +=1

    #os.system('taskkill /f /im LKH-2.exe')
    #fname_txt = fname_tsp + str(index) +'.txt'
#get tsp result
    tsp_result = []

    #time.sleep(TIME_DELAY)
    
    with open(fname_txt, 'r') as f:
        pattern = r'TOUR_SECTION'
        line = f.readline()
        cost = float(line.split('.')[1])
        #how to return cost to outside???
        while re.match(pattern,line)==None:
            line = f.readline()

        pattern = r'-1'
        line = f.readline()

        while re.match(pattern,line)==None:
            tsp_result.append(fleet_dict[cluster][date][int(line)-1])
            line = f.readline()
            
        tsp_result.append(0)

    return tsp_result
            
        
'''================================================'''

def routing(date, cluster):
    #fname = 'tmp_tsp'
    fname = 'tsp_C' + str(cluster) + '_D' + str(date)
    problem = distance_matrix(fleet_dict[cluster][date], customer_dict)

#----------------

    result = TSP_solver(problem, fname, cluster, date)
    #time.sleep(TIME_DELAY)

    return result
#---------------------------------
    #time.sleep(TIME_DELAY)


# initialize a routing dict

route_dict={i:[{'Cost':0, 'Route':[]} for j in range(PERIOD)]
             for i in cluster_dict}

def routing_single(date,cluster,a_fleet,route_c):
# route_c is route_dict[cluster]
# a_fleet is fleet_dict[cluster]
    if len(a_fleet[date]) <= 3:
        
        route_c[date]['Route'] = a_fleet[date][:]
        route_c[date]['Route'].append(0)
    else:
        route_c[date]['Route'] = routing(date,cluster)
            
    route_c[date]['Cost'] = \
                          route_cost(route_c[date]['Route'])

def route_cost(route):
    cost = 0
    for i in range(len(route)-1):
        cost += distance_m(route[i],route[i+1])
    return round(cost,2)

# make routing for all in fleet dict
for cluster in fleet_dict:
    for date in range(len(fleet_dict[cluster])):

        routing_single(date,cluster,fleet_dict[cluster],
                       route_dict[cluster])
        
        #print(cluster, '\t',date)

       
print('route_dict =',route_dict)



def total_cost(route_dict):
    c = 0
    for cluster in route_dict:
        for date in range(len(route_dict[cluster])):
            c += route_dict[cluster][date]['Cost']

    return c

routing_break = time.time()

#4 Local Search  

def remain_cap_list(cluster):
    cap_remain_list = [VEHICLE_ACAP for i in range(PERIOD)]
    for i in range(len(scheduling_dict[cluster])):
    #for i in range(len(a_fleet)):
        cap_remain_list[i] -= sum(scheduling_dict[cluster][i])
    return cap_remain_list


def where_to_insert(acc, route):
# route is route_dict[cluster][day]['Route']
    old_cost = math.inf
    for i in range(len(route)-1):
        new_cost = distance_m(route[i],acc) + distance_m(acc,route[i+1]) \
                   - distance_m(route[i],route[i+1])
        if new_cost < old_cost:
            old_cost = new_cost
            index = i
    return index
        

def try_changing_vehicle_between_days(cluster, acc, dayA, dayB,route):
# route is route_dict[cluster]
    dayA_cost = route_cost(route[dayA]['Route'])
    dayB_cost = route_cost(route[dayB]['Route'])
    acc_index = cluster_dict[cluster]['Customer'].index(acc)
    
    old_cost = dayA_cost + dayB_cost
    old_index = route[dayA]['Route'].index(acc)
    saving_A = distance_m(route[dayA]['Route'][old_index-1],
                          route[dayA]['Route'][old_index]) \
             + distance_m(route[dayA]['Route'][old_index],
                          route[dayA]['Route'][old_index+1]) \
             - distance_m(route[dayA]['Route'][old_index-1],
                          route[dayA]['Route'][old_index+1])
    
    new_index = where_to_insert(acc, route[dayB]['Route'])
    
    cost_B = distance_m(route[dayB]['Route'][new_index],acc) \
             +distance_m(route[dayB]['Route'][new_index+1],acc) \
             -distance_m(route[dayB]['Route'][new_index], \
                         route[dayB]['Route'][new_index+1])
    if saving_A > cost_B:


        # change scheduling
        scheduling_dict[cluster][dayB][acc_index] += scheduling_dict[cluster][dayA][acc_index]
        scheduling_dict[cluster][dayA][acc_index] = 0

        # change fleet
        fleet_dict[cluster][dayB].append(acc)
        fleet_dict[cluster][dayA].remove(acc)

        # change route
        route[dayA]['Cost'] -= saving_A
        route[dayB]['Cost'] += cost_B
        route[dayA]['Route'].remove(acc)
        route[dayB]['Route'].insert(new_index+1, acc)
        return True
    return False

def try_swap_vehicles_between_days(cluster, accA, accB, dayA, dayB, route):
# swap account A in dayA and account B in dayB
    #print('Try This',cluster, accA, accB, dayA, dayB)
    dayA_cost = route_cost(route[dayA]['Route'])
    dayB_cost = route_cost(route[dayB]['Route'])

    # index in scheduling array
    accA_index = cluster_dict[cluster]['Customer'].index(accA)
    accB_index = cluster_dict[cluster]['Customer'].index(accB)

    # index in route
    old_indexA = route[dayA]['Route'].index(accA)
    old_indexB = route[dayB]['Route'].index(accB)
    
    saving_A_dayA = distance_m(route[dayA]['Route'][old_indexA-1],
                          route[dayA]['Route'][old_indexA]) \
             + distance_m(route[dayA]['Route'][old_indexA],
                          route[dayA]['Route'][old_indexA+1]) \
             - distance_m(route[dayA]['Route'][old_indexA-1],
                          route[dayA]['Route'][old_indexA+1])
    #print(accA,accB,dayA,dayB,old_indexB)
    saving_B_dayB = distance_m(route[dayB]['Route'][old_indexB-1],
                          route[dayB]['Route'][old_indexB]) \
             + distance_m(route[dayB]['Route'][old_indexB],
                          route[dayB]['Route'][old_indexB+1]) \
             - distance_m(route[dayB]['Route'][old_indexB-1],
                          route[dayB]['Route'][old_indexB+1])
    # remove A and B in route first
    route[dayA]['Route'].remove(accA)
    route[dayB]['Route'].remove(accB)

    # find where to insert
    new_i_A_dayB = where_to_insert(accA, route[dayB]['Route'])
    new_i_B_dayA = where_to_insert(accB, route[dayA]['Route'])
    
    cost_A_dayB = distance_m(route[dayB]['Route'][new_i_A_dayB],accA) \
             +distance_m(route[dayB]['Route'][new_i_A_dayB+1],accA) \
             -distance_m(route[dayB]['Route'][new_i_A_dayB], \
                         route[dayB]['Route'][new_i_A_dayB+1])

    cost_B_dayA = distance_m(route[dayA]['Route'][new_i_B_dayA],accB) \
             +distance_m(route[dayA]['Route'][new_i_B_dayA+1],accB) \
             -distance_m(route[dayA]['Route'][new_i_B_dayA], \
                         route[dayA]['Route'][new_i_B_dayA+1])

    if cost_A_dayB + cost_B_dayA < saving_A_dayA + saving_B_dayB:
        #print(acc,'dayA=',dayA)
        #print(cluster,accA,accB,dayA,dayB)
        # change scheduling

        # it is possible that there is accA in dayB before swap
        # how to handle this?
        scheduling_dict[cluster][dayB][accA_index]+=scheduling_dict[cluster][dayA][accA_index]
        scheduling_dict[cluster][dayA][accA_index]=0
        scheduling_dict[cluster][dayA][accB_index]+=scheduling_dict[cluster][dayB][accB_index]
        scheduling_dict[cluster][dayB][accB_index]=0
        
        # change fleet
        fleet_dict[cluster][dayA].append(accB)
        fleet_dict[cluster][dayA].remove(accA)
        fleet_dict[cluster][dayB].append(accA)
        fleet_dict[cluster][dayB].remove(accB)

        # change route
        route[dayA]['Cost'] += (cost_B_dayA - saving_A_dayA)
        route[dayB]['Cost'] += (cost_A_dayB - saving_B_dayB)
        
        route[dayA]['Route'].insert(new_i_B_dayA +1, accB)
        route[dayB]['Route'].insert(new_i_A_dayB +1, accA)
        return True
    
# add the removed acc back to route
    else:
        route[dayA]['Route'].insert(old_indexA , accA)
        route[dayB]['Route'].insert(old_indexB , accB)
        return False


            
def moving_in_scheduling(cluster):
# a_fleet is fleet_dict[cluster]
# moving one vehicle to another day
# update cap_remain_list

#changing A to B
    global counter_re
    cap_list = remain_cap_list(cluster)
    for dayA in range(PERIOD):
        for dayB in range(PERIOD):
            if cap_list[dayB] >0 and dayA != dayB:
                #route is route_dict[cluster]
                # sched is scheduling_dict[cluster]
                for indexA in range(len(cluster_dict[cluster]['Customer'])):
                    if scheduling_dict[cluster][dayA][indexA] >0:
                        if cap_list[dayB] >= scheduling_dict[cluster][dayA][indexA]:
                            acc = cluster_dict[cluster]['Customer'][indexA]
                            if try_changing_vehicle_between_days( \
                                cluster, acc, dayA, dayB, route_dict[cluster]):
# ............
                                #print('LS!!', acc)
                                counter_re += 1
                                moving_in_scheduling(cluster)
                                #cap_list = remain_cap_list(cluster)

                                      
def swap_in_scheduling(cluster):
    # A is larger than B
    global counter_sw
    cap_list = remain_cap_list(cluster)
    for dayA in range(PERIOD):
        for dayB in range(PERIOD):
            if cap_list[dayB] > cap_list[dayA] and dayA != dayB:
                for indexA in range(len(cluster_dict[cluster]['Customer'])):
                    for indexB in range(len(cluster_dict[cluster]['Customer'])):
                        if scheduling_dict[cluster][dayA][indexA] > \
                           scheduling_dict[cluster][dayB][indexB] and \
                           scheduling_dict[cluster][dayB][indexB] > 0:
                            if cap_list[dayB] >= \
                               scheduling_dict[cluster][dayA][indexA] - \
                               scheduling_dict[cluster][dayB][indexB]:
                                accA = cluster_dict[cluster]['Customer'][indexA]
                                accB = cluster_dict[cluster]['Customer'][indexB]
                                if try_swap_vehicles_between_days( \
                                    cluster, accA, accB, dayA, dayB, route_dict[cluster]):
                                     #print('LS_Swap!', accA,accB,dayA,dayB,cluster)
                                    counter_sw += 1
                                    swap_in_scheduling(cluster)

  
cost = total_cost(route_dict)
print('Before LS =', round(cost,2))

counter_re = 0
counter_sw = 0

LS_sw_start = time.time()
if LS_sw == True:
    for i in route_dict:
        swap_in_scheduling(i)
    LS_SW_cost = total_cost(route_dict)        
    print('After LS_SW =', round(LS_SW_cost,2))
        
LS_sw_end = time.time()

LS_re_start = time.time()

if LS_re == True:
    for i in route_dict:
        moving_in_scheduling(i)
    LS_RE_cost = total_cost(route_dict)        
    print('After LS_RE =', round(LS_RE_cost,2))

LS_re_end = time.time()





LS_cost = total_cost(route_dict)
print('After LS =', round(LS_cost,2))
print('Improvement = ', round((cost-LS_cost)*100/cost,4),'%')

LS_break = time.time()

#5 Visusalization


def draw_all_cluster(cluster_dict):

    x_list = [0]
    y_list = [0]

    x_list[0] = customer_dict[0]['Longitude']
    y_list[0] = customer_dict[0]['Latitude']

    for cluster in cluster_dict:
        
        x_list_tmp = [0 for i in range(len(cluster_dict[cluster]['Customer'])+1)]
        y_list_tmp = [0 for i in range(len(cluster_dict[cluster]['Customer'])+1)]

        x_list_tmp[-1] = customer_dict[0]['Longitude']
        y_list_tmp[-1] = customer_dict[0]['Latitude']

        i = 0
        for acc_no in cluster_dict[cluster]['Customer']:
            x_list_tmp[i] = customer_dict[acc_no]['Longitude']
            y_list_tmp[i] = customer_dict[acc_no]['Latitude']
            i += 1

        x_list.extend(x_list_tmp)
        y_list.extend(y_list_tmp)

    plt.figure('Line fig')
    ax = plt.gca()

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.plot(x_list, y_list, color='r', linewidth=1, alpha=0.6)

    plt.show()

def draw_a_day(route, date):

    route_list = []

    for cluster in route:
        route_list.extend(route[cluster][date]['Route'])
        route_list.pop()
        
    route_list.append(0)
    
    x_list = [None] * len(route_list)
    y_list = [None] * len(route_list)

    i=0
    for acc in route_list:
        x_list[i] = customer_dict[acc]['Longitude']
        y_list[i] = customer_dict[acc]['Latitude']
        i+=1
#===============

    all_x_list = [None] * len(customer_dict)
    all_y_list = [None] * len(customer_dict)

    i=0
    for acc in customer_dict:
        all_x_list[i] = customer_dict[acc]['Longitude']
        all_y_list[i] = customer_dict[acc]['Latitude']
        i+=1
    
#===============

        
    plt.figure('Line fig',figsize=(6,4))
    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])
    
    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')

    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.plot(x_list, y_list, color='r', linewidth=1, alpha=0.8, zorder = 10)

    ax.scatter(all_x_list, all_y_list, marker='.', c='k', s=1,zorder = 20)

    ax.scatter(x_list[0], y_list[0], marker='s', c='b', zorder=30)

    plt.show()

def draw_periods(route):
    for i in range(PERIOD):
        draw_a_day(route,i)

#draw_a_day(route_dict,3)
#draw_all_cluster(cluster_dict)
#5 Evaluation
def cost_sum(route_dict):
    s = 0
    for i in route_dict:
        for j in range(len(route_dict[i])):
            s += route_dict[i][j]['Cost']
            
    round(s,0)
    return s

def running_time():
    input_time = round(input_break - start_time,3)
    cluster_time = round(cluster_break - input_break,3)
    scheduling_time = round(scheduling_break - cluster_break,3)
    routing_time = round(routing_break - scheduling_break,3)
    LS_time = round(LS_break - routing_break, 3)

    re_time = round(LS_re_end - LS_re_start, 3)
    sw_time = round(LS_sw_end - LS_sw_start, 3)

    print(DATA_FILE)
    print('input time:\t',input_time, ' s')
    print('cluster time:\t',cluster_time, ' s')
    print('scheduling time:\t',scheduling_time, ' s')
    print('routing time:\t',routing_time, ' s')
    #print('LS time :\t', LS_time, ' s')

    print('Reallocation :\t', re_time, ' s')
    print('Swap :\t', sw_time, ' s')
    
    print('total time: ', round(LS_break - start_time,3), 's')
    print(len(cluster_dict))
    
# delete files, can be turn off
running_time()

print(counter_re, 'times reallocations')
print(counter_sw, 'times swap')

# delete tmp file produced by LKH solver
for cluster in fleet_dict:
    for date in range(len(fleet_dict[cluster])):
        fname = 'tsp_C' + str(cluster) + '_D' + str(date)
        if_exist_del(fname, '.par')
        if_exist_del(fname, '.tsp')
        if_exist_del(fname, '.txt')
 
# close cmd pop out by LKH solver
# failed, does not work
'''
print(solver_counter)
while solver_counter > 0:
    os.system('taskkill /f /im LKH-2.exe')
    time.sleep(TIME_DELAY)
    solver_counter -= 1
'''
# ---------------

if draw_route == True:
    draw_periods(route_dict)


