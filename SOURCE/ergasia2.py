import random
import sys
import csv
import pandas as pd #The pandas library
import pandas_datareader.data as web # For accessing web data
from pandas import Series, DataFrame #Main pandas data structures
import matplotlib.pyplot as plt #main plotting tool for python
import matplotlib as mpl
import seaborn as sns #A more fancy plotting library
from datetime import datetime #For handling dates
import scipy as sp #library for scientific computations
from scipy import stats #The statistics part of the library
import seaborn as sns #A more fancy plotting library
import numpy as np
from numpy import genfromtxt
import scipy as sp
import json
import itertools
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import keyboard 
from copy import deepcopy



#########################               
                
########## Vhma 2

#########################

def ReadMovies():#######load movies in df
    movies_df = pd.read_csv('movies.csv')
    
    return movies_df

#########################               
                
########## Vhma 3

#########################

def TriangularMatrixOfPairsCounters():####### 3a create a vector for all possible pairs counters
    
    global under_Triangular
    all=(k*(k-1))/2
    all=int(all)
    under_Triangular =[0 for i in range(all ) ]

    
    for bsk in userBaskets:
        real_ids=map_movies(bsk)
        pairs=create_pairs(real_ids)
        for pair in pairs:
            j=pair[0]
            i=pair[1]
            if(j<i):
                i=pair[0]
                j=pair[1]
           
         
                
            pos=(i-1)*(k-i/2)+j-i-1 
            pos=int(pos)
            
            under_Triangular[pos]+=1
    
    return under_Triangular

def HashedCountersOfPairs(userBaskets):###### 3b create a hash table with keys-value pairs to count the frequent pairs only
    
    global hash_dict
    hash_dict={}
    first_bsk=0
    kala8ia=0
    for bsk in userBaskets:
       
        all_bsk=create_pleiades(bsk,2)#### create 2-size tuples from bsk
        
        if(first_bsk==0):
                first_bsk=1
                for i in all_bsk:
                    if(i[0]<i[1]):#### check tuples to be sorted
                        hash_dict[(i[0],i[1])]=1
                    else:
                        hash_dict[(i[1],i[0])]=1
                        
                continue
            
        for i in all_bsk:
       
            flag=0
            keys=hash_dict.keys()
            if((i[0],i[1]) in keys):
                if(i[0]<i[1]):#### check tuples to be sorted
                    hash_dict[(i[0],i[1])]+=1
                else:
                    hash_dict[(i[1],i[0])]+=1
                    
            else:
                if(i[0]<i[1]):
                    hash_dict[(i[0],i[1])]=1
                else:
                    hash_dict[(i[1],i[0])]=1
        kala8ia+=1      
             
    return   hash_dict        

def create_kpleiades(L,L1):###### compute Ck from Lk-1 X L1
    k_pleiada={}
    
    for i in L:
        for j in L1:
            
            if(not(set(j).issubset(set(i)))):# j not in i in order to match the tuples
                
                new_tuple=i+j
                b=sorted(new_tuple)## sort tuple 
                k_pleiada[tuple(b)]=0
         
    return k_pleiada

#########################               
                
########## Vhma 4

#########################

def myApriori(itemBaskets, min_frequency,max_length):### 4  A-priori algorithm

    global frequencies,take
    i=1
    frequency_dict={}
    nextLevelEmpty=False
    N=len(itemBaskets)
    C={}
    L1={}
    requent_itemsets=[]
    L_tonos=[]
    for bsk in range(N):## compute C1
        
        i_sunola=create_pleiades(itemBaskets[bsk],1)
        for monosunolo in i_sunola:
            if monosunolo in C.keys():
                C[monosunolo]+=1
            else:
                C[monosunolo]=1
            
          
    for sunolo in C.keys():## compute L1 
            
            frequency=C[sunolo]/N
            
            if( frequency >= min_frequency):

                    L1[sunolo]=frequency
                    
    requent_itemsets.append(sorted(list(L1.keys())))
    for el in L1.keys(): 
        
        frequency_dict[el]=L1[el]
        

    C={}
    
    L2={}
    
    C=HashedCountersOfPairs(itemBaskets) ##### compute C2 using HashedCountersOfPairs from 3b
        
    for sunolo in C.keys(): ##### compute L2
            
            frequency=C[sunolo]/N
            
            if( frequency >= min_frequency):

                    L2[sunolo]=frequency
    i+=1              
    requent_itemsets.append(sorted(list(L2.keys())))
    for el in L2.keys():
        frequency_dict[el]=L2[el]
        
    Lk=L2.copy()
    i=3
    while(i<=max_length and  (len(Lk)>=1 ) ):########## from step 3 and then...until max_length
        Ck={}
        prev_freq={}
        prev_freq=Lk.copy()### save Lk-1
        
        Lk={}
        
        print("bhma i ",i)
        
        Ck=create_kpleiades(list(prev_freq.keys()),list(L1.keys()))###  create Ck= Lk-1 X L1
        prev_freq={}
        
        
        for bsk in range(N):
            
            bucket_size=len(itemBaskets[bsk])
            bsk_set=set(itemBaskets[bsk])
        
            if(bucket_size<50):#### check if the current bucket is small so is better to create the tuples of busket
                
                i_bucket=create_pleiades(itemBaskets[bsk],i)
                
                
                for tupl in i_bucket:## for every possible tuple of busket check if is in Ck dict
                
                    #if(tupl in ):  #set(tupl).issubset(set(real_ids))
                    
                    if( tupl in Ck.keys() ):

                        Ck[tupl]+=1
            else:###### if busket is too big is better to check  for every   possible tuple of Ck  if is subset of busket
                    ### than create alla possible tuples and check all of them
                for tupl in Ck:

                    if(( set(tupl)).issubset(bsk_set) ) :###ckeck for every possibly frequent tuple of is in busket
                        
                        Ck[tupl]+=1
                    
                    
                    
        for sunolo in Ck.keys():### compute the real frequent tuples based on min_frequency
            
            frequency=Ck[sunolo]/N
            
            if( frequency >= min_frequency):

                    Lk[sunolo]=frequency
                    
        requent_itemsets.append(sorted(list(Lk.keys())))
        for el in Lk.keys():
            frequency_dict[el]=Lk[el]
            
        
        
        
        i+=1
        
    frequencies=frequency_dict.copy()
         
    return requent_itemsets                
                
                

########################

#### Vhma 5

#######################


usersInSample=[]

def ReservoirSampling(i,SampleOfBaskets,current_user):### Reservoir Sampling Algorithm
    
        if(i<=sampling_size):
            SampleOfBaskets[current_user]=frozenset()
    
            usersInSample.append(current_user)
        if(i>sampling_size):
            j=random.randint(0, i)
            if(j<sampling_size):
                userGoOut=usersInSample[j]
                usersInSample[j]=current_user
                SampleOfBaskets.pop(userGoOut)
                SampleOfBaskets[current_user]=frozenset()
                
        return  SampleOfBaskets
                
        
    
    

def sampledApriori(all_positive): #### 5 
    global true_positives,possibly,FrequentMovieSet
    possibly=0
    true_positives=0
    SetOfUsers=[]
    is_interupted=0
    SampleOfBaskets={}
    ratings_stream = pd.read_csv('ratings_shuffled.csv')##### set the stream
    print(ratings_stream.head(10))
    
    for index, row in ratings_stream.iterrows():##### take the sample
        if keyboard.is_pressed('y') or keyboard.is_pressed('Y'):
            is_interupted=1
            break
        
        current_user=int(row['userId'])
        current_movie=int(row['movieId'])
        rating=float(row['rating'])
        ####if(rating>=min_rating):
        if(current_user not in SetOfUsers):
            
            SetOfUsers.append(current_user)
            SampleOfBaskets=ReservoirSampling(len(SetOfUsers),SampleOfBaskets,current_user)

        if((current_user in SampleOfBaskets.keys() )and rating>=min_rating):
           # print(current_user,"edww",current_movie)
            SampleOfBaskets[current_user]=SampleOfBaskets[current_user].union({current_movie})
            
    
    
    Apriori_in=[]
    for i in SampleOfBaskets.keys():##### transfom the sample to fit in the Apriori input
        Apriori_in.append(list(SampleOfBaskets[i]))
    print("Apriori input: " ,Apriori_in[0],len(Apriori_in))
    print(min_frequency,max_length)
    Result=myApriori(Apriori_in,min_frequency ,max_length)#####the Result of A-priori
    #print("Apriori Results: ",Result)
    
    PossiblyFrequentMovieSet={}
    for lista_ in range(len(Result)):#### create a dict for every possible frequent tuple 
                                        #####the A-PRIORI returned based to the sample...
        for pleiada in Result[lista_]:
        
                PossiblyFrequentMovieSet[frozenset(pleiada)]=0### set value in 0 in order to see if whice of them
                                                            #### are frequent and whice are not.
                
    possibly=len(PossiblyFrequentMovieSet)
    print("the number of possible frequent sets according to the sample is ",possibly)      
    AllBackets={}
    for index, row in ratings_stream.iterrows():#### second stream scan
        current_user=int(row['userId'])
        current_movie=int(row['movieId'])
        rating=float(row['rating'])
        if(rating>=min_rating):
            if(current_user not in AllBackets.keys()):### create all buskets
                AllBackets[current_user]=[current_movie]
            else:
                AllBackets[current_user].append(current_movie)
    ##
    for bucket in AllBackets.keys():### for all buskets

        for possible in PossiblyFrequentMovieSet.keys():####for all possibly frequent tuples we find from sample
            try:

                if( set(possible).issubset(set(AllBackets[bucket])) ): ##compute the frequencies

                    PossiblyFrequentMovieSet[possible]+=1
            except:
                print(possible," ",bucket)
                print(set(possible)," ",set(bucket))
                
    FrequentMovieSet={}          
    for i in PossiblyFrequentMovieSet.keys():####compute the real frequent tuples we find from sample

            frequency=PossiblyFrequentMovieSet[i]/N
            
            if( frequency > min_frequency):

                    FrequentMovieSet[i]=frequency
    ## print the results the F1,Precision,Recall to see if this is accurence...
    print("the real frequent sets from ",possibly,"  we find in the sample is ",len(FrequentMovieSet))
    true_positives=len(FrequentMovieSet.keys())
    false_positives=possibly-true_positives
    
     
    print("all_positives: ",all_positive)
    false_negatives=all_positive-true_positives
    print("False positives: ",false_positives," True positives: ",true_positives," False negatives: ",false_negatives)
    
    PRECISION = true_positives / ( true_positives + false_positives )
    RECALL = true_positives / ( true_positives + false_negatives )
    print("PRECISION: ",PRECISION," RECALL: ",RECALL)
    
    if((RECALL + PRECISION)==0):
        F1=0
    else:
        F1 = 2 * RECALL * PRECISION / ( RECALL + PRECISION )
    print("PRECISION: ",PRECISION," RECALL: ",RECALL," F1: ",F1)
    print("After the filtering with min frequency we have false_positives:",0)
    false_positives=0
    PRECISION = true_positives / ( true_positives + false_positives )
    RECALL = true_positives / ( true_positives + false_negatives )
    print("PRECISION: ",PRECISION," RECALL: ",RECALL)
    
    if((RECALL + PRECISION)==0):
        F1=0
    else:
        F1 = 2 * RECALL * PRECISION / ( RECALL + PRECISION )
    print("PRECISION: ",PRECISION," RECALL: ",RECALL," F1: ",F1)
   
              


########################

#### Vhma 6

########################
def create_Rules(requent_itemsets,sunolo,rules_df,min_confidence,MinLift,MaxLift,hypo,con,block_con,block_hypo):
                                        ##### function i use to create the rules retrospectively
        #global rules_df
        hypo=sorted(hypo)
        con=sorted(con)
        hypo=tuple(hypo)
        con=tuple(con)
        #print("ypo ",ypo,"con ",con)
        confidence=frequencies[sunolo]/frequencies[hypo]### compute confidence,lift,interest for the rule
        lift=confidence/frequencies[con]
        interest=confidence-frequencies[con]
        if(confidence>=min_confidence):##### check if his values are appropriate
            cond_=0
            if(MinLift==-1):
                if( lift<MaxLift ):
                    cond_=1
                                ##### check if his values are appropriate
                    
            if(MaxLift==-1):   
                if(lift>MinLift ):
                    cond_=1
                    
            if((lift>MinLift and MinLift>1) or (lift<MaxLift and   MaxLift>0 and MaxLift<1)):
                cond_=1
                
            if(cond_==1):##### check if his values are appropriate
                if( len(rules_df[rules_df['rule']==(str(list(hypo))+"->"+str(list(con)))])==0 ):
                           #### check if the rule exist if not is added in dataframe
                        rules_df.loc[len(rules_df)] = [set(sunolo),str(list(hypo))+"->"+str(list(con)),list(hypo),list(con),frequencies[hypo],confidence,lift,interest,len(rules_df)+1]
            
            
                for subset in itertools.combinations(hypo,len(hypo)-1):### go deep to explore all rules under this one
                    subset=sorted(subset) 
                    hypo=subset
                    con=set(sunolo)-set(subset)
                    if(len(hypo)==1):
                        block_hypo.append(hypo[0])### update block hypothesis ids we have one more list  for conclusion
                    if(len(hypo)>0 and len(con)>0 and len(set(con).intersection(block_con))==0 and( len(set(hypo).intersection(block_hypo))==0 or len(hypo)>1) ):
                        ### check if is need to go deeper
                        create_Rules(requent_itemsets,sunolo,rules_df,min_confidence,MinLift,MaxLift,hypo,con,block_con,block_hypo)
            
        

def AssociationRulesCreation(requent_itemsets,min_confidence,MinLift,MaxLift):#### 6
    
    #if( MinLift < 1  or MaxLift>1):
        #print("Not good values for MinLift,MaxLift!")
        #return
    cols=['itemset','rule', 'hypothesis', 'conclusion','frequency','confidence','lift', 'interest', 'rule ID']
    
    rules_df=pd.DataFrame(columns=cols)### set the Dataframe
    
    
    for i in range(1,len(requent_itemsets)): #### from 2-tuples sets and then
        print("start the set of : ",i+1)
        for sunolo in requent_itemsets[i]: #### for every tuple
            
            block_con=[]##set list for block ids to conclusion
            for j in range(len(sunolo)-1,-1,-1): #### for all rules like [all itemset -1]->1 from the end to begin
                
                lista=list(sunolo)
                
                con=[lista[j]]
                    
                sunolo_=list(deepcopy(sunolo))### set the hypothesis and the conclusion 
                sunolo_.pop(j)
                hypo=sunolo_
                block_hypo=[]## set list for block ids to hypothesis
                ###call the create_Rules function
                create_Rules(requent_itemsets,sunolo,rules_df,min_confidence,MinLift,MaxLift,hypo,con,block_con,block_hypo)
                block_con.append(lista[j])##update list for block ids to conclusion
                
    return rules_df

##############################

####### voithitikes

##############################
def create_pairs(list_of_same_movies):###create pairs from a list
    n=len(list_of_same_movies)
    
    pairs_list=[]
    for i in range(n):
        for j in range(i+1,n):
            lst=[]
            lst.append(list_of_same_movies[i])
            lst.append(list_of_same_movies[j])
            lst.sort(reverse=True)
            pairs_list.append(lst)
    return pairs_list    

  
def create_pleiades(lista,i):### create tuples with size i from a list
    all_possible=[]
    
        
    for subset in itertools.combinations(lista,i):
        subset=sorted(subset) ### sort tuples  
        sub=tuple(subset)
        
        all_possible.append(sub)
    	
    return all_possible #list(set(all_possible))
        
    

    
def map_movies(lista):###used in 3a in order to have sequence in movie ids
    new_list=[]
    
       
    for i in lista:
        new_list.append(movieMap[i])
    return new_list
        





def draw_graph(rules,draw_choice):### the top rule graph

    G = nx.DiGraph()

    color_map = []
    final_node_sizes = []

    color_iter = 0

    NumberOfRandomColors = 100
    edge_colors_iter = np.random.rand(NumberOfRandomColors)

    node_sizes = {}     # larger rule-nodes imply larger confidence
    node_colors = {}    # darker rule-nodes imply larger lift
    
    for index, row in rules.iterrows():

        color_of_rule = edge_colors_iter[color_iter]

        rule = row['rule']
        rule_id = row['rule ID']
        confidence = row['confidence']
        lift = row['lift']
        itemset = row['itemset']
        hypothesis=row['hypothesis']
        conclusion=row['conclusion']
        
        G.add_nodes_from(["R"+str(rule_id)])

        node_sizes.update({"R"+str(rule_id): float(confidence)})

        node_colors.update({"R"+str(rule_id): float(lift)})
        
        for item in hypothesis:
            G.add_edge(str(item), "R"+str(rule_id), color=color_of_rule)

        for item in conclusion:
            G.add_edge("R"+str(rule_id), str(item), color=color_of_rule)

        color_iter += 1 % NumberOfRandomColors

    print("\t++++++++++++++++++++++++++++++++++++++++")
    print("\tNode size & color coding:")
    print("\t----------------------------------------")
    print("\t[Rule-Node Size]")
    print("\t\t5 : lift = max_lilft, 4 : max_lift > lift > 0.75*max_lift + 0.25*min_lift")
    print("\t\t3 : 0.75*max_lift + 0.25*min_lift > lift > 0.5*max_lift + 0.5*min_lift")
    print("\t\t2 : 0.5*max_lift + 0.5*min_lift > lift > 0.25*max_lift + 0.75*min_lift")
    print("\t\t1 : 0.25*max_lift + 0.75*min_lift > lift > min_lift")
    print("\t----------------------------------------")
    print("\t[Rule-Node Color]")
    print("\t\tpurple : conf > 0.9, blue : conf > 0.75, cyan : conf > 0.6, green  : default")
    print("\t----------------------------------------")
    print("\t[Movie-Nodes]")
    print("\t\tSize: 1, Color: yellow")
    print("\t----------------------------------------")

    max_lift = rules['lift'].max()
    min_lift = rules['lift'].min()

    base_node_size = 500
    
    for node in G:

        if str(node).startswith("R"): # these are the rule-nodes...
                
            conf = node_sizes[str(node)]
            lift = node_colors[str(node)]
            
            # rule-node sizes encode lift...
            if lift == max_lift:
                final_node_sizes.append(base_node_size*5*lift)

            elif lift > 0.75*max_lift + 0.25*min_lift:
                final_node_sizes.append(base_node_size*4*lift)

            elif lift > 0.5*max_lift + 0.5*min_lift:
                final_node_sizes.append(base_node_size*3*lift)

            elif lift > 0.25*max_lift + 0.75*min_lift:
                final_node_sizes.append(base_node_size*2*lift)

            else: # lift >= min_lift...
                final_node_sizes.append(base_node_size*lift)

            # rule-node colors encode confidence...
            if conf > 0.9:
                color_map.append('purple')

            elif conf > 0.75:
                color_map.append('blue')

            elif conf > 0.6:
                color_map.append('cyan')

            else: # lift > min_confidence...
                color_map.append('green')

        else: # these are the movie-nodes...
            color_map.append('yellow') 
            final_node_sizes.append(2*base_node_size)

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]

    if draw_choice == 'c': #circular layout
        nx.draw_circular(G, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=True)

    elif draw_choice == 'r': #random layout
        nx.draw_random(G, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=True)

    else: #spring layout...
        pos = nx.spring_layout(G, k=16, scale=1)
        nx.draw(G, pos, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=False)
        nx.draw_networkx_labels(G, pos)    

    plt.show()

    # discovering most influential and most influenced movies
    # within highest-lift rules...
    outdegree_rules_sequence = {}
    outdegree_movies_sequence = {}
    indegree_rules_sequence = {}
    indegree_movies_sequence = {}
    
    outdegree_sequence = nx.out_degree_centrality(G)
    indegree_sequence = nx.in_degree_centrality(G)

    for (node, outdegree) in outdegree_sequence.items():
        # Check if this is a rule-node
        if str(node).startswith("R"):
            outdegree_rules_sequence[node] = outdegree
        else:
            outdegree_movies_sequence[node] = outdegree
            
    for (node, indegree) in indegree_sequence.items():
        # Check if this is a rule-node
        if str(node).startswith("R"):
            indegree_rules_sequence[node] = indegree
        else:
            indegree_movies_sequence[node] = indegree

    max_outdegree_movie_node = max(outdegree_movies_sequence, key=outdegree_movies_sequence.get)
    max_indegree_movie_node = max(indegree_movies_sequence, key=indegree_movies_sequence.get)
    print("\tMost influential movie (i.e., of maximum outdegree) wrt involved rules: ",max_outdegree_movie_node)
    print("\tMost influenced movie (i.e., of maximum indegree) wrt involved rules: ",max_indegree_movie_node)





###################################################

# Input from User

####################################################



print("==================== A PRIORI INPUT ====================")
min_frequency=float(input("Minimun Frequency, MinSupport, as a number in (0,1) : "))
min_confidence=float(input("Minimun Confidence ( Confidence(A->B) = frequency(A union B) / frequency(A)) : "))
min_lift=float(input("Minimun Lift (Lift(A->B) = Confidence(A->B) / frequency(B))(ignore value: -1) : "))
max_lift=float(input("Maximum Lift (Lift(A->B) = Confidence(A->B) / frequency(B))(ignore value: -1) : "))
max_length=int(input("Maximum Length of Itemset : "))
print("==================== RATINGS INPUT ======================")
min_rating=float(input("Put movies in user 's basket only if exceeds a minimum rating (give value in [0,5]) : "))
print("==================== RESERVOIR INPUT ======================")
sampling_size=int(input("Sample size for Reservoir Sampling Algorithm ( -1 to avoid reservoir sampled Apriori step ) : "))

###################################################

# loading from file

####################################################



def CreateMovieBaskets():#### load the buskets
    global movieMap,movieList,ratingList
    with open('ratings.csv','r') as rating:
        
        rating= csv.reader(rating,delimiter=',')#,header=None
        next(rating)
    
        ###########userList define
        userList=[]
        previous_user=1
        movies_of_user=[]
        ###########movieMap define
        movieMap={}
        movie_id=1
        ###########movieList define
        movieList={}
        ratingList={}
        movies_of_user=[]
        ratings_of_user=[]
        
        for line in rating:
            ###########userList
            rating_i=float(line[2])
            if(rating_i >= min_rating):
                user_newi=int(line[0])
                if (previous_user != user_newi):
           
                    userList.append(movies_of_user)
                    ratingList[previous_user]=ratings_of_user
                    movies_of_user=[]
                    ratings_of_user=[]
                    previous_user=user_newi
        
                movie_i=int(line[1])
        
            
                movies_of_user.append(movie_i)#### allagh
                ratings_of_user.append(rating_i)
            ###########userList
            ###########movieMap
                if (not(movieMap.get(movie_i)) ):
                    movieMap[movie_i]=movie_id
                    movie_id=movie_id+1
            ###########movieMap
            ###########movieList
                all_movie_keys=movieList.keys()
                if(movie_i in all_movie_keys):
                    users_of_movie=movieList.get(movie_i)
                    if( not(user_newi in users_of_movie) ):
                        users_of_movie.append(user_newi)
                        users_of_movie.sort()
                        movieList.update({movie_i: users_of_movie})
                

                else:
                    li=[]
                    li.append(user_newi)
                    movieList[movie_i]=li
                    
        userList.append(movies_of_user)
        ratingList[previous_user]=ratings_of_user
    
    
    return userList   



            
        
        
                    
            
                




        
        

    
userBaskets=CreateMovieBaskets()


N=len(userBaskets)
#print("userBaskets[0]:",userBaskets[0])
#print('\n \n ')
print('\n \n ')
#print('Users :  ',N)
#print(userBaskets)
k=len(movieMap)
#print('Movies : ',k)
print('\n  ')
movies_df=ReadMovies()
all_movies=movies_df.shape[0]
#print(movies_df[:5])





#under_Triangular=TriangularMatrixOfPairsCounters()



#hash_table=HashedCountersOfPairs()
#print(hash_table)
print("======================= A PRIORI EXECUTION ==========================")
print("Input parameters:")
print("(min_frequency,min_confidence,min_lift,max_lift,max_length,min_rating,sampling_size)=[",min_frequency,",",min_confidence,",",min_lift,",",max_lift,",",max_length,",",min_rating,",",sampling_size,"]")

print("Please wait...")
frequencies={}
take=1


print("======================= A PRIORI OUTPUT ===============================")

#print("Apriori Results(4): ",requent_itemsets)


        

time0=time.time()
#for i in range(1,6):
   # for bsk in range(len(userBaskets)):
       # i_bucket=create_pleiades(userBaskets[bsk],i)
requent_itemsets=myApriori(userBaskets, min_frequency ,max_length)
all_freq=len(frequencies)
print("all_freq",all_freq)
count=0
for i in range(len(requent_itemsets)):
    count+=len(requent_itemsets[i])
    

print(count)
time1=time.time()
print("the execute time of A priori is ",time1-time0)
if(sampling_size!=-1):
    print("====== Execute Sampling A-priori ========")
    all_frequencies=frequencies.copy()
    sampledApriori(all_freq) #### vhma 5           
    frequencies=all_frequencies.copy()        

print("Create Rules: ")

print("Please wait...")
time2=time.time()
rules_df=AssociationRulesCreation(requent_itemsets,min_confidence,min_lift,max_lift)#requent_itemsets
time3=time.time()
print("The execute time for rule creation is: ",time3-time2)
print("Discovered Rules:",rules_df.shape[0])
#print(rules_df)
file_name = 'Rules_file='+str(N)+'_minfreq='+str(min_frequency)+'_minconf='+str(min_confidence)+'_minlif='+str(min_lift)+'_maxlif='+str(max_lift)+'_minrating='+str(min_rating)+'maxlength='+str(max_length)+'.csv'
rules_df.to_csv(file_name)
file_name_2 = 'Apriori_file='+str(N)+'_minfreq='+str(min_frequency)+'_minrating='+str(min_rating)+ '_maxlength='+str(max_length)+'.csv'
f = open(file_name_2,'w')
line_=1
for i in frequencies.keys():
    row = str(line_)+' '+'key: '+str(i)+' value: '+str(frequencies[i])+'\n'
    f.write(row)
    line_+=1
    
f.close()
    

answer=' '







def presentResults(rules_df,movie_df):
    global answer
    
    while(True):

        
        print("===========================================================================")
        print("(a) List ALL discovered rules                      [format: a] ")
    
        print("(b) List all rules containing a BAG of movies      [format: ")
        
        print("in their <ITEMSET|HYPOTHESIS|CONCLUSION>           b,<i,h,c>,<comma-sep. movie IDs>] ")
        
        print("(c) COMPARE rules with <CONFIDENCE,LIFT>           [format: c] ")
        
        print("(h) Print the HISTOGRAM of <CONFIDENCE|LIFT >      [format: h,<c,l >] ")
        
        print("(m) Show details of a MOVIE                        [format: m,<movie ID>] ")
        
        print("(r) Show a particular RULE                         [format: r,<rule ID>] ")
        
        print("(s) SORT rules by increasing <CONFIDENCE|LIFT >    [format: s,<c,l >] ")
        
        print("(v) VISUALIZATION of association rules             [format: v,<draw_choice: ")
        print("(sorted by lift)                                   [c(ircular),r(andom),s(pring)]>, ")
        print("                                                   <num of rules to show>] ")
        print("(e) EXIT                                           [format: e] ")
        print("===========================================================================")
        print("\n")
        answer=input("Provide your option : ")
        if(answer=='e'):
            break
        elif(answer =='a'):
            answer=' '
            for i in range(rules_df.shape[0]):
                print("Rule( ",rules_df['rule ID'][i]," ) : ",rules_df['rule'][i])
            
        answer_array=answer.split(",")
        if(answer_array[0]=='r'):
            
            answer=' '
            for j in range(1,len(answer_array)):
                item_set=list(rules_df['itemset'].loc[rules_df['rule ID']==int(answer_array[j])].values[0])
                id_=rules_df['rule ID'].loc[rules_df['rule ID']==int(answer_array[j])].values[0]
                #print(rules_df.loc[rules_df['rule ID']==int(answer_array[1])])
                print("Rule ID         =                          ",id_)
                print("Itemset         =                          ",item_set)
                print("Involved Movies: ")
                for i in item_set:
                    print("The title of movie ",i," is :   ",movie_df[['title']].loc[movie_df['movieId']==i].values[0][0])
                print("Rule            =                          ",rules_df['rule'].loc[rules_df['rule ID']==int(answer_array[j])].values[0])
                print("Frequency       =                          ",rules_df['frequency'].loc[rules_df['rule ID']==int(answer_array[j])].values[0])
                print("Confidence      =                          ",rules_df['confidence'].loc[rules_df['rule ID']==int(answer_array[j])].values[0])
                print("Lift            =                          ",rules_df['lift'].loc[rules_df['rule ID']==int(answer_array[j])].values[0])
                print("Interest        =                          ",rules_df['interest'].loc[rules_df['rule ID']==int(answer_array[j])].values[0])
        
        if(answer_array[0]=='c'):
            
            fig = sns.scatterplot(rules_df['lift'],rules_df['confidence'])
            fig.set_xlabel('Lift')
            fig.set_ylabel('Confidence')
            plt.show()
            #fig.show()
            
        if(answer_array[0]=='v'):  
           
            if(len(answer_array)==3):
                if(answer_array[1] in ['c','r','s']):
                    top_df=rules_df.copy()
                    top_df.sort_values(by='lift',ascending=False,inplace=True)
                    draw_graph(top_df.head(int(answer_array[2])),answer_array[1])
                    
        if(answer_array[0]=='h'):  
            
            if(len(answer_array)==2):
                
                if(answer_array[1]=='c'):
                    min_conf=rules_df['confidence'].min()
                    max_conf=rules_df['confidence'].max()
                    print("min,max",min_conf,max_conf)
                    num_bins =[round(min_conf+i*0.05,2) for i in range(math.ceil((max_conf-min_conf)/0.05)+1)]
                    
                    print(num_bins)
                    #n, bins, patches = plt.hist(rules_df['confidence'], num_bins, facecolor='blue', alpha=0.50)
                    #plt.xlim(xmin=min_conf, xmax = max_conf)
                    rules_df['confidence'].plot.hist(bins=12, alpha=0.5)
                    #plt.xticks(num_bins)
                    plt.title("Histogram of CONFIDENCES among discovered rules")
                    plt.xlabel("Confidence")
                    plt.ylabel("Number of Rules")
                    plt.tight_layout()
                    plt.show() 
                   
                
                if(answer_array[1]=='l'):
                    
                    min_lift=rules_df['lift'].min()
                    max_lift=rules_df['lift'].max()
                    print("min,max",min_lift,max_lift)
                    num_bins =[round(min_lift+i*0.5,2) for i in range(math.ceil((max_lift-min_lift)/0.5)+1)]
                     
                    print(num_bins)
                    #n, bins, patches = plt.hist(rules_df['lift'], num_bins, facecolor='red', alpha=0.50)
                    rules_df['lift'].plot.hist(bins=12, alpha=0.5)
                    plt.title("Histogram of Lifts among discovered rules")
                    plt.xlabel("Lift")
                    plt.ylabel("Number of Rules")
                    plt.tight_layout()
                    plt.show()
                   
            
        if(answer_array[0]=='b'):
            movie_list=[]
            answer=' '
            for i in range(2,len(answer_array)):
                movie_list.append(int(answer_array[i]) )
                                  
            if(answer_array[1]=='i'):
            
                for j in range(rules_df.shape[0]):
                 
                    if(set(movie_list).issubset(set(rules_df['itemset'][j]) )  ):
                        
                            print("Rule(",rules_df['rule ID'][j],")       ",rules_df['rule'][j])
                                  
            if(answer_array[1]=='h'):
                                  
                for j in range(rules_df.shape[0]):
                 
                    if(set(movie_list).issubset(set(rules_df['hypothesis'][j]) )  ):
                        
                            print("Rule(",rules_df['rule ID'][j],")       ",rules_df['rule'][j])
                                                                   
            if(answer_array[1]=='c'):
                
                for j in range(rules_df.shape[0]):
                 
                    if(set(movie_list).issubset(set(rules_df['conclusion'][j]) )  ):
                        
                            print("Rule(",rules_df['rule ID'][j],")       ",rules_df['rule'][j])
              
                        
                            
        if(answer_array[0]=='m'):
            
            print("\n")
            print("Movie id =   ",movies_df.loc[movies_df['movieId']==int(answer_array[1])].values[0][0])
            print("Title    =   ",movies_df.loc[movies_df['movieId']==int(answer_array[1])].values[0][1])
            print("Genre    =   ",movies_df.loc[movies_df['movieId']==int(answer_array[1])].values[0][2])
            answer=" "
            
           
        if(answer_array[0]=='s'):
            sorted_df=rules_df.copy()
            #print(sorted_df)
            print(sorted_df.shape)
            if(len(answer_array)==1):
                continue
            if(answer_array[1]=='c'):
                sorted_df.sort_values('confidence',ascending=True,inplace=True)
                #print(sorted_df)
                for i in sorted_df.index:
                    print("Rule(",rules_df['rule ID'][i],"): ",sorted_df['rule'][i],"with confidence : ",sorted_df['confidence'][i])
                    
            if(answer_array[1]=='l'):
                    sorted_df.sort_values(by='lift',ascending=True, inplace=True)
                    for i in sorted_df.index:
                        print("Rule(",rules_df['rule ID'][i],"): ",sorted_df['rule'][i]," with lift   : ",sorted_df['lift'][i])
                        
            answer=" "
                
    
presentResults(rules_df,movies_df)

#cols=['itemset','rule', 'hypothesis', 'conclusion','frequency','confidence','lift', 'interest', 'rule ID']
                                  
    
    
    
    
    

#len(rules_df[(rules_df['confidence']>=0.95) & (rules_df['confidence']<=1)])

#rules_df=pd.DataFrame(columns=["support: 0.05","support: 0.1","support: 0.15","support: 0.2"])
count=0  
confidence_lines={}
lift_lines={}
freqs=[0.05,0.1,0.15,0.20]#0.30]#0.05,

for i in freqs:
       # i_bucket=create_pleiades(userBaskets[bsk],i)
    confidence_for_line={}
    lift_for_line={}
    time0=time.time()
    print("execute for min_frequency: ",i)
    requent_itemsets=myApriori(userBaskets, i ,max_length)
    time1=time.time()
    
    print("The execute time for a-priori is: ",time1-time0)
          
           

    print("Create Rules: ")            
    print("Please wait...")
    time2=time.time()
    rules_df=AssociationRulesCreation(requent_itemsets,min_confidence,min_lift,max_lift)#requent_itemsets
    time3=time.time()
    print("The execute time for rule creation is: ",time3-time2)
    print("Discovered Rules:",rules_df.shape[0])
    
    
    
    
    min_conf=rules_df['confidence'].min()
    max_conf=rules_df['confidence'].max()
    #print("min,max (confidence)",min_conf,max_conf)
    
    #num_bins1 =[round(min_conf+j*0.05,2)  for j in range(math.ceil((max_conf-min_conf)/0.05)+1)]
                    
    #print("confidence bins",num_bins1)
    n1, bins1, patches = plt.hist(rules_df['confidence'], 12, facecolor='blue', alpha=0.50)
    print("@@@n1: ",n1)
    print("bins1 &&&: ",bins1)
    for j in range(len(bins1)-1):
        
        confidence_for_line[bins1[j]]=n1[j]         
                    
    min_lift=rules_df['lift'].min()
    max_lift=rules_df['lift'].max()
    #print("min,max (lift)",min_lift,max_lift)
    
    #num_bins2 =[round(min_lift+j*0.5-0.05,1) for j in range(12)]
                     
    #print("lift bins",num_bins2)
    n2, bins2, patches = plt.hist(rules_df['lift'], 12, facecolor='red', alpha=0.50)
    print("@@n2: ",n2)
    print("bins2: &&&&",bins2)
    for j in range(len(bins2)-1):
        
        lift_for_line[bins2[j]]=n2[j]         
              
    confidence_lines[count]=confidence_for_line
    lift_lines[count]=lift_for_line
    count+=1
    print("============")  
    
                

conf_lines_df=pd.DataFrame(confidence_lines[0].values(),index=confidence_lines[0].keys(),columns=["min_frequency:"+str(freqs[0])])
for i in range(1,len(freqs)):
    conf_lines_df["min_frequency:"+str(freqs[i])]=confidence_lines[i].values()
    
print(conf_lines_df.head(5))
print(conf_lines_df[['min_frequency:'+str(0.2)]].head(5))
print(lift_lines)


lift_lines_df=pd.DataFrame(lift_lines[0].values(),index=lift_lines[0].keys(),columns=["min_frequency:"+str(freqs[0])])
for i in range(1,len(freqs)):
    lift_lines_df["min_frequency:"+str(freqs[i])]=lift_lines[i].values()

print(lift_lines_df.head(5))
print(lift_lines_df[['min_frequency:'+str(0.2)]].head(5))
plt.show()

lift_lines_df['min_frequency:'+str(freqs[0])].plot(label = 'min_frequency:'+str(freqs[0]),color="b")
lift_lines_df['min_frequency:'+str(freqs[1])].plot(label = 'min_frequency:'+str(freqs[1]),color="r")
lift_lines_df['min_frequency:'+str(freqs[2])].plot(label = 'min_frequency:'+str(freqs[2]),color="g")
lift_lines_df['min_frequency:'+str(freqs[3])].plot(label = 'min_frequency:'+str(freqs[3]),color="orange")
_ = plt.legend(loc='best')
plt.title("lines of LIFTS")
plt.xlabel("Lift")
plt.ylabel("Number of Rules")
plt.show()



conf_lines_df['min_frequency:'+str(freqs[0])].plot(label = 'min_frequency:'+str(freqs[0]),color="b")
conf_lines_df['min_frequency:'+str(freqs[1])].plot(label = 'min_frequency:'+str(freqs[1]),color="r")
conf_lines_df['min_frequency:'+str(freqs[2])].plot(label = 'min_frequency:'+str(freqs[2]),color="g")
conf_lines_df['min_frequency:'+str(freqs[3])].plot(label = 'min_frequency:'+str(freqs[3]),color="orange")
_ = plt.legend(loc='best')
plt.title("lines of CONFIDENCES")
plt.xlabel("Confidence")
plt.ylabel("Number of Rules")
plt.show()



                
                
        
        
        
    
