import math
import random
import numpy as np
import pandas
import matplotlib.pyplot as plt
import time
import progress.bar as Bar

start = time.time() #user time
t = time.process_time() # process time
 
#Nivestar

#Flags
CODE_VERBOSE = True
DEBUG = False
MINIMIZATION = True #flag to change the AG to maximization (False) or minimization (True)
GRAPH = True

#calibration variables -----------
global_best_execution = []
global_nTimes_got_optimal_solution = 0
#-----------

#Travelling Salesman Problem 

class Genetic_algorithm():

	def __init__(self, n_population=100, cross_rate=1, n_generations=100, mutate_rate=0.10, n_elits=1, 
			selection_method='R', tournament_count = 2, mutation_burst = True):
			
		self.my_population = []
		self.fits = []

		self.file_reading()
		self.mapping()

		self.mutation_burst = mutation_burst
		self.cross_rate = cross_rate
		self.mutation_rate = mutate_rate
		self.n_population = n_population
		self.n_generations = n_generations
		self.n_elits = n_elits 
		self.selection_method = selection_method
		self.tournament_count = tournament_count
		assert( self.n_elits != 0 ), "Number of elits must be bigger than 0"

	def file_reading(self):

		self.distance_matrix, self.solution = read_file()

	def mapping(self): #precision and genes can change...

		self.precision = 1
		count_cities = len(self.distance_matrix)
		self.genes = count_cities

	def evolve(self):
		
		#global for calibrate
		global global_best_execution, global_nTimes_got_optimal_solution
		best_generation_list = []	
		generation, wait, boost = 0, 0, 0

		if CODE_VERBOSE:
			print("Creating a po pulation of ", self.n_population, " individuals")
		
		self.create_population()
		best = self.func_obj( self.my_population[0] )
		best_ind = self.my_population[0]

		while (generation < self.n_generations):
			
			wait, boost = self.mutation_burst_function(best_generation_list, wait, boost)
			if CODE_VERBOSE:
				print("Generation ", generation)
				
			self.fits = []
			self.fits = self.evaluation( self.my_population)
			if DEBUG:
				print("Population:\n", self.my_population[:])
				print("Scores:\n", self.fits[:])
			self.crossover()
			if( generation != self.n_generations -1): # if it is the last generation, no mutate
				self.mutation()
			elit = self.elitism()
			best, best_ind = self.get_best(best, best_ind, elit)
			best_generation_list.append(elit)
			self.my_population = self.inter_population.copy()
			generation+=1

		if CODE_VERBOSE:
			print("END")
		if DEBUG:
			print("Final Population:\n", self.my_population)
			print("Final Score Population:\n", self.fits)	
			print("Elits:\n", best_generation_list)
		if GRAPH:
			self.draw_graph(best_generation_list)

		print(f"The individual: { best_ind } was the best, with fitness: {best}") 
		print(f"Solution:\t{self.solution}, Solution fitness: {self.func_obj(self.solution)}")
		if(self.func_obj(self.solution) == best):
			print("\nBest Solution Reached")
			global_nTimes_got_optimal_solution +=1
		global_best_execution.append(best)

	def create_population(self):
		self.my_population.append([2, 9, 15, 7, 5, 11, 4, 10, 8, 6, 14, 12, 3, 1, 13]) #add 450 individual fit value to fix the start
		for _ in range (self.n_population):
			self.my_population.append( self.create_chromossomes() )	

	def create_chromossomes(self): #can change... start with best known result from other runs

		new_born = [*range(1, self.genes+1, 1)]
		random.shuffle(new_born)
		return new_born 

	def mutation_burst_function(self, best, wait, boost): 		### burst mutation when it still stuck

		if self.mutation_burst:
			count_generations_boost = round(5* (self.n_generations/100)) #duration is 5% of number of generations
			lasted_elit = best[-5:] # if the last five elits don't changed, we improve the mutation rate
			if(  (len(lasted_elit) == 5) and (len(np.unique(lasted_elit)) == 1) and (wait == 0)):	
				if CODE_VERBOSE:
					print("Stuck, improving mutation rate to 40%...")
				if DEBUG:
					print("Bests so far...\n", best)
				self.mutation_rate *= 2	#mutation rate boost for X generations
				boost = count_generations_boost  #boost time
				wait = count_generations_boost #idle time 
			elif(boost > 0):
				boost -=1 #still in boost time	
			elif(wait == 5 and boost == 0): #return old mutation rate
				self.mutation_rate /= 2
				if CODE_VERBOSE:
					print("The mutation rate returned to default...")
				wait -=1  
			elif(wait > 0 ):  
				wait -=1 #still in idle time

		return wait, boost

	def evaluation(self, population): 

		new_fits = []
		for ind in population:
			x = []
			for i in range(self.genes):
				x.append(ind[i])
			new_fits.append(self.func_obj(x))
		return new_fits

	def func_obj(self, ind): #can change...

		distance = 0
		for c, _ in enumerate(ind):
			if (c < self.genes-1):
				distance += self.distance_matrix[ind[c]-1][ind[c+1]-1]
		distance += self.distance_matrix[ind[self.genes-1]-1][ind[0]-1]
		return distance

	def crossover(self):

		count_cross = round(self.n_population * self.cross_rate)
		if(count_cross % 2 != 0):
			count_cross +=1
		self.inter_population = []

		selection_probs = self.make_roullete()

		for _ in range(0, count_cross, 2):
			if self.selection_method == 'T':
				fathers = self.selection_tournament()
			elif self.selection_method == 'R':
				fathers = self.selection_roulette(selection_probs)
			else:
				print("Error: Selection Method invalid")

			temp = [fathers[0], fathers[1]]
			self.ox_crossover(temp)
	
		rest_pop = self.n_population - count_cross
		for _ in range (rest_pop):
			self.inter_population.append( random.choice(self.my_population) )
		assert (self.n_population - self.n_elits <= len(self.inter_population) ), "Error: Intermediate population \
			is larger than the initial population"

	def selection_tournament(self):

		fathers = []

		for _ in range(2):
			competitors = random.sample(self.fits, self.tournament_count)
			if MINIMIZATION:
				winner = min(competitors)			
			else:
				winner = max(competitors)
			winner_index = self.fits.index(winner)
			fathers.append(winner_index)
		return fathers

	def make_roullete(self):

		roulette_values = []
		for ind in range (self.n_population):
			if(MINIMIZATION):
				fit_r = 1/self.fits[ind]   
			else:
				fit_r = self.fits[ind]  
			roulette_values.append(fit_r) #probability 
		sum_r = sum(roulette_values)
		selection_probs = [ c/sum_r for c in roulette_values]
		return selection_probs

	def selection_roulette(self, selection_probs):

		fathers = []
		for _ in range(2):
			index_sorted_father = np.random.choice(self.n_population, p=selection_probs)
			fathers.append(index_sorted_father)
		return fathers

	def ox_crossover(self, fathers): 

		father1 = self.my_population[fathers[0]]
		father2 = self.my_population[fathers[1]]

		#sorting chromosses
		len_individual = self.genes*self.precision
		sorted_chromossome1 = random.randrange( round(0.2*len_individual), round(0.4*len_individual))
		sorted_chromossome2 = random.randrange( round(0.5*len_individual), round(0.8*len_individual))
		
		start = sorted_chromossome1
		end = sorted_chromossome2

		slice1 = father1[start:end]
		slice2 = father2[start:end]
		son1 = [-1] * self.genes
		son2 = [-1] * self.genes
		son1[start:end] = slice1 #receive a piece of father 1
		son2[start:end] = slice2 #receive a piece of father 2
		rest_father1 = father1[end:] +  father1[:start] + father1[start:end] #array with elements to check
		rest_father2 = father2[end:] +  father2[:start] + father2[start:end] #array with elements to check
		#son 1
		index = end
		for value in rest_father2:
			if(value in son1):
				pass  #already exist in son, just pass
			else:
				if(index != self.genes): #end of son chromosomes? 
					son1[index] = value
					index+=1
				else:  # start to fill in the initials
					index=0
					son1[index] = value
					index+=1
		#son 2
		index = end
		for value in rest_father1:
			if(value in son2):
				pass  #already exist in son, just pass
			else:
				if(index != self.genes): #end of son chromossoms? 
					son2[index] = value
					index+=1
				else:  # start to fill in the initials
					index=0
					son2[index] = value
					index+=1

		self.inter_population.append(son1)
		self.inter_population.append(son2)

	'''
	def two_points_crossover(self, fathers): #Travelling Salesman Problem 

		f1_index = fathers[0]
		f2_index = fathers[1]
		father1 = self.my_population[f1_index]
		father2 = self.my_population[f2_index]
		sorted_chromossome = random.randrange(1, self.genes*self.precision-1)
		father1_chromossome = father1[:sorted_chromossome]
		father2_chromossome = father2[sorted_chromossome:]
		new_son = father1_chromossome + father2_chromossome
		new_son2 = father2_chromossome + father1_chromossome
		self.inter_population.append(new_son)
		self.inter_population.append(new_son2) 
	'''

	def mutation(self):

		for ind in self.inter_population:
			for index, city in enumerate(ind):   #mutation for gene
				self.mutation_method(index, ind)

	def mutation_method(self, index, ind): #method switch gene

		if random.uniform(0, 1) < self.mutation_rate:
			sorted_chromossome = random.randrange(1, self.genes*self.precision -1)
			while(sorted_chromossome == index):
				sorted_chromossome = random.randrange(1, self.genes*self.precision -1)
			switcher = ind[sorted_chromossome]
			ind[sorted_chromossome] = ind[index]
			ind[index] = switcher

	def elitism(self):

		fits_sorted = self.fits.copy()
		if(MINIMIZATION):
			fits_sorted.sort() 
		else:	
			fits_sorted.sort(reverse=True)
		best_fits = fits_sorted[:self.n_elits]

		for i in range(self.n_elits):
			best_index = self.fits.index(best_fits[i])
			r = random.randrange( 0, len(self.inter_population))
			self.inter_population[r] = self.my_population[best_index]

		return fits_sorted[0]

	def get_best(self, best, best_ind, new_best):

		if(MINIMIZATION):
			if(new_best < best):
				best = new_best
				best_ind = self.my_population[ self.fits.index(best) ]
		else:
			if(new_best > best):
				best = new_best
				best_ind = self.my_population[ self.fits.index(best) ]
		return best, best_ind

	def draw_graph(self, best_generation_list):

		best_generation_list.insert(0,450) #graphic size fixed 450
		plt.plot([*range(self.n_generations+1)], best_generation_list )
		plt.axline( (0, self.func_obj(self.solution)), (self.n_generations, 
			self.func_obj(self.solution)), color='g', linestyle='--', label="Best Solution")
		plt.ylabel("Distance")
		plt.xlabel("Generations")
		plt.legend(loc='lower left')
		plt.show() 

def read_file():

	f_distanceMatrix = np.loadtxt("tests/lau15_dist.txt", dtype='int')

	n_fsolution= "tests/lau15_tsp.txt"
	f_solution = open(n_fsolution)
	solution = f_solution.readlines()
	solution = [int(s) for s in solution]
	
	return f_distanceMatrix, solution

def calibrate():

	global global_best_execution, global_nTimes_got_optimal_solution, DEBUG, CODE_VERBOSE, GRAPH

	CODE_VERBOSE = False
	DEBUG = False
	GRAPH = False

	
	n_generations_list = [50, 200]
	n_population_list = [50, 100, 250]
	mutation_rate_list = [0.01, 0.05, 0.1]
	cross_rate_list = [0.7, 1]
	n_elits_list = [1, 5]
	selection_method_list = ['T', 'R']
	tournament_list = [2, 10] 

	df = pandas.DataFrame()
	mutation_column = []
	cross_column = []
	population_column = []
	generation_column = []
	elits_column = []
	selection_column = []
	tournament_column = []
	params_column = []

	list_best_execution = [] #best result for execution 
	times_best_solution = 0

	final_mean_list_best_execution = [] #mean of best results for execution 
	final_sum_times_best_solution = []

	times_repetition = 10

	#progress estimative ...
	total_progress = len(n_generations_list) * len(n_population_list) *	len(mutation_rate_list ) * len(cross_rate_list ) \
		* len(n_elits_list ) * len(selection_method_list ) * len(tournament_list ) * times_repetition
	my_bar = Bar.ShadyBar('Calibrating...', max=total_progress,  suffix='%(percent)d%%')

	for g in n_generations_list:
		for p in n_population_list:
			for m in mutation_rate_list:
				for c in cross_rate_list:
					for e in n_elits_list:
						for s in selection_method_list:
							for t in tournament_list:
								mutation_column.append(m)
								cross_column.append(c)
								population_column.append(p)
								generation_column.append(g)
								elits_column.append(e)
								selection_column.append(s)
								tournament_column.append(t)
								print("Execution with the current params: ")
								print(f'Generation: {g}, Population: {p}, Mutation Rate: {m}, Cross_Rate: {c}, Elitism: {e}, Selection_Method: {s}, Tournament: {t}')
								params_column.append(f'Mutation_Rate {m}, Cross_Rate {c}, Population {p}, Generation {g}, Elitism {e}, Selection_Method {s}, Tournament: {t}')
								
								for _ in range(times_repetition): #doing X times									
									print('\n\n')
									my_bar.next()
									print('\n\n')

									ga = Genetic_algorithm(mutate_rate=m, cross_rate=c, n_population=p, n_generations=g, n_elits=e,
										selection_method= s, tournament_count=t)
									ga.evolve()
									list_best_execution.append(global_best_execution)
									times_best_solution += global_nTimes_got_optimal_solution
								
									#resets
									global_best_execution = []
									global_nTimes_got_optimal_solution = 0

								final_mean_list_best_execution.append(np.mean(list_best_execution))
								final_sum_times_best_solution.append(times_best_solution)
								
								#clean lists
								list_best_execution.clear()
								times_best_solution = 0

								if(s == 'R'): #Torunament permutation is only relevant if selection method is 'T'
									break
							
	df['Params'] = params_column
	df["Mutation_Rate"] = mutation_column
	df["Cross_Rate"] = 	cross_column
	df["Population_Size"] = population_column
	df["Total_Generations"] = generation_column
	df["Elistism"] = elits_column
	df["Selection_Method"] = selection_column
	df["Tournament_Count"] = tournament_column
	df["Bests_Results"] = final_mean_list_best_execution
	df["Number_times_got_best_solution"] = final_sum_times_best_solution

	my_bar.finish()
	df.to_csv('results_params_AG.csv', sep=';')

if __name__ == "__main__":
	
	ga = Genetic_algorithm( mutate_rate=0.05, cross_rate=1, n_population=200, n_generations= 100, n_elits=1,
		 selection_method= 'T', tournament_count=10)
	if CODE_VERBOSE:
			print("Params= \n Mutate_Rate = ", ga.mutation_rate, "\nCross Rate =", ga.cross_rate, 
				"\nN_Population =", ga.n_population, "\nN_Generations = ", ga.n_generations, "\nN_elits = ", ga.n_elits,
				"\nSelection Method = ", ga.selection_method, "\nTournament_count = ", ga.tournament_count)
	
	Calibrate = False
	if Calibrate:
		calibrate()
	else:
		ga.evolve()

	end = time.time() #user
	user_time = end - start 
	elapsed_time = time.process_time() - t #process

	print("="*100)
	print("User time: %s" % user_time)
	print("Process time: %s" % elapsed_time)
	print( time.strftime("%H hours %M minutes %S seconds", time.gmtime(user_time)) ) #outra forma elegante de ver
	print("="*100)
