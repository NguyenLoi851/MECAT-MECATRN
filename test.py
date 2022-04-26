import numpy as np

def tournamentSelectionIndividual(sizeOfPopulation, k, scalarFitness):
    selected = np.random.randint(low=0, high=sizeOfPopulation, size=k)
    maxScalarFitness = np.max(scalarFitness[selected])
    print("gg", maxScalarFitness)
    print("hh", np.where(scalarFitness == maxScalarFitness)[0])
    result = np.random.choice(np.where(scalarFitness[selected] == maxScalarFitness)[0])
    print(selected)
    print("tt",scalarFitness[selected])
    return int(result)

print(tournamentSelectionIndividual(10, 5, np.array([3,5,2,1,4,5,2,3,5,1])))
print(tournamentSelectionIndividual(10, 5, np.array([3,5,2,1,4,5,2,3,5,1])))
print(tournamentSelectionIndividual(10, 5, np.array([3,5,2,1,4,5,2,3,5,1])))
print(tournamentSelectionIndividual(10, 5, np.array([3,5,2,1,4,5,2,3,5,1])))
print(tournamentSelectionIndividual(10, 5, np.array([3,5,2,1,4,5,2,3,5,1])))
print(tournamentSelectionIndividual(10, 5, np.array([3,5,2,1,4,5,2,3,5,1])))