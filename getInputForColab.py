"""
To run in google colab, upload file and 
replace part of last code to get input from data in file by this code
"""

mecatDataFiles=['l100_4_20.test','l110_4_20.test','l120_4_20.test','l130_4_20.test','l140_4_20.test','l150_4_20.test','l160_4_20.test','l170_4_20.test','l180_4_20.test','l190_4_20.test',
                'm100_4_20.test','m110_4_20.test','m120_4_20.test','m130_4_20.test','m140_4_20.test','m150_4_20.test','m160_4_20.test','m170_4_20.test','m180_4_20.test','m190_4_20.test']

mecatDataFiles = sorted(mecatDataFiles,reverse=False)

for i in range(len(mecatDataFiles)):
    # task1 = getInputFromFile(mecatDataPath+'/'+mecatDataFiles[i])
    # task2 = getInputFromFile(mecatDataPath+'_rn/rn_'+mecatDataFiles[i])
    task1 = getInputFromFile(mecatDataFiles[i])
    task2 = getInputFromFile('rn_'+mecatDataFiles[i])
    tasks = list([task1, task2])
    print('Task 1 and 2 is from file: ', mecatDataFiles[i])
    resultPopulation,_ = mfea(tasks, 0.3, 350)
    print("-----")
    for i in range(len(resultPopulation)):
        print("Task", i+1)
        print(tasks[i].evaluateIndividualFactorialCost(resultPopulation[i]))
        print()
    print("-----")
    print()