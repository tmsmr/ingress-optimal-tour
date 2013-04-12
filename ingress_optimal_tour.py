__author__ = 'thomas@thomas-maier.net'

import sys, numpy, scipy.sparse.csgraph, random, pprint, copy, os

MAX_HACKS = 4
PORTAL_COOLDOWN = 300.0

POPSIZE = 30
ITERATIONS = 2000
CROSSPROP=0.99
MUTPROP=0.3

def main():
	infile, outfile, speed, evolutions = parseArguments()
	log('input-file: %s' % infile)
	log('output-file: %s' % outfile)
	log('speed: %skm/h' % speed)
	log('MAX_HACKS: %s' % MAX_HACKS)
	log('PORTAL_COOLDOWN: %s' % PORTAL_COOLDOWN)

	portalNames, weightMatrix = parseInputFile(infile)
	log('parsed portals:\n%s' % portalNames)
	log('parsed initial path distances(m):\n%s' % weightMatrix)

	weightMatrix = convertWeightMatrixToTime(weightMatrix, speed)
	log('converted path durations(s):\n%s' % weightMatrix)

	weightMatrix = populateWeightMatrix(weightMatrix)
	log('populated weight matrix(s):\n%s' % weightMatrix)

	portals =[]
	for i in xrange(MAX_HACKS):
		portals.extend(range(len(weightMatrix[0])))

	evolutionNr = 1
	while True:
		log('starting evolution %d' % evolutionNr)
		evolution(portals, weightMatrix, portalNames, outfile)
		if int(evolutions) != 0:
			if evolutionNr >= int(evolutions): break
		evolutionNr += 1

def log(string):
	print('\n[INGRESS-OPT] %s' % string)

def parseArguments():
	if len(sys.argv) != 5:
		print('\nusage: python %s inputfile outputfile speed(km/h) evolutions(0 for infinite)\n' % sys.argv[0])
		sys.exit(1)
	return sys.argv[1:]

def parseInputFile(filename):
	f = open(filename, 'r')
	content = f.readlines()
	portalNames = map(lambda x: x.strip(), content[0].split(',')[1:])
	weightMatrix = numpy.zeros((len(portalNames), len(portalNames)))
	for line in xrange(1,(len(portalNames) + 1)):
		for col in xrange(1,(len(portalNames) + 1)):
			if line == col:
				weightMatrix[line-1][col-1] = 0
				continue
			value = content[line].split(',')[col].strip()
			if len(value) == 0:
				weightMatrix[line-1][col-1] = 0
				continue
			weightMatrix[line-1][col-1] = parseDistance(content[line].split(',')[col].strip())
	return (portalNames, weightMatrix)

def parseDistance(dist):
	if dist[-2:] == 'km':
		return float(dist[:-2]) * 1000
	if dist[-1:] == 'm':
		return float(dist[:-1])

def convertWeightMatrixToTime(weightMatrix, speed):
	speed_m_s = float(speed)/3.6
	for line in xrange(len(weightMatrix)):
		for col in xrange(len(weightMatrix[line])):
			if line == col:
				continue
			if weightMatrix[line][col] != 0:
				weightMatrix[line][col] = float(weightMatrix[line][col])/speed_m_s
	return weightMatrix

def populateWeightMatrix(weightMatrix):
	for line in xrange(len(weightMatrix)):
		for col in xrange(len(weightMatrix[line])):
			if line == col:
				pass
			if weightMatrix[line][col] != 0:
				weightMatrix[col][line] = weightMatrix[line][col]
	weightMatrix = scipy.sparse.csgraph.shortest_path(weightMatrix, method='D', directed=False)
	for line in xrange(len(weightMatrix)):
		for col in xrange(len(weightMatrix[line])):
			if line == col:
				weightMatrix[line][col] = PORTAL_COOLDOWN
	return weightMatrix

def evolution(portals, weightMatrix, portalNames, outfile):
	log('creating %d chromosomes' % POPSIZE)
	population = numpy.zeros((POPSIZE, len(portals)))
	costs = numpy.zeros(POPSIZE)
	for i in xrange(POPSIZE):
		chromosom = list(portals)
		random.shuffle(chromosom)
		population[i] = chromosom

	log('starting %d iterations' % ITERATIONS)
	for i in xrange(ITERATIONS):
		if i % 100 == 0: sys.stdout.write(str(i))
		sys.stdout.write('.')
		sys.stdout.flush()
		for j in xrange(POPSIZE):
			costs[j] = roundtripDuration(population[j], weightMatrix)
		sortedIndex = costs.argsort(axis = 0)
		sortedCost = costs[sortedIndex]
		invertedCosts = 1 / sortedCost
		sortedPopulation = population[sortedIndex]

		##selection
		sel1 = -1
		sel2 = -1

		rand1 = invertedCosts.sum() * numpy.random.rand()
		j = 1
		while True:
			if rand1 < invertedCosts[:j].sum(axis=0):
				sel1 = j
				break
			j += 1
		sel1 -= 1

		while True:
			rand2 = invertedCosts.sum() * numpy.random.rand()
			j = 1
			while True:
				if rand2 < invertedCosts[:j].sum(axis=0):
					sel2 = j
					break
				j += 1
			sel2 -= 1
			if sel1 != sel2:
				break

		parent1 = sortedPopulation[sel1]
		parent2 = sortedPopulation[sel2]

		##crossing
		portalIndex2portalID = []
		for j in xrange(len(portals)):
			portalIndex2portalID.append([portals[j], 0])


		crossrn = numpy.random.rand()

		if crossrn < CROSSPROP:
			parent1_conv = convertForCrossing(copy.deepcopy(parent1), copy.deepcopy(portalIndex2portalID))
			parent2_conv = convertForCrossing(copy.deepcopy(parent2), copy.deepcopy(portalIndex2portalID))

			pos = int(numpy.ceil(numpy.random.rand()*len(parent1_conv)))

			head1 = parent1_conv[:pos]
			tailindex = 0
			tail1 = numpy.zeros(len(parent1_conv) - pos)
			for j in xrange(len(parent1_conv)):
				if parent2_conv[j] not in head1:
					tail1[tailindex] = parent2_conv[j]
					tailindex += 1
			child1 = numpy.append(head1, tail1)

			head2 = parent2_conv[:pos]
			tailindex = 0
			tail2 = numpy.zeros(len(parent1_conv) - pos)
			for j in xrange(len(parent1_conv)):
				if parent1_conv[j] not in head2:
					tail2[tailindex] = parent1_conv[j]
					tailindex += 1
			child2 = numpy.append(head2, tail2)

			child1 = convertFromCrossing(copy.deepcopy(child1), portalIndex2portalID)
			child2 = convertFromCrossing(copy.deepcopy(child2), portalIndex2portalID)

		else:
			child1 = copy.deepcopy(parent1)
			child2 = copy.deepcopy(parent2)

		#mutation
		mutrand = numpy.random.rand()
		if mutrand < MUTPROP:
			mutIndex = numpy.ceil(numpy.random.rand(2)*(len(parent1) - 1))
			first = child1[mutIndex[0]]
			second = child1[mutIndex[1]]
			child1[mutIndex[0]] = second
			child1[mutIndex[1]] = first

		mutrand = numpy.random.rand()
		if mutrand < MUTPROP:
			mutIndex = numpy.ceil(numpy.random.rand(2)*(len(parent1) - 1))
			first = child2[mutIndex[0]]
			second = child2[mutIndex[1]]
			child2[mutIndex[0]] = second
			child2[mutIndex[1]] = first

		#selection
		costChild1 = roundtripDuration(child1, weightMatrix)
		costChild2 = roundtripDuration(child2, weightMatrix)

		replace1 = False
		replace2 = False
		index = POPSIZE-1

		while index > 0:
			if sortedCost[index] > costChild1 and not replace1:
					sortedPopulation[index] = child1
					replace1 = True
			elif sortedCost[index] > costChild2 and not replace2:
					sortedPopulation[index] = child2
					replace2 = True
			if replace1 and replace2:
				break
			index -= 1
		population = sortedPopulation

	log('best result:\n%s' % population[0])
	log('length: %s' % roundtripDuration(population[0], weightMatrix))
	saveResult(roundtripDuration(population[0], weightMatrix), population[0], portalNames, outfile)

def saveResult(duration, portals, portalNames, outfile):
	if not os.path.exists(outfile):
		os.system('echo -1 > %s' % outfile)
	f = open(outfile, 'r')
	lastBestResult = int(f.readline())
	f.close()
	if duration < lastBestResult or lastBestResult == -1:
		log('NEW BEST RESULT!')
		f = open(outfile, 'w')
		f.write('%d\n' % int(duration))
		for i in  xrange(len(portals)):
			f.write('%s\n' % portalNames[int(portals[i])])
		f.close()


def convertForCrossing(chromosom, index2portal):
	filter = index2portal
	for i in xrange(len(chromosom)):
		for j in xrange(len(filter)):
			if chromosom[i] == filter[j][0] and filter[j][1] == 0:
				chromosom[i] = j
				filter[j][1] = 1
				break
	return chromosom

def convertFromCrossing(chromosom, index2portal):
	for i in xrange(len(chromosom)):
		chromosom[i] = index2portal[int(chromosom[i])][0]
	return chromosom

def roundtripDuration(nptour, weightMatrix):
	tour = list(nptour)
	weights = []
	for i in xrange(len(tour) - 1):
		weight = weightMatrix[tour[i]][tour[i + 1]]
		lastSame = 0
		j = i - 1
		while True:
			if j < 0:
				break
			lastSame += weights[j]
			if lastSame + weight >= PORTAL_COOLDOWN:
				break
			if tour[i + 1] == tour[j]:
				weight = PORTAL_COOLDOWN - lastSame
				break
			j -= 1
		weights.append(weight)
	return sum(weights)

if __name__ == '__main__':
	main()