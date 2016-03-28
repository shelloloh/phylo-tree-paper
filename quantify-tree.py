__author__ = 'Jonathan Lee'
# import the needed modules
from ete2 import Tree
import csv
import numpy
from sklearn import preprocessing

def preparedata(treefile = "tree/000.nwk", groupfile = 'tree/group.csv', spacetimefile = "tree/location_year.csv"):

	# open the tree structure file
	t = Tree(treefile)
	rootNode = t.children[1]

	# import the group file and convert it to dictionary
	with open(groupfile) as f:
		reader = csv.reader(f)
		groupList = list(reader)
	dictGroup = dict(groupList)

	#import location and time file and convert it to dictionary
	with open(spacetimefile) as f:
		reader = csv.reader(f)
		spaceTimeList = list(reader)
	dictSpaceTime = {}
	for i in range(len(spaceTimeList)):
		spaceTimeList[i][1] = str(spaceTimeList[i][1].split("/")[0])
		dictSpaceTime[spaceTimeList[i][0]] = [spaceTimeList[i][1],spaceTimeList[i][2],spaceTimeList[i][3]]

	# assign the features to each leaf
	names = []
	for leaf in t:
		if leaf.name == "AF117241":
			leaf.add_feature("groupID", 99)
			leaf.add_features(groupID = 99, year = int(dictSpaceTime["AF117241"][0]), latitude = float(dictSpaceTime["AF117241"][1]), longitude = float(dictSpaceTime["AF117241"][2]))
		else:
			leaf.add_features(groupID = int(dictGroup[leaf.name]), year = int(dictSpaceTime[leaf.name][0]), latitude = float(dictSpaceTime[leaf.name][1]), longitude = float(dictSpaceTime[leaf.name][2]))
	t.set_outgroup(t&"AF117241")

	# generate regular variable names
	groups = []
	for i in range(1, 15):
		locals()["group"+str(i)] = t.search_nodes(groupID = i)
		groups.append(locals()["group" + str(i)])
	return t, groups, t&"AF117241"


# define the function which build the node matrix
def buildnodematrix(tree):
	nodematrix = []
	for leaf in tree:
		ancestors = leaf.get_ancestors()
		ancestors.insert(0, leaf)
		nodematrix.append(ancestors[::-1])
	for i in nodematrix:
		i.insert(0, rootnode)
	return nodematrix


# define the function which transform the node matrix into distance matrix
def builddistmatrix(nodematrix):
	distmatrix = []
	for node in nodematrix:
		nodevector=[]
		for obj in range(0,len(node)-1):
			nodevector.append(node[obj].get_distance(node[obj + 1]))
		distmatrix.append(nodevector)
	return distmatrix

def existmatrix(distmatrix):
	existmatrix = []
	for vector in distmatrix:
		addon = [1 for i in vector]
		addon += [0,] * (87-len(addon))
		existmatrix.append(addon)
	return existmatrix

def newdistmatrix(distmatrix):
	newdist = []
	for i in distmatrix:
		i += [0,] * (87 - len(i))
		newdist.append(i)
	return newdist

def spacematrix(tree):
	spacematrix = []
	for leaf in tree:
		spacematrix.append([leaf.latitude, leaf.longitude])
	spacematrix = numpy.array(spacematrix).T
	covmatrix = numpy.cov(spacematrix)
	result = sum(sum(covmatrix))
	vector = [result,] * 87
	finalmatrix = []
	for leaf in tree:
		finalmatrix.append(vector)
	return finalmatrix


def timematrix(tree):
	earlyyear = []
	lateyear = []
	for leaf in tree:
		earlyyear.append(leaf.year)
		lateyear.append(leaf.year)
	earlyyear.pop(0)
	early = min(earlyyear)
	late = max(lateyear)
	gap = late - early
	result = gap/91.0
	vector = [result,] * 87
	finalmatrix = []
	for leaf in tree:
		finalmatrix.append(vector)
	return finalmatrix


tree, groups, rootnode = preparedata()

allinone = []
for group in groups:
	nodematrix = buildnodematrix(group)
	distmatrix = builddistmatrix(nodematrix)
	exist = existmatrix(distmatrix)
	newdist = newdistmatrix(distmatrix)
	space = spacematrix(group)
	time = timematrix(group)
	scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
	finaldist = numpy.array(exist) * numpy.array(newdist)
	finalspace = numpy.array(exist) * numpy.array(space)
	finaltime = numpy.array(exist) * numpy.array(time)
	finalspace = scaler.fit_transform(finalspace)
	finaldist = scaler.fit_transform(finaldist)
	wholematrix = 0.4 * finaldist + 0.4 * finalspace + 0.2 * finaltime
	names = []
	for i in group:
		names.append(i.name)
	wholematrix = list(wholematrix.T)
	wholematrix.append(names)
	wholematrix = numpy.array(wholematrix)
	wholematrix = list(wholematrix.T)
	allinone += wholematrix

with open("final.csv","w") as f:
	k = csv.writer(f)
	k.writerows(allinone)



