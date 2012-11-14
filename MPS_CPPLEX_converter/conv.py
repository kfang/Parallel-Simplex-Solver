from array import array

data = []




file = open('sample.mps', 'r')

line = file.readline()

# for each line of the file, strip out the spaces and store them in data
while(line):
	splitLine = line.split(' ')

	noSpaceLine = []
	for item in splitLine:
		if (item != ''):
			noSpaceLine.append(item)
	data.append(noSpaceLine)
	line = file.readline()

# last element of data has '\n' in it, get rid of it
for arr in data:
	lastElement = arr[len(arr) - 1]
	arr[len(arr) - 1] = lastElement.split('\n')[0]
	print arr

# parse out data and store them into objects

print 'NAME: ' + name
def doName():
	for line in data:
		command = line[0]
		if (command == "NAME"):
			name = line[1]
			break

def doRows():
	start = False
	for line in data:
		command = line[0]
		if (start):
			if (command == 'N')
		if (command == "ROWS"):
			start = True
