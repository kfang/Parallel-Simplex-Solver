from array import array

data = []

file = open('sample.mps', 'r')

line = file.readline()

while(line):
	splitLine = line.split(' ')

	noSpaceLine = []
	for item in splitLine:
		if (item != ''):
			noSpaceLine.append(item)
	data.append(noSpaceLine)
	line = file.readline()

for arr in data:
	print arr
