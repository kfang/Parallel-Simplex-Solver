import sys
from random import randint

def main():
	num_rows = int(sys.argv[2])
	num_cols = int(sys.argv[3])
	name = sys.argv[1]
	print "NAME          " + name
	print "ROWS"
	print " N  profit"
	for i in range(num_rows):
		print " L  r" + str(i)
	print "COLUMNS"
	percent_done = 0
	for i in range(num_cols):
		obj = "    c" + str(i)
		obj = spaces(obj, 14)
		obj = obj + "profit"
		obj = spaces(obj, 35)
		obj = obj + str(randint(1,9))
		print obj
		if ((i/num_cols) >= ((percent_done+10)/100)):
			percent_done += 10
			print "Percent Done: " + str(percent_done) + "%"
		for j in range(num_rows):
			l = "    c" + str(i)
			l = spaces(l, 14)
			l = l + "r" + str(j)
			l = spaces(l, 35)
			l = l + str(randint(1,9))
			print l
	print "RHS"
	for i in range(num_rows):
		l = "    rhs"
		l = spaces(l, 14)
		l = l + "r" + str(i)
		l = spaces(l, 35)
		l = l + str(randint(1,9))
		print l
	print "ENDATA"

def spaces(l, i):
	while(len(l) < i):
		l = l + " "
	return l

if __name__ == '__main__':
    main()