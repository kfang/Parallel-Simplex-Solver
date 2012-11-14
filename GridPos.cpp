
using namespace std;

class GridPos {
public:
	int row;
	int col;
	double val;
	GridPos(int, int, double);	
};

GridPos::GridPos (int r, int c, double v) {
	row = r;
	col = c;
	val = v;
}