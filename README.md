# kdtree
This is a Python implementation for constructing KD-Tree, searching the (k) nearest neighbors for any query point. The neighboring points can be used to calculate normal vector as feature representation to conduct tracking algorithms.

1). You need to have all the points stored in a .txt file as arranged in *.txt files under the test/ folder. For example, 

1.2,3.0,4.0<br />
7.8,9.0,2.0

represent two 3D points.

2). Currently, this program only supports 3D points. 

3). To test this program, simply run
 * $ python kdtree.py -i /path/to/pointcloud/.txt/file
