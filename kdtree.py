import argparse
import copy
import utils
import numpy as np
from node import Node
from bisect import bisect

class KDTree(object):
  """Construct a kd tree"""
  def __init__(self, file_path):
    """
      Args:
        file_path: a path to a text file that stores points 
    """
    self.pc_path = file_path
    self.dim = 3

    # ------------
    # Build a tree
    # ------------
    # 1. Initialize all points
    points = utils.read_points(self.pc_path)
    points = np.array(points, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    self.points = np.unique(points)

    # 2. Build the tree
    index_x = np.argsort(self.points, order=('x','y','z'))
    index_y = np.argsort(self.points, order=('y','z','x'))
    index_z = np.argsort(self.points, order=('z','x','y'))
    self.root = self.build_tree(self.points, index_x, index_y, index_z, 0)

  def partition(self, points, index, median_point_index, axis):
    """ Partition the index array according to 'level' or 'axis'
        Args:
          points: a np.ndarray of 3-ele tuples
          index: the index array to be partitioned accroding to 'axis'
          median_point_index: the index of partitioning point
          axis: the reference partitioning axis
        Returns:
          a lower index array and an upper index array
    """
    median_point = points[median_point_index]
    index_lower = []
    index_upper = []
    for i in index:
      if i == median_point_index:
        continue
      if points[i][axis] < median_point[axis]:
        index_lower.append(i)
      else:
        index_upper.append(i)
    return index_lower, index_upper

  def build_tree(self, points, index_x, index_y, index_z, level):
    """ Recursively build a kd tree

        Args:
          points: a np.ndarray of 3-ele tuples
          index_x: the sorted index of points in the order of superkey 'x-y-z'
          index_y: the sorted index of points in the order of superkey 'y-z-x'
          index_z: the sorted index of points in the order of superkey 'z-x-y'
          level: The current depth in the tree

        Returns:
          The root Node
    """
    # print (index_x, index_y, index_z)

    assert len(index_x) == len(index_y) == len(index_z)
    length = len(index_x)

    if length == 0:
      return None
    elif length == 1:
      # If length is 1, index_x, index_y and index_z contain the same index 
      return Node(points[index_x[0]])
    else:
      # Partition accordint to x-axis
      if level == 0: 
        # partition 3 index arrays according to the median of index_x
        median_point_index = index_x[length/2]
        # partition x index
        index_x_lower = copy.deepcopy(index_x[:length/2])
        index_x_upper = copy.deepcopy(index_x[length/2+1:])
        # partition y index
        index_y_lower, index_y_upper = self.partition(points, index_y, median_point_index, level)
        # partition z index
        index_z_lower, index_z_upper = self.partition(points, index_z, median_point_index, level)
      
      # Partition accordint to y-axis
      elif level == 1:
        median_point_index = index_y[length/2]
        # partition y index
        index_y_lower = copy.deepcopy(index_y[:length/2])
        index_y_upper = copy.deepcopy(index_y[length/2+1:])
        # partition z index
        index_z_lower, index_z_upper = self.partition(points, index_z, median_point_index, level)
        # partition x index
        index_x_lower, index_x_upper = self.partition(points, index_x, median_point_index, level)

      # Partition accordint to z-axis
      elif level == 2:
        median_point_index = index_z[length/2]
        # partition z index
        index_z_lower = copy.deepcopy(index_z[:length/2])
        index_z_upper = copy.deepcopy(index_z[length/2+1:])
        # partition x index
        index_x_lower, index_x_upper = self.partition(points, index_x, median_point_index, level)
        # partition y index
        index_y_lower, index_y_upper = self.partition(points, index_y, median_point_index, level)

      del index_x, index_y, index_z # Avoid memory increasing during recurssion
      res = Node(points[median_point_index])
      res.left = self.build_tree(points, index_x_lower, index_y_lower, index_z_lower, (level+1)%self.dim)
      res.right = self.build_tree(points, index_x_upper, index_y_upper, index_z_upper, (level+1)%self.dim)
      return res

  def nearest_neighbor(self, query, curr_node, level):
    """ Return the nearest point from self.points for query point
        Args:
          query: the query point (a tuple)
          curr_node: the current node to be processed
          level: current condition axis (0:x, 1:y, 2:z)

        Returns:
          The cloest points (if they have the same distance to the query point) in the kd tree
    """

    # leaf node:
    if curr_node.left is None and curr_node.right is None:
      sqr_dis = np.sum((np.array(query) - np.array(curr_node.get_position())) ** 2)
      return [curr_node], sqr_dis

    # Current node has at least one child
    curr_best_nodes = [curr_node] # there may be multiple best nodes
    curr_best_sqr_dis = np.sum((np.array(query) - np.array(curr_node.get_position())) ** 2)

    # left branch if possible
    if curr_node.left is not None:
      best_left_children, best_left_sqr_dis = self.nearest_neighbor(query, curr_node.left, (level+1)%self.dim)
      if best_left_sqr_dis < curr_best_sqr_dis:
        curr_best_nodes = best_left_children
        curr_best_sqr_dis = best_left_sqr_dis
      elif best_left_sqr_dis == curr_best_sqr_dis:
        curr_best_nodes += best_left_children

    # right branch
    if curr_node.right is not None:
      # calculate the axis-distance
      axis_sqr_dis = (query[level] - curr_node.get_position()[level]) ** 2
      if axis_sqr_dis < curr_best_sqr_dis:
        # intersected, there could be closer points on the other branch
        best_right_children, best_right_sqr_dis = self.nearest_neighbor(query, curr_node.right, (level+1)%self.dim)
        if best_right_sqr_dis < curr_best_sqr_dis:
          curr_best_nodes = best_right_children
          curr_best_sqr_dis = best_right_sqr_dis
        elif best_right_sqr_dis == curr_best_sqr_dis:
          curr_best_nodes += best_right_children

    return curr_best_nodes, curr_best_sqr_dis

  def k_nearest_neighbors(self, query, curr_node, k, level):
    """ Find the cloest k points to the query point
        Args:
          query: The query point
          curr_node: The current node to be processed
          k: the number of nearest neighbors to be returned
          level: current condition axis (0:x, 1:y, 2:z)
        Returns:
          A list of k nearest nodes if this tree contains at least k nodes,
          o.w. return all the nodes in the tree
    """
    # leaf node:
    if curr_node.left is None and curr_node.right is None:
      sqr_dis = np.sum((np.array(query) - np.array(curr_node.get_position())) ** 2)
      return [curr_node], [sqr_dis]

    # Current node has at least one child
    curr_best_nodes = [curr_node] # there may be multiple best nodes
    curr_best_sqr_dis = [np.sum((np.array(query) - np.array(curr_node.get_position())) ** 2)]

    # left branch if possible
    if curr_node.left is not None:
      # return a list of at most k nodes, arranged as the distances changing from smallest distance in best_left_sqr_dis
      best_left_children, best_left_sqr_dis = self.k_nearest_neighbors(query, curr_node.left, k, (level+1)%self.dim)
      # print ('left children before')
      # for node in best_left_children:
      #   print node.get_position()

      # The current node should be inserted into the best nodes so far
      if len(best_left_children) < k or curr_best_sqr_dis[0] < best_left_sqr_dis[-1]:
        idx = bisect(best_left_sqr_dis, curr_best_sqr_dis[0])
        best_left_sqr_dis.insert(idx, curr_best_sqr_dis[0])
        best_left_children.insert(idx, curr_best_nodes[0])
      # print ('left children after')
      # for node in best_left_children:
      #   print node.get_position()

      # take the first k (or less) elements
      curr_best_nodes = best_left_children[:k]
      curr_best_sqr_dis = best_left_sqr_dis[:k]

    
    # right branch if possible
    if curr_node.right is not None:
      # calculate the axis-distance
      axis_sqr_dis = (query[level] - curr_node.get_position()[level]) ** 2
      # need to search right branch, o.w. go upwind
      if len(curr_best_nodes) < k or curr_best_sqr_dis[-1] > axis_sqr_dis:
        # Have to find the k best nodes from right branch to guarantee return k best nodes from this subtree
        best_right_children, best_right_sqr_dis = self.k_nearest_neighbors(query, curr_node.right, k, (level+1)%self.dim)
        # current_best_nodes and best_right_children can both contain 1 to k elements, need to merge them 
        tmp_dis = []
        tmp_nodes = []
        curr_idx = 0
        right_idx = 0
        while right_idx<len(best_right_sqr_dis) and curr_idx<len(curr_best_sqr_dis) and len(tmp_dis)<k:
          if best_right_sqr_dis[right_idx] < curr_best_sqr_dis[curr_idx]:
            tmp_dis.append(best_right_sqr_dis[right_idx])
            tmp_nodes.append(best_right_children[right_idx])
            right_idx += 1
          else:
            tmp_dis.append(curr_best_sqr_dis[curr_idx])
            tmp_nodes.append(curr_best_nodes[curr_idx])
            curr_idx += 1

        # The tmp size of the array just filled
        tmp_size = len(tmp_dis)
        if tmp_size < k:
          # need to fill from non-empty array
          if curr_idx == len(curr_best_sqr_dis):
            # curr_best_sqr_dis equals to 'empty'
            tmp_dis += best_right_sqr_dis[right_idx:min(right_idx+k-tmp_size,len(best_right_sqr_dis))]
            tmp_nodes += best_right_children[right_idx:min(right_idx+k-tmp_size,len(best_right_children))]
          elif right_idx == len(best_right_sqr_dis):
            tmp_dis += curr_best_sqr_dis[curr_idx:min(curr_idx+k-tmp_size,len(curr_best_sqr_dis))]
            tmp_nodes += curr_best_nodes[curr_idx:min(curr_idx+k-tmp_size,len(curr_best_nodes))]

        curr_best_nodes = tmp_nodes
        curr_best_sqr_dis = tmp_dis

    return curr_best_nodes, curr_best_sqr_dis



def main(pc_path):
  # Read points as a list of tuples
  tree = KDTree(pc_path)

  # Find the cloest neighbors
  nn,_ = tree.nearest_neighbor((7.0, 6.0, 1.0), tree.root, 0)
  print ('The cloest neighbor for query point (7.0, 6.0, 1.0):')
  for node in nn:
    print node.get_position()

  print ('\n')
  
  # Find the k nearest neighbors
  k = 3
  print ('The %d nearest neighbors for query point (8.0, 3.0, 1.0):' % k)
  nn, _ = tree.k_nearest_neighbors((8.0, 3.0, 1.0), tree.root, k, 0)
  for node in nn:
    print node.get_position()





if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Construct a KD-tree from a list of 3D points")
  parser.add_argument('-i', '--input', metavar='', type=str, help="The path to input point cloud file", required=True)
  # parser.add_argument('-o', '--output', metavar='', type=str, help="The path to output folder", required=True)
  args = parser.parse_args()
  main(args.input)
