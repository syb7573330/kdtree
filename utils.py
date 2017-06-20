def split_point(pt_str):
  """ Split a string of 'x,y,z' into a list of [x, y, z]
      Args:
        pt_str: "x,y,z"
  """
  res = pt_str.split(',')
  return [float(x) for x in res]


def read_points(pc_path):
  """ Read point cloud text file, where each line lists a point in the pattern 
      of 'x, y, z' 

      Args:
        pc_path: the path to point cloud file

      Returns:
        A list of tuples (x, y, z)
  """
  tmp_res = []
  with open(pc_path, 'r') as file:
    tmp_res = file.readlines()

  return [tuple(split_point(x)) for x in tmp_res]
