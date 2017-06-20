class Node(object):
  """ The Node object constructs each node in a KD tree 
  """
  def __init__(self, point):
    # Args: point: a 3-ele tuple
    assert len(point) == 3

    self.x = point[0]
    self.y = point[1]
    self.z = point[2]
    self.left = None
    self.right = None

  def get_position(self):
    # return position tuple
    return (self.x, self.y, self.z)

  def get_children(self):
    # return childen Node tuple
    return (self.left, self.right)