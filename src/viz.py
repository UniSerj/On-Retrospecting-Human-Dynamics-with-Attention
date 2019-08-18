"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import matplotlib.markers as style
import matplotlib.cm as heatmap
import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

class Ax3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
    self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1

    # self.K   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
    self.K   = np.array([3,4,8,9,13,14,15,16,18,19,20,26,27,28])-1



    # Left / right indicator
    self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # self.I   = np.array([1,2,3,4,5,1,7,8, 9,10,1, 13,14,15,14,18,19,20,21,21,21,14,26,27,28,29,29])-1
    # self.J   = np.array([2,3,4,5,6,7,8,9,10,11,13,14,15,16,18,19,20,21,22,23,24,26,27,28,29,30,31])-1
    # self.K   = np.array([2,3,4,5,7,8,9,10,13,14,15,18,19,20,26,27,28])
    # self.LR = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],dtype=bool)

    self.ax = ax

    vals = np.zeros((32, 3))

    # Make connection matrix
    self.plots = []
    self.circles = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    for i in np.arange( len(self.K) ):
      x_dot = vals[self.K[i]][0]
      y_dot = vals[self.K[i]][1]
      z_dot = vals[self.K[i]][2]
      r = 0.1
      circle_style = style.MarkerStyle(marker='o',fillstyle='full')
      circle = ax.scatter(x_dot, y_dot, z_dot, s=np.pi*r**2*0, c='#d8ac08', alpha=1.0, marker=circle_style)
      self.circles.append(circle)

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, weights, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (32, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
    if len(weights) != 0:
      weights = np.reshape( weights, (32, -1) )
      sw = []

      for j in self.K:
        sw.append(np.sum(weights[j-1]))

      cmap = heatmap.get_cmap('autumn_r')
      # max = np.max(sw)

      tmp = np.exp(sw)
      sum = np.sum(tmp)
      max = np.max(tmp)
      tmp = tmp/sum
      # print(tmp)
      rgba = cmap(tmp/np.max(tmp)*0.8+0.2)

      # rgba = cmap(sw/max)
      # print(sw)
      
      for j in range(0, len(sw)):
        sw[j] = sw[j]*20

      # cmap = heatmap.get_cmap('autumn_r')
      # max = np.max(sw)
      # rgba = cmap(sw/max)

      for i in np.arange( len(self.K) ):
        x_dot = vals[self.K[i]][0]
        y_dot = vals[self.K[i]][1]
        z_dot = vals[self.K[i]][2]

        x_dot = tuple([x_dot])
        y_dot = tuple([y_dot])
        z_dot = np.array([z_dot])

        self.circles[i]._offsets3d = (x_dot,y_dot,z_dot)
        self.circles[i].set_sizes(np.array([sw[i]]))
        max = np.max(sw)

        # self.circles[i].set_alpha(sw[i]/max)
        self.circles[i].set_color(rgba[i])

    r = 750;
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')
