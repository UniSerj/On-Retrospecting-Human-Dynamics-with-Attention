from __future__ import print_function
import h5py
import numpy as np

#### print the average error saved in the sample file###
dataset = 'human3.6m'
# dataset = 'cmu'


# path='./samples_SW/'
path = './'
if dataset=='human3.6m':
    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]
else:
    actions = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running",
               "soccer", "walking", "washwindow"]
time = [1,3,7,9,24]
errors=[]

print()
print("{0: <16} |".format("milliseconds"), end="")
for ms in [80, 160, 320, 400, 1000]:
    print(" {0:5d} |".format(ms), end="")
print()


with h5py.File(path+'samples.h5', 'r' ) as h5f:
    for action in actions:
        error=h5f['mean_{}_error'.format(action)][:]
        print("{0: <16} |".format(action), end="")
        for ms in time:
            print(" {0:.3f} |".format(error[ms]), end="")
        print()
        errors.append(error)
errors=np.array(errors)
error_ave=np.mean(errors,axis=0)
print("{0: <16} |".format("average"), end="")
for ms in time:
    print(" {0:.3f} |".format(error_ave[ms]), end="")
print()


