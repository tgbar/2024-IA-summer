import numpy as np
from getdist import MCSamples, plots
# conda install -c conda-forge getdist


# np.save("myname.npy", samples)
samples = np.load("sample-test.npy")
labels = ["Om", "h"]

# use getdist
#
samples_g = MCSamples(samples=samples, names=labels, labels=labels, settings=dict(smooth_scale_2D=3))

g = plots.get_subplot_plotter()
g.triangle_plot([samples_g], filled=True, param_limits={'h':(10.0,30.0),'Om':(0.1,0.4)})

print(samples_g.getInlineLatex('h', limit=1))
print(samples_g.getInlineLatex('Om', limit=1))

g2 = plots.get_single_plotter()
g2.plot_1d(samples_g,'Om')