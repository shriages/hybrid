import matplotlib as mpl
import matplotlib.pyplot as plt

# Restore plt.register_cmap / plt.get_cmap removed in Matplotlib 3.9
if not hasattr(plt, "register_cmap"):
    def register_cmap(name=None, cmap=None, *, override_builtin=False, **kwargs):
        if name is None:
            name = cmap.name
        if cmap is not None and getattr(cmap, "name", None) != name:
            cmap = cmap.copy()
            cmap.name = name
        mpl.colormaps.register(cmap=cmap, name=name, force=True)
    plt.register_cmap = register_cmap

if not hasattr(plt, "get_cmap"):
    def get_cmap(name=None, lut=None):
        if name is None:
            name = mpl.rcParams["image.cmap"]
        if isinstance(name, mpl.colors.Colormap):
            return name
        cmap = mpl.colormaps[name]
        return cmap.resampled(lut) if lut is not None else cmap
    plt.get_cmap = get_cmap