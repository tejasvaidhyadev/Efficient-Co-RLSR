from scipy.stats import wilcoxon
def wilcoxon_test(x, y = None):
    if y == None:
        w,p = wilcoxon(x) # incase difference is provided
    else:
        w,p = wilcoxon(x,y)
    return w,p