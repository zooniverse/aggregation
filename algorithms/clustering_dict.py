__author__ = 'greg'
import agglomerative
clustering_dict = {}

# maps each sahpe to a clustering algorithm and any key word param
clustering_dict["seasons"] = {"point":(agglomerative.Agglomerative,{}),"ellipse":(agglomerative.Agglomerative,{})}