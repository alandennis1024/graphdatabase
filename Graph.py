# Databricks notebook source
# MAGIC %md 
# MAGIC # What is a Graph?
# MAGIC For our purposes we are discussing directed graphs.  More generally, graphs are not necessarily directed.
# MAGIC A collection of vertices/nodes and edges connecting them.
# MAGIC 
# MAGIC ![](https://raw.githubusercontent.com/alandennis1024/graphdatabase/main/images/SimpleGraph.png)
# MAGIC 
# MAGIC Graph characteristics:
# MAGIC - Vertices are items of interst. 
# MAGIC - Edges are relationships between them. 
# MAGIC - Edges have a direction. They start at one vertex and terminate at another.
# MAGIC - Attributes are on both vertices and edges.  Examples include id, name, weight, and so on.
# MAGIC 
# MAGIC For a detailed discussion of graphs and graph theory -  
# MAGIC Cusack, C. A., & Santos, D. A. (2021). An Active Introduction to Discrete Mathematics and Algorithms. GNU. https://cusack.hope.edu/Notes/?Instructor=Books 

# COMMAND ----------

# MAGIC %md
# MAGIC # Why use a graph processing system?
# MAGIC - What you are modling is a network or tree (the data is a graph) 
# MAGIC - Classis example is links in the web, useful to calculate PageRank
# MAGIC ## I could do that in a relational database, right?
# MAGIC You could, but it is about using the tool that can address the problem best. Graph processing systems are optimized for graphs.
# MAGIC ## Calculate things like
# MAGIC - Longest path
# MAGIC - Shortest path
# MAGIC - Least expensive path (if edges have costs/weights)
# MAGIC - Vertex with most outgoing edges
# MAGIC - Vertex with most incoming edges
# MAGIC - Vertex with most overall connections (in and out)
# MAGIC - In some cases, Visual Exploration of Data (easy to see clusters/hotspots)

# COMMAND ----------

# MAGIC %md 
# MAGIC # What is GraphFrames
# MAGIC - Combination of graph processing from GraphX with Spark DataFrames
# MAGIC - Essentially contains two DataFrames, one for edges and one for verticies

# COMMAND ----------

v = spark.sql("select * from vertices")
display(v)

# COMMAND ----------

# MAGIC %md
# MAGIC GraphFrames require an id column to uniquely identify a given vertex.

# COMMAND ----------

v = v.withColumnRenamed("Node","id")
display(v)

# COMMAND ----------

e = spark.sql("select * from edges")
display(e)

# COMMAND ----------

# MAGIC %md
# MAGIC GraphFrame expects two columns in the dataset that contains the edges. They need to be named src (for the source of the edge) and dest (for the sink of the edge).

# COMMAND ----------

e = e.withColumnRenamed("From","src").withColumnRenamed("To","dst")
display(e)

# COMMAND ----------

from graphframes import *
g = GraphFrame(v, e)

# COMMAND ----------

display(g.edges)

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

# MAGIC %md
# MAGIC # Example - Prerequisites in undergraduate programs
# MAGIC Typically, classes in undergraduate are gated by prerequisites.
# MAGIC - Example: Video Game Programming requires completion of OO Programming
# MAGIC - One class often unlocks multiple higher numbered classes
# MAGIC - Important for students to take the right classes (and gain the right knowledge) at the right time

# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt
def PlotDag(graph):
  G = nx.DiGraph()
  for x in graph.edges.collect():
    src = x["src"]
    dst = x["dst"]
    G.add_edge(src,dst)
  plt.figure(1,figsize=(8,8)) 
  pos = nx.spiral_layout(G)
  nx.draw(G,pos=pos,with_labels = True, arrows = True,edge_color = 'b', arrowstyle='fancy' )
  plt.show()
def PlotGraph(graph):
  G = nx.from_pandas_edgelist(graph.edges.toPandas(),'src','dst')
  plt.figure(1,figsize=(8,8)) 
  nx.draw(G,with_labels = True, arrows = True, arrowsize=20,edge_color = 'b', arrowstyle='fancy' )
  plt.show()
#PlotGraph(g)
PlotDag(g)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Neo4j visualization of the graph
# MAGIC ![](https://raw.githubusercontent.com/alandennis1024/graphdatabase/main/images/neo4jgraph.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph Algorithms
# MAGIC ## Triangle Count
# MAGIC Display the nodes that are part of triangles. Triangles are three nodes where all nodes have a relationship with each other.

# COMMAND ----------

display(g.triangleCount().filter("count >0"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Graph queries
# MAGIC ### inDegree - sum of the edges ending at each vertex

# COMMAND ----------

from pyspark.sql.functions import desc
display(g.inDegrees.join(g.vertices,"id").orderBy(desc("inDegree")))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### outDegree - sum of the edges starting at each vertex

# COMMAND ----------

display(g.outDegrees.join(g.vertices,"id").orderBy(desc("outDegree")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### degree - Total edges starting or ending at a vertex

# COMMAND ----------

display(g.degrees.join(g.vertices,"id").orderBy(desc("degree")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph Algorithms

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connected Components
# MAGIC From GraphFrames documentation 
# MAGIC 
# MAGIC         Computes the connected component membership of each vertex and returns a graph with each vertex assigned a component ID.

# COMMAND ----------

!mkdir /tmp/checkpoints

# COMMAND ----------

# must set checkpoing directory before calling connectedComponents
sc.setCheckpointDir("/tmp/checkpoints")

# COMMAND ----------

from pyspark.sql.functions import col
result = g.connectedComponents()
display(result)

# COMMAND ----------

groupedByComponent = result.groupBy(col("component")).count().orderBy(desc("count"))
display(groupedByComponent)

# COMMAND ----------

disconnectedVertices =groupedByComponent.filter("count = 1").join(g.vertices,g.vertices.id ==groupedByComponent.component ) 
display(disconnectedVertices)

# COMMAND ----------

# MAGIC %md 
# MAGIC We have several isollated nodes. We can remove them with dropIsolatedVertices()

# COMMAND ----------

justConnectedG =g.dropIsolatedVertices()
display(justConnectedG.vertices)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Strongly connected component
# MAGIC From Wikipedia - In the mathematical theory of directed graphs, a graph is said to be strongly connected if every vertex is reachable from every other vertex. 

# COMMAND ----------

 from pyspark.sql.functions import col
# Compute the strongly connected component (SCC) of each vertex and return a graph with each vertex assigned to the SCC containing that vertex.
result = g.stronglyConnectedComponents(maxIter=5)
#result.select("id", "Name").orderBy("component").show()
display(result.groupBy(col("component")).count().orderBy(desc("count")))
#Note there are no vertices that are part of a subgraph that are strongly connected

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Breadth-first search (BFS)
# MAGIC Breadth-first search (BFS) finds the shortest path(s) from one vertex (or a set of vertices) to another vertex (or a set of vertices). The beginning and end vertices are specified as Spark DataFrame expressions.

# COMMAND ----------

paths = g.bfs("Name = 'ITSS 230'", "Name ='ITSS 439'")
display(paths)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## PageRank
# MAGIC Run the Google PageRank algorithm on the graph to determine most important vertices

# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices.select("id","pagerank").join(g.vertices,"id").orderBy(desc("pagerank")))


# COMMAND ----------

display(results.edges.select("src", "dst", "weight").orderBy(desc("weight")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Label Proagation Algorithm (LPA)
# MAGIC 
# MAGIC Run static Label Propagation Algorithm for detecting communities in networks.
# MAGIC 
# MAGIC Each node in the network is initially assigned to its own community. At every superstep, nodes send their community affiliation to all neighbors and update their state to the **mode** community affiliation of incoming messages.
# MAGIC 
# MAGIC LPA is a standard community detection algorithm for graphs. It is very inexpensive computationally, although (1) convergence is not guaranteed and (2) one can end up with trivial solutions (all nodes are identified into a single community).

# COMMAND ----------

g2 = GraphFrame(v.withColumnRenamed("Label","Classname"), e)
result = g2.labelPropagation(maxIter=5)
print(result.columns)
display(result.orderBy(col('label')))
#result.select("id", "label").show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Motif finding
# MAGIC A way to look for patterns in a graph.
# MAGIC GraphFrames use a domain specific langauge (DSL) to express structural queries. 
# MAGIC 
# MAGIC General form is (a)-[e]->(b)  where a and b are vertices and e is an edge connecting them.
# MAGIC 
# MAGIC Results can be combined with filter clause to restrict further, such as .filter("a.id != b.id") to remove selfloops

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check for two node loops

# COMMAND ----------

motifs = g.find("(a)-[ab]->(b); (b)-[ba]->(a)")
display(motifs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check for three node loops

# COMMAND ----------

motifs = g.find("(a)-[ab]->(b); (b)-[bc]->(c);(c)-[ca]->(a)")
display(motifs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find chains of 2 vertices

# COMMAND ----------

chain2 = g.find("(a)-[ab]->(b)")
display(chain2)

# COMMAND ----------

# MAGIC %md
# MAGIC Since result of find is a DataFrame we can apply standard constructs to it.

# COMMAND ----------

from pyspark.sql.functions import col
display(chain2.select(col("a.Name").alias("PreReqCode"),col("a.Label").alias("PreReqName"),col("b.Name").alias("ClassCode"),col("b.Label").alias("ClassName")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find chains of 3 vertices.

# COMMAND ----------

chain3 = g.find("(a)-[ab]->(b); (b)-[bc]->(c)")
display(chain3)

# COMMAND ----------

display(chain3.select("a.Label", "b.Label", "c.Label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find chains of 4 vertices.

# COMMAND ----------

chain4 = g.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d)")
display(chain4)

# COMMAND ----------

display(chain4.select("a.Label", "b.Label", "c.Label","d.Label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find chains of 5 vertices.

# COMMAND ----------

chain5 = g.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d); (d)-[de]->(e)")
display(chain5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find chains of 6 vertices.

# COMMAND ----------

chain6 = g.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d); (d)-[de]->(e); (e)-[ef]->(f)")
display(chain6)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find chains of 5 vertices that end with Capstone II.

# COMMAND ----------

chain5Filter = g.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d); (d)-[de]->(e)").filter("e.Name = 'ITSS 439'")
display(chain5Filter)

# COMMAND ----------

display(chain5Filter.select("a.Label", "b.Label", "c.Label","d.Label", "e.Label"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Where to learn more
# MAGIC - GraphFrames documentation https://graphframes.github.io/graphframes/docs/_site/index.html
# MAGIC - GraphX documentation https://spark.apache.org/docs/latest/graphx-programming-guide.html

# COMMAND ----------

# MAGIC %md 
# MAGIC # Pizza Example
# MAGIC ![](https://raw.githubusercontent.com/alandennis1024/graphdatabase/main/images/PizzaState.jpg)

# COMMAND ----------

pizzav = spark.sql("select * from pizzav")
pizzae = spark.sql("select * from pizzae")
pizzaGraph = GraphFrame(pizzav,pizzae)
PlotGraph(pizzaGraph)

# COMMAND ----------

display(pizzaGraph.vertices)

# COMMAND ----------

display(pizzaGraph.edges)


# COMMAND ----------

for x in pizzaGraph.vertices.collect():
  print(x.Name)


# COMMAND ----------

def PlotGraphWithLabels(graph,vertexLabel,edgeLabel):
  plt.figure(1,figsize=(8,8)) 
  G = nx.DiGraph()
  for x in graph.edges.collect():
    src = x["src"]
    dst = x["dst"]
    weight = x[edgeLabel]
    #print(src,dst,weight)
    G.add_edge(src,dst,weight=weight)
  pos = nx.circular_layout(G)
  #nx.draw(G,with_labels = True, arrows = True, arrowsize=20,edge_color = 'b', arrowstyle='fancy' )
  node_labels = nx.get_node_attributes(G,vertexLabel)
  print("node labels",node_labels)  
  nx.draw_networkx_labels(G, pos)
  
  edge_labels = nx.get_edge_attributes(G,edgeLabel)
  #nx.draw_networkx_edge_labels(G, pos, labels = edge_labels)
  nx.draw_networkx_edges(G, pos, edge_color='green',     arrowstyle='-|>')
  #####
  plt.figure(2,figsize=(8,8)) 
  pos = nx.circular_layout(G)
  #nx.draw(G,with_labels = True, arrows = True, arrowsize=20,edge_color = 'b', arrowstyle='fancy' )
  node_labels = nx.get_node_attributes(G,vertexLabel)
  print("node labels",node_labels)  
  #nx.draw_networkx_labels(G, pos)
  
  edge_labels = nx.get_edge_attributes(G,edgeLabel)
  nx.draw_networkx_edge_labels(G, pos, edge_labels  = edge_labels)
  nx.draw_networkx_edges(G, pos, edge_color='green',     arrowstyle='-|>')
  plt.show()
PlotGraphWithLabels(pizzaGraph,"Name","weight")

# COMMAND ----------

# MAGIC %md
# MAGIC We have some self-loops, where there is a probability of staying in the same state

# COMMAND ----------

display(pizzaGraph.edges)


# COMMAND ----------

display(pizzaGraph.edges.filter("src == dst"))

# COMMAND ----------

# MAGIC %md
# MAGIC In a Markov chain, the sum of the outgoing probabilites must add up to 1

# COMMAND ----------

display(pizzaGraph.edges.groupBy("src").sum("weight"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Converting Graph to state transition matrix

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
# Note the use of a alphabetDesc stringOrderType to ensure both columns have the same mappings
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index",stringOrderType="alphabetDesc").fit(pizzaGraph.edges) for column in list(set(pizzaGraph.edges.columns)-set(['weight'])) ]
pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(pizzaGraph.edges).transform(pizzaGraph.edges)
display(df_r)


# COMMAND ----------

display(df_r.select("src","src_index").distinct().orderBy("src_index"))

# COMMAND ----------

display(df_r.select("dst","dst_index").distinct().orderBy("dst_index"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating the Transition Matrix
# MAGIC Create a matrix initialized to zero, then set the weights for places where transitions have a probability

# COMMAND ----------

import numpy as np
vertexcount = pizzaGraph.vertices.count()
M = np.zeros((vertexcount,vertexcount))
for row in df_r.collect():
  src = int(row["src_index"])
  dst = int(row["dst_index"])
  weight = row["weight"]
  M[src][dst] = weight

# COMMAND ----------

# MAGIC %md 
# MAGIC M is the transition matrix. Given a transition matrix, we can use Markov methods and raise M to a power n, where nis the units of time that have elapsed.
# MAGIC 
# MAGIC Initial state transition probabilites

# COMMAND ----------

M

# COMMAND ----------

# MAGIC %md
# MAGIC What are the chances that a person in line will continue to wait in line after a unit of time has elapsed?

# COMMAND ----------

n = 2
M ** n

# COMMAND ----------

# MAGIC %md
# MAGIC The probability of someone waiting after one unit of time has elapsed (at T=2) goes from 50% to 25%.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Message passing in the graph
# MAGIC Sum up the outgoing weights for each vertex, it should be 1

# COMMAND ----------

from pyspark.sql.functions import avg as sqlavg
from pyspark.sql.functions import sum as sqlsum
from graphframes.lib import AggregateMessages as AM
from graphframes.examples import Graphs


msgToSrc = AM.edge["weight"]
msgToDst = AM.dst["id"]
agg = pizzaGraph.aggregateMessages(
  sqlsum(AM.msg).alias("output"),sendToSrc = msgToSrc)
display(agg)


# COMMAND ----------


