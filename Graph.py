# Databricks notebook source
# MAGIC %md 
# MAGIC # What is a Graph?
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
# MAGIC For a detailed discussion of graphs consider 
# MAGIC 
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

vertices = spark.sql("select * from vertices")
display(vertices)

# COMMAND ----------

# MAGIC %md
# MAGIC GraphFrames require an id column to uniquely identify a given vertex.

# COMMAND ----------

vertices = vertices.withColumnRenamed("Node","id")
display(vertices)

# COMMAND ----------

edges = spark.sql("select * from edges")
display(edges)

# COMMAND ----------

# MAGIC %md
# MAGIC GraphFrame expects two columns in the dataset that contains the edges. They need to be named src (for the source of the edge) and dest (for the sink of the edge).

# COMMAND ----------

edges = edges.withColumnRenamed("From","src").withColumnRenamed("To","dst")
display(edges)

# COMMAND ----------

from graphframes import *
g = GraphFrame(vertices, edges)
print(g)

# COMMAND ----------

# MAGIC %md
# MAGIC # Example - Prerequisites in undergraduate programs
# MAGIC Typically, classes in undergraduate are gated by prerequisites.
# MAGIC - Example: OO Programming -> Video Game Programming
# MAGIC - One class often unlocks multiple higher numbered classes
# MAGIC - Important for students to take the right classes (and gain the right knowledge) at the right time

# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt

def PlotGraph(graph):
  G = nx.from_pandas_edgelist(g.edges.toPandas(),'src','dst')
  plt.figure(1,figsize=(8,8)) 
  nx.draw(G,with_labels = True, arrows = True, arrowsize=20,edge_color = 'b', arrowstyle='fancy' )
  plt.show()
PlotGraph(g)

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
display(g.inDegrees.join(vertices,"id").orderBy(desc("inDegree")))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### outDegree - sum of the edges starting at each vertex

# COMMAND ----------

display(g.outDegrees.join(vertices,"id").orderBy(desc("outDegree")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### degree - Total edges starting or ending at a vertex

# COMMAND ----------

display(g.degrees.join(vertices,"id").orderBy(desc("degree")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Strongly connected component
# MAGIC From Wikipedia - In the mathematical theory of directed graphs, a graph is said to be strongly connected if every vertex is reachable from every other vertex. 

# COMMAND ----------

# Compute the strongly connected component (SCC) of each vertex and return a graph with each vertex assigned to the SCC containing that vertex.
result = g.stronglyConnectedComponents(maxIter=5)
#result.select("id", "Name").orderBy("component").show()
display(result.groupBy(col("component")).count().orderBy(desc("count")))
#Note there are no vertices that are part of a subgraph that are stronlgy connected

# COMMAND ----------

# MAGIC %md 
# MAGIC ## PageRank
# MAGIC Run the Google PageRank algorithm on the graph to determine most important vertices

# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices.select("id","pagerank").join(vertices,"id").orderBy(desc("pagerank")))


# COMMAND ----------

display(results.edges.select("src", "dst", "weight").orderBy(desc("weight")))

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

chain5Filter = g.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d); (d)-[de]->(e)").filter("e.id = 12")
display(chain5Filter)

# COMMAND ----------

display(chain5Filter.select("a.Label", "b.Label", "c.Label","d.Label", "e.Label"))

# COMMAND ----------


