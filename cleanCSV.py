import pandas
import sys

df = pandas.DataFrame(pandas.read_csv(sys.argv[1]))
df = df.dropna()
df.to_csv("clean-" + sys.argv[1])