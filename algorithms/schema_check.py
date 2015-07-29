__author__ = 'greg'
import json
from jsonschema import validate

s = ""
with open("/home/greg/test.json","rb") as f:
    for l in f.readlines():
        s += l.strip()

t = json.loads(s)

schema_str = ""
with open("aggregation_schema.json","rb") as f:
    for l in f.readlines():
        schema_str += l.strip()

schema = json.loads(schema_str)

print json.dumps(t, sort_keys=True,indent=4, separators=(',', ': '))

validate(t["568731"],schema)