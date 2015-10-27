#!/usr/bin/env python
__author__ = 'ggdhines'
from transcription import Tate
import json

project_id = 245
environment = "development"

first = True

with open("/tmp/transcription.json","wb") as f:
    with Tate(project_id,environment) as project:
        f.write("[")
        for subject_id,aggregations in project.__yield_aggregations__(121):
            empty = True

            if "T2" not in aggregations:
                continue

            for cluster_index,cluster in aggregations["T2"]["text clusters"].items():
                if cluster_index == "all_users":
                    continue

                if cluster["num users"] < 5:
                    continue

                if empty and not first:
                    f.write(",")

                if first:
                    first = False



                if empty:
                    m = project.__get_subject_metadata__(subject_id)
                    metadata = m["subjects"][0]["metadata"]
                    f.write("{\"subject_id\": " + str(subject_id) + ", \"metadata\": " + json.dumps(metadata) + ", \"aggregated_text\":[")
                    f.write("{\"coordinates\":" + str(cluster["center"][:-1])+", \"text\":\"")
                else:
                    f.write(",{\"coordinates\":" + str(cluster["center"][:-1])+", \"text\":\"")
                empty = False

                line = ""




                for c_i,c in enumerate(cluster["center"][-1]):
                    if ord(c) in [24,27]:
                        options = set([individual_text[c_i] for coord,individual_text in cluster["cluster members"]])
                        line += "<disagreement>"
                        for c in options:
                            line += "<option>"
                            if ord(c) == 24:
                                pass
                            else:
                                if c == "\"":
                                    line += "\\\""
                                elif c == "\\":
                                    line += "\\\\"
                                else:
                                    line += c
                        line += "</disagreement>"
                    else:
                        if c == "\"":
                            line += "\\\""
                        elif c == "\\":
                            line += "\\\\"
                        else:
                            line += c



                f.write(line + "\"}")

            if not empty:
                f.write("]}")
        f.write("]")

with open("/tmp/transcription.json","r") as f:
    transcriptions = f.read()
    s = json.loads(transcriptions)