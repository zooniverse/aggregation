__author__ = 'greg'
#don't mention subject_id because that MUST always be first
global_required_param = {"user_name":str,"time_stamp":str,}
required_param = dict()
required_param["animalMarking"] = dict(global_required_param.items() + {"animal":str}.items())

possible_param = dict()
possible_param["animalMarking"] = {"x": float, "y": float, "label": str}
