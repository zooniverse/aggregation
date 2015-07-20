__author__ = 'greg'
import requests
import pymongo
import json
import csv

token_mapping = {}

with open("/home/greg/Dropbox/cloudsight.csv","rb") as f:
    c = csv.reader(f)
    for url,token in c:
        token_mapping[url] = token

# connect to the mongo server
client = pymongo.MongoClient()
db = client['serengeti_2015-02-22']
classification_collection = db["serengeti_classifications"]
subject_collection = db["serengeti_subjects"]
user_collection = db["serengeti_users"]

non_blanks = []

# for subject in subject_collection.find({"tutorial":{"$ne":True},"metadata.retire_reason":{"$ne":"blank"}}).limit(20000):
#     non_blanks.append(subject["zooniverse_id"])
#
# print random.sample(non_blanks,300)

subjects = [u'ASG0001wqt', u'ASG0001enl', u'ASG0000ssy', u'ASG0001fhf', u'ASG00008y2', u'ASG0001ofr', u'ASG0000098', u'ASG00005ln', u'ASG0001374', u'ASG00008e1', u'ASG0000fg0', u'ASG00017gx', u'ASG00002gp', u'ASG0000f06', u'ASG0001hlb', u'ASG0001aeh', u'ASG0000f28', u'ASG0000s2y', u'ASG0000std', u'ASG00019a5', u'ASG0001pq0', u'ASG0001cj3', u'ASG0001dyc', u'ASG0000s23', u'ASG0001c3a', u'ASG00006dk', u'ASG0001dyr', u'ASG0000vlz', u'ASG00007kw', u'ASG0001i0q', u'ASG00012b3', u'ASG0001acp', u'ASG0001fj5', u'ASG0001aqt', u'ASG00017pp', u'ASG00007v8', u'ASG0000fia', u'ASG0001cph', u'ASG0000swp', u'ASG0001g9s', u'ASG0001lk6', u'ASG0001cws', u'ASG0001wbb', u'ASG0000rou', u'ASG0000h7a', u'ASG00017gn', u'ASG00002ht', u'ASG0001aok', u'ASG0001dp9', u'ASG0000psn', u'ASG0001uv1', u'ASG00019o5', u'ASG0000s66', u'ASG0000qrj', u'ASG0001oh2', u'ASG0000ztz', u'ASG00007kv', u'ASG0000vvx', u'ASG0000wja', u'ASG00008fv', u'ASG0000gm7', u'ASG0001wuy', u'ASG0000spz', u'ASG0001asj', u'ASG000127w', u'ASG00000ul', u'ASG0000pua', u'ASG0001b9x', u'ASG00005yw', u'ASG00004wh', u'ASG0000gdx', u'ASG00001c8', u'ASG0000vl5', u'ASG000133p', u'ASG0001eot', u'ASG0000615', u'ASG0001hco', u'ASG0000dfk', u'ASG0000gv2', u'ASG0001qf4', u'ASG0001ur7', u'ASG0001hrk', u'ASG0001ew5', u'ASG000061i', u'ASG0000aez', u'ASG0000py2', u'ASG000013t', u'ASG00001l9', u'ASG0000rrh', u'ASG00012z1', u'ASG0001hho', u'ASG00005x2', u'ASG00007ok', u'ASG0001ll3', u'ASG0001ps8', u'ASG0000up6', u'ASG0001fpr', u'ASG0000g14', u'ASG0001obo', u'ASG0000exz', u'ASG0000qfh', u'ASG0001ab0', u'ASG0001f0e', u'ASG00012ws', u'ASG0001cwh', u'ASG0000rwe', u'ASG0000ak9', u'ASG0001u6h', u'ASG000093k', u'ASG0001hhr', u'ASG0001e2y', u'ASG0001q0g', u'ASG0000ghb', u'ASG0000kri', u'ASG0000hi4', u'ASG0001hw1', u'ASG00008xe', u'ASG0000rgz', u'ASG0000krt', u'ASG0001uvz', u'ASG0000fqd', u'ASG00012cz', u'ASG00012u4', u'ASG00007od', u'ASG0000u8v', u'ASG0001ofd', u'ASG00017i1', u'ASG00019om', u'ASG0001psy', u'ASG0000fhg', u'ASG000086q', u'ASG0000vrb', u'ASG0001f4m', u'ASG0001hsk', u'ASG0001bgq', u'ASG000093q', u'ASG0000vr0', u'ASG0001pnw', u'ASG0000eog', u'ASG0001eyz', u'ASG00007nb', u'ASG0001lj9', u'ASG000028v', u'ASG0001q0u', u'ASG0000046', u'ASG0000ktt', u'ASG00008hn', u'ASG00019go', u'ASG0001huy', u'ASG00017ub', u'ASG0000qfr', u'ASG000042t', u'ASG0001lsk', u'ASG0001aqp', u'ASG0000891', u'ASG0000pxd', u'ASG00013hx', u'ASG0001enc', u'ASG0000rk3', u'ASG0000feb', u'ASG0001hpy', u'ASG0000u5r', u'ASG0000ad4', u'ASG00012xt', u'ASG0001g5c', u'ASG0001ok0', u'ASG0000ad9', u'ASG00011y4', u'ASG0000fmk', u'ASG000138l', u'ASG0000gn2', u'ASG0001am6', u'ASG0000f9r', u'ASG0001agt', u'ASG000050z', u'ASG00013b7', u'ASG00005pc', u'ASG0001i1t', u'ASG00007hl', u'ASG0000s44', u'ASG00005co', u'ASG000065i', u'ASG0000pfj', u'ASG0001r72', u'ASG00019g8', u'ASG00012bi', u'ASG0000dan', u'ASG0001px8', u'ASG0000ffj', u'ASG000037j', u'ASG0001ezj', u'ASG0000325', u'ASG0000rv6', u'ASG0001c7e', u'ASG0000crh', u'ASG0001hta', u'ASG0001h9a', u'ASG00012xi', u'ASG0001dxh', u'ASG0000fpz', u'ASG0000kph', u'ASG0001ael', u'ASG0001exh', u'ASG0001cuk', u'ASG00001j1', u'ASG0000g8k', u'ASG0001ik6', u'ASG0000wgt', u'ASG0001g5t', u'ASG0001fil', u'ASG000015q', u'ASG0001bl5', u'ASG00012hv', u'ASG0001aod', u'ASG0001vfi', u'ASG00008e2', u'ASG0001unf', u'ASG00018ku', u'ASG00012k1', u'ASG0001cgg', u'ASG0001g2a', u'ASG00007gv', u'ASG0001e5s', u'ASG0000886', u'ASG0001wyp', u'ASG00006j6', u'ASG0000zjy', u'ASG0001bvq', u'ASG00008g0', u'ASG0001uss', u'ASG0001dpi', u'ASG00012hw', u'ASG0000gb5', u'ASG00002ir', u'ASG0000cp9', u'ASG000095b', u'ASG0000gde', u'ASG00007fc', u'ASG0001ad0', u'ASG0001emr', u'ASG0001c77', u'ASG0001ahk', u'ASG0000ct3', u'ASG0000wq0', u'ASG0001cxd', u'ASG0001ual', u'ASG000089k', u'ASG00012cs', u'ASG0001h9d', u'ASG0000gft', u'ASG0000s19', u'ASG0000rvd', u'ASG0001337', u'ASG0001hud', u'ASG0000u5z', u'ASG00005dj', u'ASG0001302', u'ASG0001qbl', u'ASG00019ea', u'ASG00012ni', u'ASG00003yf', u'ASG0000899', u'ASG00000wx', u'ASG0001fjb', u'ASG0000fez', u'ASG000096p', u'ASG00019d7', u'ASG00008ex', u'ASG0000pgd', u'ASG000091n', u'ASG000020a', u'ASG0001774', u'ASG00002db', u'ASG0001fjz', u'ASG0001aen', u'ASG000025c', u'ASG0000mll', u'ASG0000st5', u'ASG0001agf', u'ASG000152t', u'ASG0000etx', u'ASG0000q8j', u'ASG0001cj4', u'ASG0001hn1', u'ASG0000vrs', u'ASG0001h44', u'ASG0001ejk', u'ASG0000fbi', u'ASG00012vb', u'ASG0001fk1', u'ASG0000s5r', u'ASG0001h9v', u'ASG0001lpk', u'ASG00019ly', u'ASG0001dnv', u'ASG0001b4y', u'ASG0000byc', u'ASG00000n8', u'ASG00000i9', u'ASG00003i3']

for subject_id in subjects:
    images = subject_collection.find_one({"zooniverse_id":subject_id})["location"]["standard"]
    for i in images:
        assert isinstance(i,unicode)
        slash_index = i.rfind("/")
        url = "https://static.zooniverse.org/www.snapshotserengeti.org/subjects/standard/" + i[slash_index+1:]

        if url in token_mapping:
            print "already done"
            continue

        header = {"Authorization":"CloudSight FH4Bnx5ahv3_r3V9Ja8bcg"}
        footer = {"image_request[remote_image_url]":url,"image_request[locale]":"en-US"}

        r = requests.post("https://api.cloudsightapi.com/image_requests",headers=header,data=footer)
        if r.status_code != 200:
            print r.text
            print url

        t = json.loads(r.text)
        print url + "," +str(t["token"])
