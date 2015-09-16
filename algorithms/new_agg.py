__author__ = 'greg'
def long_substr(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and is_substr(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    return substr

def is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True


t = ['ARTS -> painting - sculpting', 'ARTS- painting- sculpture']
t= ['a show at Galarie Jean Castel, Paris in Spring 1947 but had to cancel it because', 'a show at Galarie Jean Castel, Paris in Spring 1947 but had to cancel it because']
t = ['My darling little one We arrived here about 5.30 yesterday', 'My darling little one We arrived here about 5.30 yesterday', 'My darling little one we arrived here about 5.30 yesterday']
# t = ['based on my head (about life-size). A small seated male figure (wood). A', 'based on my head (about life-size). A small seated male figure (wood). A']
t = ['love as a snare which weakened the capacity to work,spftened', 'love as a snare which weakened the capacity to work, spftened', 'love as a snare which weakened the capacity to work sftned']
t = ['me.On the intellectual level, K never wanted to be passionately', 'me. On the intellectual level, K never wanted to be passionately', 'me.On the intellectual level, K never wanted to be passionately']
t = ['ago-back in the 30s - I fused in his emotions as both friend', 'ago -- bacj in the 30s - I fused in his emotions as both friend', 'ago-back in the 30s - I fused in his emotions as both friend']
t = ['After all, he could just walk out now but he caanot renounce', 'After all, he could just walk out now but he caanot renounce', 'After all, he could just walk out now but he cannot renounce']
print long_substr(t)