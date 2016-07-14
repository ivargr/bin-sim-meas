
def read_data(f,):
    text = f.read()
    parts = text.split(":")
    names = []
    effect_sizes = []
    size_factors = []
    name = ""
    effect=""
    ratio =""
    for i,part in enumerate(parts):
        if (i==0):
            name = part.replace("Effect","")
            continue

        parts2 = part.split("]")
        if (i%2==1):
            effect = [float(x) for x in  parts2[0].replace("[","").strip().split()]
            name = parts2[1].strip().replace("SizeRatio","")
        else:
            ratio = [float(x.strip()) for x in  parts2[0].replace("[","").split(",")]
        if (i%2==0):
            names.append(name)
            effect_sizes.append(effect)
            size_factors.append(ratio)
   
    return names, effect_sizes,size_factors

if __name__ =="__main__":
    f = open('sizeandeffect.txt', 'r')
    names, effects,ratios = read_data(f)
    f.close()
    for n,e,r in zip(names,effects,ratios):
        print "-----------------------"
        print n
        print e
        print r
