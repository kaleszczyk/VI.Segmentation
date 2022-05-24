import xml.etree.ElementTree as et

tree = et.parse("colours.xml")
p = tree.find("colours")
names = list(tree.iter("name"))
rs = list(tree.iter("R"))
gs = list(tree.iter("G"))
bs = list(tree.iter("B"))

for i in range(0, len(rs)): 
    print(names[i].text)
    R = rs[i].text
    G = gs[i].text
    B = bs[i].text



