lines = open('/usr2/home/zdou/textsum/download/wmt_zhen/tag.wmt-train.acl.len.zh', 'r').readlines()
out = open('/usr2/home/zdou/textsum/download/wmt_zhen/score.tag.wmt-train.acl.len.zh', 'w')
for l in lines:
  out.write('1.0\n')
