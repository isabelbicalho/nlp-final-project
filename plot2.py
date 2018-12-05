import matplotlib.pyplot as plt

#plt.plot(x, losses, 'r', x, acc, 'b', lw=2)
#plt.show()
#plt.title('Loss training')
train  = (44404, 22314, 11163, 5263, 5185)[::-1]
btrain = (66718, 21611)[::-1]
test = (4913, 2457, 1281, 618, 546)[::-1]
btest = (7370, 2445)[::-1]

fig, ax = plt.subplots()
index = [0, 1, 2, 3, 4]
bar_width = 0.35
opacity=0.8

rects1 = plt.bar(index, train, bar_width, alpha=opacity, color='b', label="Train samples")
rects2 = plt.bar([i + bar_width for i in index], test, bar_width, alpha=opacity, color='g', label="Test samples")

#plt.grid(True)
plt.legend(loc=0)
plt.ylabel('number of samples')
plt.xlabel('binary ratings')
plt.tight_layout()
plt.xticks([i + bar_width/2 for i in index], ('1', '2', '3', '4', '5'))
plt.show()
