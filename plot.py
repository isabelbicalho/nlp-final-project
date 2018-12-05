import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
losses = [
    1.3116,
    1.2702,
    1.2509,
    1.1977,
    1.1824,
    1.1636,
    1.1370,
    1.1174,
    1.1037,
    1.0932,
    1.0826,
    1.0717,
    1.0644,
    1.0573,
    1.0522,
    1.0475,
    1.0465,
    1.0372,
    1.0360,
    1.0312,
    1.0340,
    1.0250,
    1.0207,
    1.0210,
    1.0153,
    1.0140,
    1.0125,
    1.0089,
    1.0033,
    1.0005,
    0.9985,
    0.9982,
    0.9949,
    0.9916,
    0.9888,
    0.9820,
    0.9839,
    0.9819,
    0.9774,
    0.9747,
    0.9771,
    0.9673,
    0.9657,
    0.9706,
    0.9611,
    0.9569,
    0.9574,
    0.9603,
    0.9581,
    0.9502,
]
acc = [
    0.4969,
    0.5025,
    0.5016,
    0.5065,
    0.5114,
    0.5166,
    0.5252,
    0.5315,
    0.5377,
    0.5414,
    0.5443,
    0.5514,
    0.5528,
    0.5560,
    0.5567,
    0.5609,
    0.5599,
    0.5624,
    0.5644,
    0.5645,
    0.5638,
    0.5680,
    0.5694,
    0.5682,
    0.5713,
    0.5716,
    0.5715,
    0.5754,
    0.5761,
    0.5772,
    0.5789,
    0.5776,
    0.5799,
    0.5817,
    0.5818,
    0.5853,
    0.5851,
    0.5846,
    0.5878,
    0.5890,
    0.5870,
    0.5908,
    0.5920,
    0.5900,
    0.5938,
    0.5966,
    0.5984,
    0.5946,
    0.5976,
    0.5976,
]

binary_loss = [
    0.5698,
    0.5313,
    0.4995,
    0.4844,
    0.4760,
    0.4643,
    0.4496,
    0.4413,
    0.4345,
    0.4345,
    0.4254,
    0.4254,
    0.4150,
    0.4121,
    0.4101,
    0.4058,
    0.4033,
    0.3997,
    0.3985,
    0.3982,
    0.3948,
    0.3900,
    0.3890,
    0.3876,
    0.3818,
    0.3818,
    0.3811,
    0.3771,
    0.3775,
    0.3717,
    0.3723,
    0.3694,
    0.3702,
    0.3651,
    0.3644,
    0.3611,
    0.3664,
    0.3550,
    0.3537,
    0.3544,
    0.3519,
    0.3494,
    0.3476,
    0.3449,
    0.3416,
    0.3446,
    0.3409,
    0.3395,
    0.3363,
    0.3317
]
binary_acc = [
    0.7536,
    0.7560,
    0.7635,
    0.7736,
    0.7794,
    0.7834,
    0.7932,
    0.7986,
    0.8018,
    0.8027,
    0.8084,
    0.8071,
    0.8127,
    0.8145,
    0.8150,
    0.8170,
    0.8179,
    0.8209,
    0.8213,
    0.8228,
    0.8230,
    0.8254,
    0.8272,
    0.8282,
    0.8301,
    0.8287,
    0.8307,
    0.8324,
    0.8334,
    0.8340,
    0.8347,
    0.8369,
    0.8378,
    0.8388,
    0.8400,
    0.8404,
    0.8379,
    0.8433,
    0.8450,
    0.8434,
    0.8458,
    0.8458,
    0.8470,
    0.8482,
    0.8485,
    0.8483,
    0.8499,
    0.8499,
    0.8534,
    0.8550
]

#plt.plot(x, losses, 'r', x, acc, 'b', lw=2)
#plt.show()
#plt.title('Loss training')
plt.plot(x, losses, 'b',      label='Multiclass model', lw=2)
plt.plot(x, binary_loss, 'g', label='Binary model', lw=2)
plt.grid(True)
plt.legend(loc=0)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()