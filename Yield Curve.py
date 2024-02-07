from tabulate import tabulate
import datetime
import math
# import pprint as pprint
import numpy as np
# from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt


class Bond:
    def __init__(self, ISIN, ten_day_price, maturity_date,
                 coupon_rate, periods):
        self.ISIN = ISIN
        self.ten_day_price = ten_day_price
        self.maturity_date = maturity_date
        self.coupon_rate = coupon_rate
        self.periods = periods


ten_day_2024_3_1 = [99.6300, 99.6400, 99.6500, 99.6610, 99.6700, 99.6870,
                    99.6800, 99.6830, 99.7080, 99.7200]
date = datetime.date(2024, 3, 1)
bond_2024_3_1 = Bond("CA135087J546", ten_day_2024_3_1, date, 2.25, 0)
ten_day_price_dates = [8, 9, 10, 11, 12, 15, 16, 17, 18, 19]

# print(bond_2024_3_1.__dict__)
# print(bond_2024_3_1.ten_day_price)

ten_day_2024_9_1 = [97.9600, 97.9800, 97.9850, 97.9820, 98.0210, 98.0540,
                    97.9740, 97.9750, 97.9990, 98.0070]
date = datetime.date(2024, 9, 1)
bond_2024_9_1 = Bond("CA135087J967", ten_day_2024_9_1, date, 1.50, 1)

ten_day_2025_3_1 = [96.4600, 96.4820, 96.5520, 96.5760, 96.6610, 96.7150,
                    96.5400, 96.4480, 96.4950, 96.4600]
date = datetime.date(2025, 3, 1)
bond_2025_3_1 = Bond("CA135087K528", ten_day_2025_3_1, date, 1.25, 2)

ten_day_2025_9_1 = [94.3400, 94.3700, 94.3800, 94.4300, 94.4900, 94.4900,
                    94.4200, 94.2500, 94.2400, 94.2200]
date = datetime.date(2025, 9, 1)
bond_2025_9_1 = Bond("CA135087K940", ten_day_2025_9_1, date, 0.5, 3)

ten_day_2026_3_1 = [92.8620, 92.8600, 92.8440, 92.8560, 93.0230, 93.0080,
                    92.7960, 92.5700, 92.5460, 92.5450]
date = datetime.date(2026, 3, 1)
bond_2026_3_1 = Bond("CA135087L518", ten_day_2026_3_1, date, 0.25, 4)

ten_day_2026_9_1 = [93.4600, 93.4400, 93.5500, 93.5300, 93.6000, 93.5700,
                    93.4000, 93.1200, 93.0800, 93.0700]
date = datetime.date(2026, 9, 1)
bond_2026_9_1 = Bond("CA135087L930", ten_day_2026_9_1, date, 1.00, 5)

ten_day_2027_3_1 = [93.2820, 93.3010, 93.2450, 93.1860, 93.4670, 93.4930,
                    93.1410, 92.8560, 92.7530, 92.7640]
date = datetime.date(2027, 3, 1)
bond_2027_3_1 = Bond("CA135087M847", ten_day_2027_3_1, date, 1.25, 6)

ten_day_2027_9_1 = [97.5800, 97.5910, 97.6030, 97.5200, 97.7400, 97.7720,
                    97.4410, 97.0920, 96.9560, 96.9470]
date = datetime.date(2027, 9, 1)
bond_2027_9_1 = Bond("CA135087N837", ten_day_2027_9_1, date, 2.75, 7)

ten_day_2028_3_1 = [100.5000, 100.4800, 100.4390, 100.3280, 100.6370, 100.6730,
                    100.2200, 99.8340, 99.66200, 99.6230]
date = datetime.date(2028, 3, 1)
bond_2028_3_1 = Bond("CA135087P576", ten_day_2028_3_1, date, 3.5, 8)

ten_day_2028_9_1 = [99.7400, 99.7200, 99.7200, 99.5500, 99.8800, 99.9100,
                    99.4400, 98.990, 98.8100, 98.7700]
date = datetime.date(2028, 9, 1)
bond_2028_9_1 = Bond("CA135087Q491", ten_day_2028_9_1, date, 3.25, 9)

bonds_list = [bond_2024_3_1, bond_2024_9_1, bond_2025_3_1, bond_2025_9_1,
              bond_2026_3_1, bond_2026_9_1, bond_2027_3_1, bond_2027_9_1,
              bond_2028_3_1, bond_2028_9_1]


# 1.1
def present_value(clean_price, coupon_rate, day):
    return round(clean_price + coupon_rate * (180 - (30 - day) - 30) / 360, 6)


list_semiunit = []
for day in ten_day_price_dates:
    lst = []
    lst.append((180 - (30 - day) - 30) / 180)
    list_semiunit.append(lst)
# print(list_semiunit)


def PV_ytm(coupon, y, dateindex, bondindex):
    c = coupon / 2
    t = list_semiunit[dateindex][0]
    sum = 0
    if bondindex == 0:
        sum += (100 + c) / ((1 + (y / 2)) ** t)
    else:
        sum += c / ((1 + (y / 2)) ** t)
        for i in range(1, bondindex):
            t += 1
            sum += c / ((1 + (y / 2)) ** t)
        sum += (100 + c) / ((1 + (y / 2)) ** (t + 1))
    return round(sum, 9)


# presnet value list
for bond in bonds_list:
    bond.present_value = []
    for idx, clean_price in enumerate(bond.ten_day_price):
        bond.present_value.append(present_value(clean_price, bond.coupon_rate,
                                                ten_day_price_dates[idx]))
    # print(bond.__dict__)

# ================================= 1.1 ======================================

def calculate_ytm(c, pv, dateindex, bondindex):
    ytm = c / 100
    condition = True
    while condition:
        # pv < (100 + (c / 2))
        if pv < (100 + (c / 2)):
            ytm += 0.000001
        else:
            ytm -= 0.000001
        total_pv = PV_ytm(c, ytm, dateindex, bondindex)

        if pv < (100 + (c / 2)):
            condition = total_pv > pv
        else:
            condition = total_pv < pv

    ytm2 = c / 100
    condition = True
    while condition:
        if pv < (100 + (c / 2)):
            ytm2 -= 0.000001
        else:
            ytm2 += 0.000001
        total_pv = PV_ytm(c, ytm2, dateindex, bondindex)

        if pv < (100 + (c / 2)):
            condition = total_pv < pv
        else:
            condition = total_pv > pv
    if pv > 100:
        return ytm2 * 100
    else:
        return ytm * 100
    # return min(ytm*100, ytm2*100)

# a0 = calculate_ytm(2.25, 100.429875, 0, 0)
# a1 = calculate_ytm(2.25, 100.446249, 1, 0)
# a2 = calculate_ytm(2.25, 100.4625, 2, 0)
# b0 = calculate_ytm(1.25, 96.993325, 0, 2)
# b1 = calculate_ytm(1.25, 96.9299162, 1, 2)
# b2 = calculate_ytm(1.25, 97.0033888, 2, 2)
# c = calculate_ytm(1, 5, 8)
# print(a0)
# print(a1)
# print(a2)
# print(b0)
# print(b1)
# print(b2)

# c = coupon rate * fv , fv = 100, pv, n = idx
super_ytm = []
for bond_idx, bond in enumerate(bonds_list):
    # print("Bond index: "+ str(bond_idx))
    ytm1 = []
    for idx, pv in enumerate(bond.present_value):
        # print("Date index: "+ str(idx))
        # ytm1= []
        # ytm_semi = ytm(bond.coupon_rate, 100, pv, idx + 1)
        ytm = calculate_ytm(bond.coupon_rate, pv, idx, bond_idx)
        ytm1.append(ytm)
    super_ytm.append(ytm1)
# print(super_ytm)
tpytm = np.transpose(super_ytm)
# print(tpytm)


super_present_value = []
for idx, bond in enumerate(bonds_list):
    super_present_value.append(bond.present_value)
tppv = np.transpose(super_present_value)
# print(tppv)


xpoints = []
for bond in bonds_list:
    xpoints.append(bond.maturity_date)
ypoints = tpytm
for i in range(len(ypoints)):
    ypoints[i] = super_ytm[i]

dates = ['Jan 8', 'Jan 9', 'Jan 10', 'Jan 11', 'Jan 12', 'Jan 15', 'Jan 16', 'Jan 17', 'Jan 18', 'Jan 19']
plt.title('five year yield curve')
plt.xlabel('time to maturity')
plt.ylabel('yield to maturity')
plt.xticks(xpoints, labels=['24/3', '24/9', '25/3', '25/9', '26/3', '26/9', '27/3', '27/9', '28/3', '28/9'])

for i in range(10):
    plt.plot(xpoints, ypoints[i], marker = 'o', ms = 3, label = dates[i])
plt.legend(loc=1, prop={'size': 6})
plt.grid()
plt.show()




# ================================== 1.2 ======================================
coupon_list = []
for i in range(10):
    coupon_list.append(bonds_list[i].coupon_rate)

# list based on certain date
def bootstraping_spot(dateindex):
    spot_rates = np.zeros(len(bonds_list))
    # c = coupon / 2
    # pv = tppv[dateindex][bondindex]
    for i in range(10):
        c = coupon_list[i] / 2
        if i == 0:
            spot_rates[0] = super_ytm[0][dateindex] / 100
            # print(tpytm[0][dateindex])
            # pv = tppv[dateindex][0]
            # spot_rate = ((100 + c)/pv)**(1/t) - 1
            # spot_rates[0] = spot_rate*200
            # print("spot rate 0:", spot_rates[0])
        else:
            # for i in range(1, 10):
            discounted_cf = 0
                # here i = bondindex
                # discounted_cf = (coupon_list[0]/2) / ((1+spot_rates[0]/2)**t)
                # if i == 1:
                #     spot_rate = ((100 + c)/(tppv[dateindex][i] - discounted_cf))** (1/(t+1)) - 1
                #     spot_rates[i] = spot_rate * 200
            t = list_semiunit[dateindex][0]
            for j in range(1, i+1):
                # print(t)
                # print("################# Inner Loop, j = ", j," ####################")
                discounted_cf += c / ((1 + spot_rates[j-1] / 2) ** t)
                t += 1
                # print("dis_cf:", discounted_cf)
                # print("coupone payment:", c)
                # print("spot rate ", j-1, ": ", spot_rates[j-1])

            # print("dis CF:", discounted_cf)
            residual = tppv[dateindex][i] - discounted_cf
            # print(residual)
            # print(tppv[dateindex][i])
            spot_rate = (((100 + c) / residual)**(1/t) - 1) * 2
            # print(t)
            # print(spot_rate)
            # print("spot rate ", i, ": ", spot_rate )
            spot_rates[i] = spot_rate
    return spot_rates

super_spot = []
for n in range(10):
    super_spot.append(bootstraping_spot(n))

xpoints = []
for bond in bonds_list:
    xpoints.append(bond.maturity_date)
ypoints = tpytm
for i in range(len(ypoints)):
    ypoints[i] = super_spot[i]

dates = ['Jan 8', 'Jan 9', 'Jan 10', 'Jan 11', 'Jan 12', 'Jan 15', 'Jan 16', 'Jan 17', 'Jan 18', 'Jan 19']
plt.title('five year spot curve')
plt.xlabel('time to maturity')
plt.ylabel('spot rate')
plt.xticks(xpoints, labels=['24/3', '24/9', '25/3', '25/9', '26/3', '26/9', '27/3', '27/9', '28/3', '28/9'])

for i in range(10):
    plt.plot(xpoints, ypoints[i], marker = 'o', ms = 3, label = dates[i])
plt.legend(loc=1, prop={'size': 6})
plt.grid()
plt.show()


#===================================1.3==================================
def forward_rate(dateindex):
    forward_lst = []
    s = super_spot[dateindex][1]
    t = 1
    for i in range(1, 5):
        t = t+2
        sp = super_spot[dateindex][t]
        numerator = (1+(sp/2))**(2*(1+i))
        denominator = (1+(s/2))**2
        forward = (numerator/denominator)**(1/(2*i)) -1
        forward_lst.append(forward*100)
    return forward_lst

super_forward = []
for n in range(10):
    super_forward.append(forward_rate(n))
# print(super_forward)

xpoints = [0, 1, 2, 3]
# for bond in bonds_list:
#     xpoints.append(bond.maturity_date)
tpfr = np.transpose(super_forward)

ypoints = super_forward
# for i in range(len(super_forward)):
#     ypoints[i] = super_forward[i]

dates = ['Jan 8', 'Jan 9', 'Jan 10', 'Jan 11', 'Jan 12', 'Jan 15', 'Jan 16', 'Jan 17', 'Jan 18', 'Jan 19']
plt.title('five year forward rate curve')
plt.xlabel('year')
plt.ylabel('forward rate')
plt.xticks(xpoints, labels=['1y1y', '1y2y', '1y3y', '1y4y'])
# plt.xticks(ticks = [0, 1, 2, 3], labels = ['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'])
for i in range(10):
    plt.plot(xpoints, ypoints[i], marker = 'o', ms = 3, label = dates[i])
plt.legend(loc=1, prop={'size': 6})
plt.grid()
plt.show()

# =================== Q5 - YTM coviraince matrix ====================
ytm_time_series = []
for i in range(1, 6):
    sub_lst = []
    for j in range(9):
        X = math.log(super_ytm[2*i - 1][j+1] / super_ytm[2*i - 1][j])
        sub_lst.append(X)
    ytm_time_series.append(sub_lst)

ytm_cormatrix = np.cov(ytm_time_series)
print(ytm_cormatrix)

# =================== Q5 - forward coviraince matrix ====================
forward_time_series = [] #4 x 9
for i in range(4):
    sub_lst = []
    for j in range(9):
        X =  math.log(super_forward[j+1][i] / super_forward[j][i])
        sub_lst.append(X)
    forward_time_series.append(sub_lst)

forward_cormatrix = np.cov(forward_time_series)
print(forward_cormatrix)

# =================== Q6 - eigenvalue/eigenvector ytm  =================
ytm_eigenvalues, ytm_eigenvectors = np.linalg.eig(ytm_cormatrix)
print("ytm eigenvalues: ", ytm_eigenvalues)
print("ytm eigenvectors: ", ytm_eigenvectors)

# =================== Q6 - eigenvalue/eigenvector forward  =================
for_eigenvalues, for_eigenvectors = np.linalg.eig(forward_cormatrix)
print("forward eigenvalues: ", for_eigenvalues)
print("forward eigenvectors: ", for_eigenvectors)
