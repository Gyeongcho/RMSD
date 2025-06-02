#3원자 구조피팅-Monte Carlo 최적화

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from random import uniform

# q 및 deltaS 데이터
q = np.linspace(0.1, 8.0, 80)
deltaS = np.array([
    0.748503553926166, 2.85608883230165, 5.94120163657277, 9.46605607958838, 12.8544066281627,
    15.6073260217998, 17.3872039987187, 18.0516615466633, 17.6353161245211, 16.2928725569246,
    14.226975918033, 11.6259312210725, 8.62993508038341, 5.33262622338519, 1.8117725571646,
    -1.82698920928553, -5.41333305935056, -8.69935764214706, -11.3791887166095, -13.138584681506,
    -13.7219640736228, -12.9981588233204, -11.0057633290118, -7.9641554889398, -4.24544340344956,
    -0.313010682325994, 3.35897023647031, 6.36606887563774, 8.43576214351944, 9.45823181245474,
    9.48134462074383, 8.67459689885024, 7.27418077825593, 5.52495167869149, 3.63418413456632,
    1.74712493118119, -0.0528966729153581, -1.72446622294297, -3.23832170398868, -4.55001266243796,
    -5.58538351614188, -6.24440229601235, -6.42247106300794, -6.04244262093669, -5.08689368532434,
    -3.61980978656757, -1.78976240196008, 0.18803551312018, 2.06733548523524, 3.6156097503246,
    4.65685960938593, 5.1019392007435, 4.95932292113988, 4.32454445047981, 3.35211725817599,
    2.21804040632166, 1.08286300321537, 0.0643293133652242, -0.774723428017778, -1.42221060637053,
    -1.89997531667319, -2.24109456028971, -2.46912932506907, -2.5857881594096, -2.57031331971575,
    -2.38995361324611, -2.01736367876601, -1.44868573771192, -0.715976206738385, 0.110521388889101,
    0.931602932287037, 1.63768453992418, 2.13179857013334, 2.35076148848868, 2.27878881784955,
    1.94997008439271, 1.43911354354808, 0.843617429530557, 0.261332591706866, -0.229765980379518
])

# Atomic form factor
a = np.array([20.1472, 18.9949, 7.5138, 2.2735])
b = np.array([4.3470, 0.3814, 27.7660, 67.8776])
c = 4.0712


def form_factor(q):
    s = q / (4 * np.pi)
    return np.sum(a[:, None] * np.exp(-b[:, None] * s ** 2), axis=0) + c


# 3-atom S(q)
def S_three_atom(q, r12, r23, r13):
    fq = form_factor(q)
    fq2 = fq ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        sin_term = (np.sin(q * r12) / (q * r12) +
                    np.sin(q * r13) / (q * r13) +
                    np.sin(q * r23) / (q * r23))
        sin_term[np.isnan(sin_term)] = 1.0  # for q→0
    return 3 * fq2 + 2 * fq2 * sin_term


def debye_model(q, r12a, r23a, r13a, r12b, r23b, r13b, scale):
    Sa = S_three_atom(q, r12a, r23a, r13a)
    Sb = S_three_atom(q, r12b, r23b, r13b)
    return scale * (Sb - Sa)


def chi2(r12a, r23a, r13a, r12b, r23b, r13b, scale):
    penalty = 0
    if r13a > r12a + r23a:
        penalty += 1e6 * (r13a - r12a - r23a) ** 2
    if r13b > r12b + r23b:
        penalty += 1e6 * (r13b - r12b - r23b) ** 2
    model = debye_model(q, r12a, r23a, r13a, r12b, r23b, r13b, scale)
    return np.sum((deltaS - model) ** 2) + penalty


# 반복 최적화
best_minuit = None
best_chi2 = np.inf

for _ in range(1000):
    init_params = {
        'r12a': uniform(1.0, 5.0),
        'r23a': uniform(1.0, 5.0),
        'r13a': uniform(2.0, 10.0),
        'r12b': uniform(1.0, 5.0),
        'r23b': uniform(1.0, 5.0),
        'r13b': uniform(2.0, 10.0),
        'scale': uniform(0.001, 0.1)
    }
    m = Minuit(chi2, **init_params)
    m.limits = [(0.1, 10.0)] * 6 + [(0.0001, 10.0)]
    m.migrad()

    if m.valid and m.fval < best_chi2:
        best_chi2 = m.fval
        best_minuit = m


params = best_minuit.values
deltaS_fit = debye_model(q, *params)

plt.figure(figsize=(9, 6))
plt.plot(q, deltaS, 'o', label='Measured ΔS(q)', markersize=4)
plt.plot(q, deltaS_fit, '-', label=(
    f'Best Fit:\n'
    f'r12a={params["r12a"]:.2f}, r23a={params["r23a"]:.2f}, r13a={params["r13a"]:.2f}\n'
    f'r12b={params["r12b"]:.2f}, r23b={params["r23b"]:.2f}, r13b={params["r13b"]:.2f}\n'
    f'scale={params["scale"]:.4f}'
))
plt.xlabel("q (Å⁻¹)")
plt.ylabel("ΔS(q)")
plt.title("Debye Model Fit (with Random Restarts)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()