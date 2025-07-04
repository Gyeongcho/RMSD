#Determination of Interatomic Distances in a Triatomic Molecule(I3) via Debye Scattering Curve Fitting

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

# q values and experimental ΔS(q) data
q = np.array([0.1 * i for i in range(1, 81)])
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

# Atomic form factor for Iodine
# Source: https://smilgies.github.io/dms79/x-rays/f0_CromerMann.txt
a = np.array([20.1472, 18.9949, 7.5138, 2.2735])
b = np.array([4.3470, 0.3814, 27.7660, 67.8776])
c = 4.0712

# Compute atomic form factor f(q)
def form_factor(q):
    s = q / (4 * np.pi)
    return np.sum(a[:, None] * np.exp(-b[:, None] * s**2), axis=0) + c

# Debye equation for a triatomic molecule
def S_three_atom(q, r12, r23, r13):
    fq = form_factor(q)
    fq2 = fq ** 2
    return 3 * fq2 + 2 * fq2 * (np.sin(q * r12) / (q * r12) + np.sin(q * r13) / (q * r13) + np.sin(q * r23) / (q * r23))

# Theoretical ΔS(q) model from initial (a) and final (b) structures
def debye_model(q, r12a, r23a, r13a, r12b, r23b, r13b, scale):
    Sa = S_three_atom(q, r12a, r23a, r13a)
    Sb = S_three_atom(q, r12b, r23b, r13b)
    return scale * (Sb - Sa)

# Chi-squared function with geometric penalty
def chi2(r12a, r23a, r13a, r12b, r23b, r13b, scale):
    penalty = 0
    # Geometric constraint: triangle inequality
    if r13a > r12a + r23a:
        penalty += 1e6 * (r13a - r12a - r23a) ** 2
    if r13b > r12b + r23b:
        penalty += 1e6 * (r13b - r12b - r23b) ** 2

    model = debye_model(q, r12a, r23a, r13a, r12b, r23b, r13b, scale)
    return np.sum((deltaS - model) ** 2) + penalty

# Perform fitting using iminuit
m = Minuit(chi2, r12a=2.5, r23a=2.5, r13a=5.0, r12b=2.5, r23b=2.5, r13b=5.0, scale=0.01)
m.limits = [(0.1, 10.0)] * 6 + [(0.0001, 10)]
m.migrad()

# Extract fitted parameters
deltaS_fit = debye_model(q, *m.values)

r12a_fit = m.values['r12a']
r23a_fit = m.values['r23a']
r13a_fit = m.values['r13a']
r12b_fit = m.values['r12b']
r23b_fit = m.values['r23b']
r13b_fit = m.values['r13b']
scale_fit = m.values['scale']

deltaS_fit = debye_model(q, r12a_fit, r23a_fit, r13a_fit, r12b_fit, r23b_fit, r13b_fit, scale_fit)
final_chi2 = chi2(r12a_fit, r23a_fit, r13a_fit, r12b_fit, r23b_fit, r13b_fit, scale_fit)

# Plotting the results
plt.figure(figsize=(9, 6))
plt.plot(q, deltaS, 'o', label='Measured ΔS(q)', markersize=4)
plt.plot(q, deltaS_fit, '-', label=(
    f'Fit:\n'
    f'r12a={r12a_fit:.2f}, r23a={r23a_fit:.2f}, r13a={r13a_fit:.2f}\n'
    f'r12b={r12b_fit:.2f}, r23b={r23b_fit:.2f}, r13b={r13b_fit:.2f}\n'
    f'scale={scale_fit:.2f}'
))
plt.xlabel("q (Å⁻¹)", fontsize=12)
plt.ylabel("ΔS(q)", fontsize=12)
plt.title("Debye Model Fit to ΔS(q)", fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Final chi-squared value output
print(f"Final chi2 value: {final_chi2:.4f}")