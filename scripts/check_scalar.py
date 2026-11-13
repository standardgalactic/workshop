from fractions import Fraction as Fr

a, b = Fr(1), Fr(4)
u, v = Fr(0), Fr(1)
eta = Fr(1, 5)
xstar = (a * u + b * v) / (a + b)


def d(c, H):
    return 1 - (1 - eta * c) ** H


print("x* =", xstar, float(xstar))
print("H, x_H, bias_formula, bias_direct, scaffold_fixed")
for H in (1, 2, 5, 20, 100):
    da, db = d(a, H), d(b, H)
    xH = (da * u + db * v) / (da + db)
    bias_direct = xH - xstar
    bias_formula = (u - v) * (b * da - a * db) / ((a + b) * (da + db))
    assert bias_direct == bias_formula
    # iterate the plain round map from x=3 in floats and check convergence
    x = 3.0
    fa, fb, fu, fv, fe = map(float, (a, b, u, v, eta))
    for _ in range(400):
        # local training
        y1 = x
        y2 = x
        for _ in range(H):
            y1 -= fe * fa * (y1 - fu)
            y2 -= fe * fb * (y2 - fv)
        x = x + 0.5 * ((y1 - x) + (y2 - x))
    # control variate version
    z = 3.0
    lam = 1.0
    for _ in range(4000):
        g = 0.5 * (fa * (z - fu) + fb * (z - fv))
        y1 = z
        y2 = z
        for _ in range(H):
            y1 -= fe * (fa * (y1 - z) + g)
            y2 -= fe * (fb * (y2 - z) + g)
        z = z + lam * 0.5 * ((y1 - z) + (y2 - z))
    stab = lam * float(a + b) * float(da / a + db / b) / 4
    print(H, float(xH), float(bias_formula), x - float(xstar), z, "stab_factor", stab)

limit = (b - a) * (u - v) / (2 * (a + b))
print("limit bias =", limit, float(limit))

# relay double counting
leaves = {"w1": (Fr(1), Fr(2)), "w2": (Fr(1), Fr(-1)), "w3": (Fr(2), Fr(3))}


def R(A):
    return (sum(leaves[i][0] * leaves[i][1] for i in A), sum(leaves[i][0] for i in A))


A, B = {"w1", "w2"}, {"w2", "w3"}
untagged = tuple(x + y for x, y in zip(R(A), R(B)))
true = R(A | B)
corrected = tuple(x + y - z for x, y, z in zip(R(A), R(B), R(A & B)))
print("untagged", untagged, "true", true, "inclusion-exclusion", corrected)
assert corrected == true and untagged != true

# fixed-point diagnostic D_H(x*): full-participation mean of the local shifts started at the reference solution
print()
print("H, D_H_formula, D_H_direct, D_H_control_variate, 2*D/(da+db) == bias")
expected = {1: "0.0000", 2: "-0.0480", 5: "-0.1690", 20: "-0.2954", 100: "-0.3000"}
for H in (1, 2, 5, 20, 100):
    da, db = d(a, H), d(b, H)
    # direct: exact local gradient steps from x*, then average the two shifts
    y1, y2 = xstar, xstar
    for _ in range(H):
        y1 -= eta * a * (y1 - u)
        y2 -= eta * b * (y2 - v)
    D_direct = ((y1 - xstar) + (y2 - xstar)) / 2
    D_formula = (u - v) * (b * da - a * db) / (2 * (a + b))
    assert D_direct == D_formula
    # the plain round map is affine with slope 1 - (da+db)/2, so the fixed point sits 2D/(da+db) from x*
    assert 2 * D_direct / (da + db) == (u - v) * (b * da - a * db) / ((a + b) * (da + db))
    # idealized control-variate round: every shift is proportional to F'(x*) = 0
    g = (a * (xstar - u) + b * (xstar - v)) / 2
    z1, z2 = xstar, xstar
    for _ in range(H):
        z1 -= eta * (a * (z1 - xstar) + g)
        z2 -= eta * (b * (z2 - xstar) + g)
    D_cv = ((z1 - xstar) + (z2 - xstar)) / 2
    assert g == 0 and D_cv == 0
    assert f"{float(D_direct):.4f}" == expected[H].replace("-0.0000", "0.0000"), (H, float(D_direct))
    if H == 1:
        assert D_direct == 0
    else:
        assert D_direct != 0
    print(H, float(D_formula), float(D_direct), float(D_cv), "ok")
