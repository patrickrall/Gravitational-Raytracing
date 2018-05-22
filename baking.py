from sympy import sqrt, sin, cos, acos, atan2, sec, cot
from sympy import symbols, simplify, lambdify, diff, Lambda
from sympy import ln
import dill, os

################################ Metrics ################################
metrics = {}

def flatmetric(x):
    g = [[0,0,0,0] for i in range(4)]

    g[0][0] = -1
    g[1][1] = 1
    g[2][2] = x[1]**2
    g[3][3] = x[1]**2 * sin(x[2])**2

    return g, ()
metrics["flat"] = flatmetric

def scmetric(x):
    g = [[0,0,0,0] for i in range(4)]

    M = symbols("M")

    g[0][0] = -(1-2*M/x[1])
    g[1][1] = 1/(1 - 2*M/x[1])
    g[2][2] = x[1]**2
    g[3][3] = x[1]**2 * sin(x[2])**2

    return g, (M, )
metrics["sc"] = scmetric

def kerrmetric(x):
    g = [[0,0,0,0] for i in range(4)]

    M, J = symbols("M, J")

    rs = 2*M #schwartzchild radius
    a = J/M

    Sigma = x[1]**2 + a**2 * cos(x[2])**2
    Delta = x[1]**2 - rs * x[1]  + a**2

    g[0][0] = -(1 - rs*x[1]/Sigma)
    g[1][1] = Sigma/Delta
    g[2][2] = Sigma
    g[3][3] = (x[1]**2 + a**2 + sin(x[2])**2 * rs*x[1]*a**2/Sigma)* sin(x[2])**2

    g[0][3] = - 2*rs*x[1]*a*sin(x[2])**2 / Sigma
    g[3][0] = g[0][3]

    return g, (M, J)
metrics["kerr"] = kerrmetric

def rnmetric(x):
    g = [[0,0,0,0] for i in range(4)]
    M, Q = symbols("M, Q")

    rs = 2*M
    rq = Q
    g[0][0] = -(1-rs/x[1] + rq**2/x[1]**2)
    g[1][1] = 1/(1- rs/x[1]+rq**2/x[1]**2)
    g[2][2] = x[1]**2
    g[3][3] = x[1]**2 * sin(x[2])**2

    return g, (M, Q)
metrics["rn"] = rnmetric

def gbmetric(x):
    g = [[0,0,0,0] for i in range(4)]
    h_pert = [[0,0,0,0] for i in range(4)]
    # epsilon = 0.01

    M, J, epsilon = symbols("M, J, e")

    rs = 2*M #schwartzchild radius
    a = J/M

    Sigma = x[1]**2 + a**2 * cos(x[2])**2
    Delta = x[1]**2 - rs * x[1]  + a**2

    F1 = -5*(x[1]-M)/(8*M*x[1]*(x[1]-rs))*(rs**2+6*M*x[1]-3*x[1]**2) - 15*x[1]*(x[1]-rs)/(16*M**2)*ln(x[1]/(x[1]-rs))
    F2 = 5*(rs**2-3*M*x[1]-3*x[1]**2)/(8*M*x[1]) + 15*(x[1]**2-rs**2)/(16*M**2)*ln(x[1]/(x[1]-rs))

    print(type(F1))
    g[0][0] = -(1 - rs*x[1]/Sigma) + epsilon*(1/(1-2*M/x[1])*((1-3*cos(x[2])**2))*F1)
    g[1][1] = Sigma/Delta + epsilon*((1 - 2*M/x[1])*(1-3*cos(x[2])**2)*F1)
    g[2][2] = Sigma + epsilon*((-1/(x[1])**2)*(1-3*cos(x[2])**2*F2))
    g[3][3] = (x[1]**2 + a**2 + sin(x[2])**2 * rs*x[1]*a**2/Sigma)* sin(x[2])**2 + epsilon*((-1/(x[1])**2*sin(x[2])**2)*(1-3*cos(x[2])**2*F2))

    g[0][3] = - 2*rs*x[1]*a*sin(x[2])**2 / Sigma
    g[3][0] = g[0][3]

    return g, (M, J, epsilon)
metrics["gb"] = gbmetric

################## Schwartzchild coordinates and metric #################

def defSphereToCart(x):
    y = [0,0,0,0]
    y[0] = x[0]
    y[1] = x[1]*sin(x[2])*cos(x[3])
    y[2] = x[1]*sin(x[2])*sin(x[3])
    y[3] = x[1]*cos(x[2])
    return y

def defCartToSphere(x):
    y = [0,0,0,0]
    y[0] = x[0]
    y[1] = sqrt(x[1]**2 + x[2]**2 + x[3]**2)
    y[2] = acos(x[3]/y[1])
    y[3] = atan2(x[2],x[1])
    return y



############################## Baking functions #######################
# These functions do all the derivatives using sympy
# then bake the symbolic expression into a quick-to-evaluate lambda

# bake a coordinate transformation
def bakeTrans(defn):
    a = symbols("a0, a1, a2, a3")
    out = defn(a)

    for i in range(4):
        out[i] = lambdify(a, simplify(out[i]))

    return lambda x: [out[0](*x), out[1](*x), out[2](*x), out[3](*x)]

# bake a coordinate transformation for velocity vectors
def bakeTransDeriv(defn):
    a = symbols("a0, a1, a2, a3")
    ad = symbols("ad0, ad1, ad2, ad3")

    trans = defn(a)

    out = [0,0,0,0]
    for i in range(4):
        for j in range(4):
            out[i] += diff(trans[i], a[j]) * ad[j]

    for i in range(4):
        out[i] = lambdify(a+ad, simplify(out[i]))

    return lambda x,dx: [out[0](*(x+dx)), out[1](*(x+dx)), out[2](*(x+dx)), out[3](*(x+dx))]

# bake a function that calculates xddot via geodesic eqn
# xddot^a = - Gamma^a_b_c xdot^b xdot^c
def bakeMetric(key):
    dill.settings['recurse'] = True

    if key+".metric" in os.listdir("."):
        print("Found pre-baked "+key)
        f = open(key+".metric", "rb")
        func = dill.load(f)
        f.close()
        return func

    print("Baking "+key)

    defn = metrics[key]

    a = symbols("a0, a1, a2, a3")
    ad = symbols("ad0, ad1, ad2, ad3")

    g,pars = defn(a)

    out = [0,0,0,0]
    for mu in range(4):
        for beta in range(4):
            for gamma in range(4):
                Gamma = diff(g[mu][beta], a[gamma])/2
                Gamma += diff(g[mu][gamma], a[beta])/2
                Gamma -= diff(g[beta][gamma], a[mu])/2
                Gamma /= g[mu][mu]

                out[mu] -= Gamma*ad[beta]*ad[gamma]

    for i in range(4):
        simp =  simplify(out[i])
        out[i] = lambdify(pars+a+ad, simp)

    func = lambda conf: (lambda x,dx: [out[0](*(conf+x+dx)), out[1](*(conf+x+dx)), out[2](*(conf+x+dx)), out[3](*(conf+x+dx))])
    f = open(key+".metric", "wb")
    dat = dill.dump(func, f)
    f.close()
    return func


############################## Sanity check #######################

# xddot from Corvin Zahn's 1990 phd thesis
def bookMetric(M):
    # metric as in the thesis
    a = symbols("t, r, th, ph")
    ad = symbols("dt, dr, dth, dph")
    t, r, th, ph = a
    dt, dr, dth, dph = ad

    out = [0,0,0,0]

    out[0] = - 2 * M * dr * dt / (r**2  * ( 1 - 2*M/r**2 ))

    out[1] = M * (dr**2 - dt**2) /( r**2  * ( 1 - 2*M/r**2 ))
    out[1] += (r - 2*M)* ( dth**2 + dph**2 * sin(th)**2)

    out[2] =  - 2*dr*dth / r   + sin(th) * cos(th) * dph**2
    out[3] =  - 2*dr*dph / r   - 2* cot(th) * dph * dth

    for i in range(4):
        out[i] = lambdify(a+ad, simplify(out[i]))

    return lambda x,dx: [out[0](*(x+dx)), out[1](*(x+dx)), out[2](*(x+dx)), out[3](*(x+dx))]
