from baking import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sympy import symbols
from multiprocessing import Pool
from functools import partial

np.seterr(all='raise')

################### Bake required functions ################
print("baking...")

M = 1
J = 1
Q = 0.9
epsilon = 0.01

# key, opts = "flat", []
# key, opts = "sc", [M]
# key, opts = "rn", [M, Q]
# key, opts = "kerr", [M, J]
key, opts = "gb", [M, J, epsilon]

ddxRaw = bakeMetric(key)

s2c = bakeTrans(defSphereToCart)
ds2c = bakeTransDeriv(defSphereToCart)
c2s = bakeTrans(defCartToSphere)
dc2s = bakeTransDeriv(defCartToSphere)


print("tracing...")

# mode, pixels = "image", 100
mode = "rays"

if key == "flat": keyTitle = "Flat Spacetime"
if key == "sc": keyTitle = "Schwarzschild Black Hole"
if key == "rn": keyTitle = "Reissner-Nordstrom Black Hole"
if key == "gb": keyTitle = "Neutron Star"
if key == "kerr": keyTitle = "Kerr Black Hole"

shift = 0

# titleOverride = None
# titleOverride = "Neutron Star Lensing. J = "+str(J) +". $\epsilon$="+str(epsilon)
titleOverride = "Neutron Star Ray Diagram. J = "+str(J) +". $\epsilon$="+str(epsilon)
# titleOverride = "Kerr Black Hole Ergosphere. J = "+str(J) #+". Shift = "+str(shift)+"$M$"
# titleOverride = "Fixed Integration Rate"
# titleOverride = "$\\theta$-Dependent Rate"


##################### Ray Tracing Utilties ################
# determine the horizon based on options
def getHorizons(opts):
    if key == "flat": return [0,0]
    if key == "sc": return [2*opts[0], 2*opts[0]]
    if key == "rn":
        horizon1 = 0.5*(2*opts[0] - sqrt((2*opts[0])**2-4*(opts[1])**2))
        horizon2 = 0.5*(2*opts[0] + sqrt((2*opts[0])**2-4*(opts[1])**2))
        return [horizon1, horizon2]

    if key == "gb" or key == "kerr":
        horizon1 = 2*opts[0]
        horizon2 = 0.5*(2*opts[0] + sqrt((2*opts[0])**2-4*(opts[1])**2))
        return [horizon1,horizon2]

horizons = getHorizons(opts)

# decide if a ray should terminate
# determine its color
def terminateRay(x, opts):

    if x[1] <= horizons[0]+0.01:
        return (0.0,0.0,0.0) # event horizon


    if mode == "image" and False: # accretion disk
        if x[1] < 5 and x[1] > horizons[1]:
            if np.abs(x[2] - np.pi/2) < 0.05:
                n = 21
                j = 0
                while x[1] > j*5/n: j+=1

                if (j)%2 == 0: return (0.9,0.9,0.9)
                else: return (0.8,0.8,0.8)

    if x[1] > 8:
        if True: # paint a patch behind the object bright
            ypos = opts["ypos"]
            cart = s2c(x)
            if (cart[2] - ypos)**2 + (cart[3])**2 < 2*M and cart[1] < 0:
                return (0.9,0.9,0.9)  # star behind

        if mode == "image": # alternating stripes in theta
            n = 9
            i = 0
            j = 0

            while x[2] < 0: x[2] += np.pi
            # while x[2] >= np.pi: x[2] -= np.pi
            while x[3] < 0: x[3] += 2*np.pi
            # while x[3] >= 2*np.pi: x[3] -= 2*np.pi

            while x[2] > i*np.pi/n: i+=1
            while x[3] > j*np.pi/n: j+=1

            if (i+j)%2 == 0: return (0.7,0.7,0.7)
            else: return (0.5,0.5,0.5)
        else: return (0.5,0.5,0.5)


    return None

# do the ray tracing, optionally tracking points encountered
# opts is just passed to terminateRay()
def doTrace(x, dx, ddxGet, opts, track=False):
    if track:
        cart = s2c(x)
        xs = [cart[1]]
        ys = [cart[2]]
        zs = [cart[3]]

    count = 0
    while True:
        end = None
        count += 1

        try:
            ddx = ddxGet(x, dx)

            theta = x[2]
            if theta > np.pi/2: theta = np.pi - theta

            step = 0.05

            if True:
                if theta < 0.3: step = 0.01
                if theta < 0.2: step = 0.001
                if theta < 0.1: step = 0.0001
                if theta < 0.08: step = 0.0001

            # ergosphere dynamics
            if key == "kerr" and x[1] < horizons[0]+0.01: step = 0.0001

            #green: numerical instability
            while False:
                ok = True
                if np.abs((dx[1] + ddx[1]*step)*step) > 0.05: ok = False
                if np.abs((dx[2] + ddx[2]*step)*step) > 0.01: ok = False
                if np.abs((dx[3] + ddx[3]*step)*step) > 0.01: ok = False
                if ok: break

                step *= 0.8
                if step < 0.0001:
                    # end = (0.0,1.0,0.0)
                    break

            dx = [dx[j] + ddx[j]*step for j in range(4)]
            x = [x[j] + dx[j]*step for j in range(4)]

            if end is None: end = terminateRay(x, opts)
        except:
            end = (1.0,0.0,0.0) #red: crashed

        if end is not None or count > 100000:
            if end is None: end = (0.0,0.0,1.0) #blue: took too long to trace
            if not track: return end
            return xs, ys, zs, end

        if track:
            cart = s2c(x)
            xs.append(cart[1])
            ys.append(cart[2])
            zs.append(cart[3])

def camiter(y, z, f, x, func):
    l = np.sqrt(f**2 + y**2 + z**2)
    dx = [1, -f/l, -y/l, -z/l]
    return func(c2s(x),dc2s(x,dx))

def camloop(ypixels, zpixels, func, parallel=False):
    x = [0,7,0,0] # focal point

    f = 1 # focal length
    w = 3 # width of film

    if isinstance(zpixels, int):
        zpixels = np.linspace(0,w/2,int(zpixels/2))
        if len(zpixels) == 1: zpixels = [0]
    if isinstance(ypixels, int):
        ypixels = np.linspace(-w/2,w/2,ypixels)
        if len(ypixels) == 1: ypixels = [0]

    data = []
    for z in zpixels:
        if parallel:
            pool = Pool(8)
            row = pool.map(partial(camiter, z=z, f=f, x=x, func=func), ypixels)
            pool.close()
        else:
            row = []
            for y in ypixels:
                row.append(camiter(y,z,f,x,func))
        data.append(row)

    if len(zpixels) > 1:
        return data[::-1] + data
    else:
        return data


# animated diagram of target behind black hole
if __name__ == "__main__" and mode == "image":
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    numscreens = 1

    def getOpts(i):
        return opts, {"ypos": shift}

        # Js = np.linspace(0, 1, numscreens)
        # return [M, Js[i]], {"ypos":0}

        # yposs = np.linspace(-5, 5, numscreens)
        # return opts, {"ypos": yposs[i]}

    def getTitle(i, opts, params):
        if titleOverride is not None: return titleOverride
        return keyTitle+" Camera Image"
        # return titleOverride + str(params["ypos"])
        # return "Kerr metric J = " v+ str(opts[1])
        # return "ypos = "+str(params["ypos"])

    screens = []

    ddxGet = ddxRaw(opts)

    for i in range(numscreens):
        thisOpts, params = getOpts(i)

        ddxGet = ddxRaw(thisOpts)
        title = getTitle(i, thisOpts, params)
        print(title)

        # action per pixel: obtain color via dotrace
        def action(x,dx):
            return doTrace(x, dx, ddxGet, params)

        screens.append(camloop(pixels, pixels, action, parallel=True))

    im = ax.imshow(screens[0])
    thisOpts, params = getOpts(0)
    ax.set_title(getTitle(0, thisOpts, params))

    # animation function
    def update(i):
        im.set_data(screens[i])
        thisOpts, params = getOpts(i)
        ax.set_title(getTitle(i, thisOpts, params))

    anim = FuncAnimation(fig, update, frames=range(numscreens), interval=500)

    # save gif
    anim.save('lensing.gif', dpi=80, writer='imagemagick')

    # show
    plt.show()


# diagram showing paths of light
if __name__ == "__main__" and mode == "rays":

    rows = 1
    cols = 1

    fig, axs = plt.subplots(rows, cols)

    if rows == 1: axs = [axs]
    if cols == 1: axs = [[ax] for ax in axs]

    def getOpts(row, col, i):
        return opts
        Js = np.linspace(0, 0.5, rows*cols)
        return [M, Js[i]]

    def getTitle(row, col, i, opts):
        if titleOverride is not None: return titleOverride
        return keyTitle+" Ray Diagram"
        return "J = "+str(opts[1])

    # plt.suptitle("Increasing Charge in Reissner-Nordstrom Metric")

    index = -1
    for row in range(rows):
        for col in range(cols):
            index += 1
            thisOpts = getOpts(row,col,index)

            ddxGet = ddxRaw(thisOpts)
            title = getTitle(row,col,index,thisOpts)
            print(title)

            ax = axs[row][col]

            # action per pixel: track and plot beam
            def action(x,dx):
                xs, ys, zs, end = doTrace(x,dx, ddxGet, {"ypos": shift}, track=True)
                # ax.scatter(xs,ys, c=end, s=0.1)
                ax.plot(xs,ys, c=end)

                return None

            # camloop(100, [0], action)
            camloop(100, [0.40], action)
            # camloop(100, [0.55], action)

            plotsize = 10

            ax.set_xlim([-plotsize,plotsize])
            ax.set_ylim([-plotsize,plotsize])

            ax.set_title(title)

            # draw a circle at event horizon
            horizons = getHorizons(thisOpts)

            cxs1, cys1, cxs2, cys2 = [], [], [], []
            for theta in np.linspace(0,2*np.pi,100):
                cxs1.append(horizons[0]*np.sin(theta))
                cys1.append(horizons[0]*np.cos(theta))
                cxs2.append(horizons[1]*np.sin(theta))
                cys2.append(horizons[1]*np.cos(theta))
            ax.plot(cxs1,cys1, c="red")
            ax.plot(cxs2,cys2, c="green")
            ax.grid(True)

    plt.show()
