import fitsio
import matplotlib.pyplot as plt
import numpy
import skimage
from skimage.filters import sobel, gaussian
import glob
# import PyGuide

from skimage.measure import regionprops, label
from sklearn.cluster import KMeans
from skimage.feature import blob_log, blob_dog, blob_doh
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform

from fibermeas.plotutils import plotCircle, imshow
from coordio.fitData import TransRotScaleModel, ModelFit, QuadrupoleModel
import time
import os

import seaborn as sns

cam1Shape = (3660, 5488)
cam2Shape = (3660, 5484)

rowPix = cam1Shape[0]
colPix = cam1Shape[1]

rowCen = int(rowPix/2)
colCen = int(colPix/2)

MaxCounts = 4095

# 0's indicate valid pixels
mask = numpy.ones((rowPix, colPix))
mask[:3020, 650:4585] = 0

# 1's indicate valid pixels
invmask = numpy.zeros((rowPix, colPix))
# invmask[:3020, 650:4585] = 1

invmask[rowCen-1200:rowCen+1201,colCen-1200:colCen+1201] = 1
# invmask[:, 819:4742] = 1
# invmask[:,:] = 1

# https://www.edmundoptics.com/p/150-x-150mm-1000mm-spacing-opal-distortion-target/18703/
# 63-991
# 1 mm spacing
# 0.5 diameter
# .001 center to center
# .003 grid corner to grid corner
# grid size is 150 x 150

# camera BAS ACA5472-17UM
# pixel size = 2.4 micron
# lens FUJ HF3520-12M

# https://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions

# SK mediandots_0.005.fits might be best

# note that PosMinusIndex in pyGuide needs to be 0 to match
# skimage measurment conventions...

# http://wyant.optics.arizona.edu/zernikes/Zernikes.pdf

# http://www.dm.unibo.it/home/citti/html/AnalisiMM/Schwiegerlink-Slides-Zernike.pdf


def getImgData(imgName):
    data = fitsio.read(imgName)
    data = skimage.util.invert(data)
    data = data - numpy.min(data)
    return data


def centroidSkimage(img, plot=True):
    # unlike pyguide, now 0,0 is corner of LL pixel

    data = getImgData(img)

    if plot:
        plt.figure()
        imshow(data)
        plt.show()

    data = data * invmask


    # data = data[rowCen-150:rowCen+150, colCen-150:colCen+150] # center
    # data = data[2775:2775+150, 700:700+150] # top left corner
    # thresh = (data > numpy.percentile(data, 95))

    if plot:
        plt.figure()
        print("data stats", numpy.median(data), numpy.mean(data), numpy.std(data))
        plt.hist(data.flatten())
        plt.show()

    thresh = (data > numpy.max(data) - 50) # 1000 counts below max

    # thresh = data > 2*numpy.std(data)

    # plt.figure()
    # imshow(thresh)
    # plt.show()

    t1 =  time.time()
    labels = label(thresh)
    print("%s labels took %.2f seconds" % (img, time.time()-t1))
    t1 = time.time()
    props = regionprops(labels, data)
    print("%s props took %.2f seconds" % (img, time.time()-t1))

    print("found %i props"%len(props))

    xCent = []
    yCent = []
    ecen = []
    rad = []

    for region in props:
        _yCent, _xCent = region.weighted_centroid
        _ecen = region.eccentricity
        _rad = region.equivalent_diameter/2
        xCent.append(_xCent)
        yCent.append(_yCent)
        ecen.append(_ecen)
        rad.append(_rad)

    centroidData = numpy.array([xCent, yCent, ecen, rad]).T


    # lq = numpy.percentile(centroidData[:,-1], 25)
    # uq = numpy.percentile(centroidData[:,-1], 99)

    # 2 sigma cut to throw out outliers

    # ecenCut = centroidData[:,2] < 0.4
    # centroidData = centroidData[ecenCut]

    # # radius cut
    # meanRad = numpy.mean(centroidData[:,-1])
    # stdRad = numpy.std(centroidData[:,-1])

    # keep = numpy.abs(centroidData[:,-1]-meanRad) < 1.5*stdRad
    # centroidData = centroidData[keep]

    # radCut = centroidData[:,-1] > lq
    # centroidData = centroidData[radCut]

    # radCut = centroidData[:,-1] < uq
    # centroidData = centroidData[radCut]

    if plot:
        plt.figure()
        plt.hist(centroidData[:, 2], bins=100)
        plt.title("ecen")

        plt.figure()
        plt.hist(centroidData[:, 3], bins=100)
        plt.title("rad")

        plt.show()
        # import pdb; pdb.set_trace()

        t1 = time.time()
        plt.figure(figsize=(8,8))
        plt.title(img)
        imshow(sobel(data))  # extent is set correctly so centers line up
        for _x, _y, _e, _r in centroidData:
            plotCircle(_x, _y, 0.8*_r)
        print("plotting took", time.time()-t1)
        plt.show()

    # import pdb; pdb.set_trace()



    # save data
    numpy.savetxt(img + ".centroids", centroidData)

    #     t1 = time.time()
    #     print(img)
    #     filename = "SK_%s.txt"%img
    #     with open(filename, "w") as f:
    #         for region in props:
    #             _yCent, _xCent = region.centroid
    #             yCent, xCent = region.weighted_centroid
    #             centErr.append(numpy.sqrt((yCent-_yCent)**2+(xCent-_xCent)**2))
    #             ecen.append(region.eccentricity)
    #             rad = region.equivalent_diameter/2
    #             f.write("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n"%(xCent, yCent, _xCent, _yCent, region.eccentricity, rad))
    #             plotCircle(xCent, yCent, rad)
    #         print("plotting took", time.time()-t1)
    #         print("mean cent error", numpy.mean(centErr))
    #         print("mean eccentricity", numpy.mean(ecen))
    #         print("")

    # plt.show()
        # break


def associate(img):
    centFile = img+".centroids"
    if not os.path.exists(centFile):
        raise RuntimeError("must centroid data first")

    centroidData = numpy.loadtxt(centFile)

    # find the central dot
    x = centroidData[:,0]
    y = centroidData[:,1]

    meanX = numpy.mean(x)
    meanY = numpy.mean(y)

    midInd = numpy.argmin(numpy.sqrt((x-meanX)**2+(y-meanY)**2))

    xMid = x[midInd]
    yMid = y[midInd]

    # find nearest neighbor at positive x
    sortDist = numpy.argsort(numpy.sqrt((x-xMid)**2+(y-yMid)**2))

    for sortInd in sortDist[1:]:
        # skip first (its the actual point)
        _x = x[sortInd]
        _y = y[sortInd]

        if _x < xMid:
            continue
        if numpy.abs(_y-yMid) > 10:
            continue

        rightInd = sortInd
        break

    xRight = x[rightInd]
    yRight = y[rightInd]
    dx = xRight - xMid
    dy = yRight - yMid

    roughScale = 1/numpy.sqrt((dx)**2+(dy)**2) # mm/pixel
    roughAngle = numpy.arctan2(dy,dx)

    print("rough angle", numpy.degrees(roughAngle))

    rotMat = numpy.array([
        [numpy.cos(roughAngle), -numpy.sin(roughAngle)],
        [numpy.sin(roughAngle), numpy.cos(roughAngle)]
    ])

    # rotate and scale centroids, about central point
    # to put units into roughly mm and aligned with xy
    x = x - xMid
    y = y - yMid

    x *= roughScale
    y *= roughScale

    xy = numpy.array([x,y])

    _xy = rotMat @ xy

    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.hist(xy[1], bins=1000)
    ax2.hist(_xy[1], bins=1000)
    plt.show()

    xy = _xy.T

    xyAssoc = []
    for x, y in xy:
        _rx = numpy.round(x)
        _ry = numpy.round(y)
        if numpy.abs(x-_rx) > 0.2:
            print("skipping centroid")
            continue
        if numpy.abs(y-_ry) > 0.2:
            print("skipping centroid")
            continue
        xyAssoc.append([x,y,_rx,_ry])

    xyAssoc = numpy.array(xyAssoc)

    err = xyAssoc[:,2:] - xyAssoc[:,:2]
    plt.figure(figsize=(9,9))
    plt.quiver(xyAssoc[:,0], xyAssoc[:,1], err[:,0], err[:,1], scale=1.5)
    plt.show()
    # sort xy by
    numpy.savetxt(img+".assoc", xyAssoc)

    # import pdb; pdb.set_trace()




    # topLX, topLY = 664.61489, 2980.8192  # top left dot
    # r1 = 7.0223764

    # x2, y2, = 4538.7292, 2994.2876 # top right dot
    # r2 = 6.4945258
    # # scale in mm/pixel
    # roughScale = 149/numpy.linalg.norm([topLX-x2, topLY-y2])
    # # angle
    # _x = x2-topLX
    # _y = y2-topLY
    # roughAngle = numpy.arctan2(_y, _x)


    # # middle of chip
    # x1, y1 = 2453.7021, 1308.4415
    # x2, y2 = 2608.5536, 1309.2168
    # roughScale = 6/numpy.linalg.norm([x1-x2, y1-y2])




    # rotMat = numpy.array([
    #     [numpy.cos(-roughAngle), -numpy.sin(-roughAngle)],
    #     [numpy.sin(-roughAngle), numpy.cos(-roughAngle)]
    # ])

    # for centType in ["SK"]:
    #     for img in imgs:
    #         if img != bestfile:
    #             continue
    #         print("")
    #         print("")
    #         print(centType, img)
    #         filename = "%s_%s.txt" % (centType, img)
    #         centData = parseFile(filename)[:, 2:4]  # just xy
    #         print(centData.shape)
    #         topLeft = numpy.array([[topLX,topLY]]*len(centData))
    #         print(topLeft.shape)
    #         diff = centData - topLeft
    #         dist = numpy.linalg.norm(diff, axis=1)
    #         arg = numpy.argmin(dist)
    #         if dist[arg] > 5: # within 5 pixels
    #             print("cannot find top left dot skipping")
    #             continue  # closer that 5 pixels

    #         # scale data to roughly mm
    #         centData = centData * roughScale

    #         # set origin to top left dot
    #         centData = centData - centData[arg,:]

    #         # rotate by rough angle
    #         centData = rotMat.dot(centData.T).T

    #         xyExpect = []
    #         xyMeas = []
    #         missingDots = 0
    #         for yDot in range(150):
    #             yDot = -1*yDot
    #             nXs = 0
    #             for xDot in range(150):
    #                 xyTest = numpy.array([xDot,yDot])
    #                 diff = centData - xyTest
    #                 dist = numpy.linalg.norm(diff, axis=1)
    #                 arg = numpy.argmin(dist)
    #                 if dist[arg] > 1:  # throw out measurements > 800 micron error, they're no good
    #                     # print("missed dot, continuing")
    #                     missingDots += 1
    #                     continue
    #                 nXs += 1
    #                 xyExpect.append(numpy.array([xDot, yDot]))
    #                 xyMeas.append(numpy.array(centData[arg,:]))
    #             if yDot == 0:
    #                 print("got %i xs"%nXs)

    #         xyMeas = numpy.array(xyMeas)
    #         xyExpect = numpy.array(xyExpect)

    #         print("found", len(xyMeas), "dots")
    #         print("missing", missingDots, "dots")

    #         dxy = xyMeas - xyExpect

    #         dx = dxy[:,0]
    #         dy = dxy[:,1]

    #         # plt.figure()
    #         # plt.hist(numpy.sqrt(dx**2+dy**2)*1000, bins=100)
    #         # plt.show()

    #         rms = numpy.sqrt(numpy.mean(dx**2+dy**2))
    #         print("rms micron", rms*1000)

    #         plt.figure(figsize=(9,9))
    #         plt.title(filename)
    #         plt.quiver(xyMeas[:,0], xyMeas[:,1], dx, dy, angles="xy", scale=20)
    #         plt.axis("equal")

    #         nPts = len(xyMeas)
    #         assocArray = numpy.zeros((nPts, 4))
    #         assocArray[:,0] = xyMeas[:,0]
    #         assocArray[:,1] = xyMeas[:,1]
    #         assocArray[:,2] = xyExpect[:,0]
    #         assocArray[:,3] = xyExpect[:,1]
    #         numpy.savetxt(filename+".npy", assocArray)



def zerns(x, y):
    # from: https://doi.org/10.1364/JOSAA.35.000840
    # stopped at s17 because there is a typo in the paper?
    out = numpy.array([
        [1, 0], #s2
        [0, 1], #s3
        numpy.sqrt(3)*numpy.array([x, y]), #s4
        numpy.sqrt(3)*numpy.array([y, x]), #s5
        numpy.sqrt(3)*numpy.array([x, -y]), #s6
        numpy.array([6*x*y, 3*x**2+9*y**2-2])/numpy.sqrt(3),  # s7
        numpy.array([9*x**2+3*y**2-2, 6*x*y])/numpy.sqrt(3),  # s8
        numpy.array([12*x*y, 6*x**2-12*y**2+1])/(2*numpy.sqrt(2)),  # s9
        numpy.array([12*x**2-6*y**2-1, -12*x*y])/numpy.sqrt(8),  # s10
        numpy.sqrt(21/62)*numpy.array([x, y])*(15*x**2+15*y**2-7),  # s11
        numpy.sqrt(7)*numpy.array([x*(10*x**2-3), y*(3-10*y**2)])/2,  # s12
        numpy.sqrt(21/38)*numpy.array([x*(15*x**2+5*y**2-4), y*(5*x**2+15*y**2-4)]),  # s13

        numpy.array([x*(35*x**2-27*y**2-6), y*(-27*y**2+35*x**2-6)])/numpy.sqrt(5/62),  # s14

        numpy.sqrt(35/3)*numpy.array([y*(3*x**2-y**2), x*(x**2-3*y**2)]),  # s15

        numpy.array([
            315*(x**2+y**2)*(5*x**2+y**2)-30*(33*x**2+13*y**2)+83,
            60*x*y*(21*(x**2+y**2)-13)
        ])/(2*numpy.sqrt(1077)),  # s16

        numpy.array([
            60*x*y*(21*(x**2+y**2)-13),
            315*(x**2+y**2)*(x**2+5*y**2) - 30*(13*x**2+33*y**2) + 83
        ])/(2*numpy.sqrt(1077)), # s17
        3*numpy.array([
            140*(860*x**4-45*x**2*y**2-187*y**4) - 30*(1685*x**2-522*y**2) + 1279,
            -40*x*y*(105*x**2+2618*y**2-783)
        ])/(2*numpy.sqrt(2489214)), # s18
        3*numpy.array([
            40*x*y*(2618*x**2+105*y**2-783),
            140*(187*x**4+45*x**2*y**2-860*y**4) - 30*(522*x**2-1685*y**2) - 1279
        ])/(2*numpy.sqrt(2489214)), # s19
        (1/16)*numpy.sqrt(7/13557143)*numpy.array([
            60*(10948*x**4-7830*x**2*y**2+2135*y**4-3387*x**2-350*y**2)+11171,
            -1200*x*y*(261*x**2-427*y**2+35)
        ]), # s20
        (1/16)*numpy.sqrt(7/13557143)*numpy.array([
            1200*x*y*(427*x**2-261*y**2+35),
            60*(2135*x**4-7830*x**2*y**2+10948*y**4-350*x**2-3387*y**2) + 11171
        ]) # s21
    ])

    return out


def zerns2(x, y, MaxOrd):
    # from https://doi.org/10.1364/OE.26.018878
    #  Pseudo-code to calculate unit-normalized Zernike polynomials and their x,y derivatives
    #
    #    Numbering scheme:
    #    Within a radial order, sine terms come first
    #            ...
    #          sin((n-2m)*theta)   for m = 0,..., [(n+1)/2]-1
    #            ...
    #             1                for n even, m = n/2
    #            ...
    #          cos((n-2m)*theta)   for m = [n/2]+1,...,n
    #            ...
    #
    #    INPUT:
    #    x, y normalized (x,y) coordinates in unit circle
    #    MaxOrd: Maximum Zernike radial order
    #
    #    OUTPUT:
    #    Zern[...]   array to receive value of each Zernike polynomium at (x,y)
    #    dUdx[...]   array to receive each derivative dU/dx at (x,y)
    #    dUdy[...]   array to receive each derivative dU/dy at (x,y)
    #
    # int MaxOrd
    # preallocate space
    nTerms = numpy.sum(numpy.arange(MaxOrd + 2)) + 1
    Zern = numpy.zeros(nTerms)
    dUdx = numpy.zeros(nTerms)
    dUdy = numpy.zeros(nTerms)

    if hasattr(x, "__len__"):
        # x, y are vectors
        Zern = numpy.array([Zern] * len(x)).T
        dUdx = numpy.array([dUdx] * len(x)).T
        dUdy = numpy.array([dUdy] * len(x)).T

    # double x, y, Zern[*], dUdx[*], dUdy[*]

    # int nn, mm, kndx, jbeg, jend, jndx, even, nn1, nn2
    # int jndx1, jndx11, jndx2, jndx21
    # double pval, qval

    # pseudocode is 1 indexed, ugh, stick with it
    # and modify later

    Zern[1] = 1.                                  # (0,0)
    dUdx[1] = 0.                                  # (0,0)
    dUdy[1] = 0.                                  # (0,0)

    Zern[2] = y                                   # (1,0)
    Zern[3] = x                                   # (1,1)
    dUdx[2] = 0.                                  # (1,0)
    dUdx[3] = 1.                                  # (1,1)
    dUdy[2] = 1.                                  # (1,0)
    dUdy[3] = 0.                                  # (1,1)

    kndx = 1                # index for term from 2 orders down
    jbeg = 2                # start index for current radial order
    jend = 3                # end index for current radial order
    jndx = 3                # running index for current Zern
    even = -1
    #  Outer loop in radial order index
    for nn in range(2, MaxOrd + 1):  # 1 indexed
        even = -even          # parity of radial index
        jndx1 = jbeg           # index for 1st ascending series in x
        jndx2 = jend           # index for 1st descending series in y
        jndx11 = jndx1 - 1      # index for 2nd ascending series in x
        jndx21 = jndx2 + 1      # index for 2nd descending series in y
        jbeg = jend + 1       # end of previous radial order +1
        nn2 = nn // 2
        nn1 = (nn - 1) // 2
        #  Inner loop in azimuthal index
        for mm in range(0, nn + 1):  # 1 indexed
            jndx += 1                  # increment running index for current Zern

            if (mm == 0):
                Zern[jndx] = x * Zern[jndx1] + y * Zern[jndx2]
                dUdx[jndx] = Zern[jndx1] * nn
                dUdy[jndx] = Zern[jndx2] * nn

            elif (mm == nn):
                Zern[jndx] = x * Zern[jndx11] - y * Zern[jndx21]
                dUdx[jndx] = Zern[jndx11] * nn
                dUdy[jndx] = -Zern[jndx21] * nn

            elif ((even > 0) and (mm == nn2)):              # logical “AND”
                Zern[jndx] = 2. * (x * Zern[jndx1] + y * Zern[jndx2]) - Zern[kndx]
                dUdx[jndx] = 2. * nn * Zern[jndx1] + dUdx[kndx]
                dUdy[jndx] = 2. * nn * Zern[jndx2] + dUdy[kndx]
                kndx += 1                        # increment kndx

            elif ((even < 0) and (mm == nn1)):              # logical “AND”
                qval = Zern[jndx2] - Zern[jndx21]
                Zern[jndx] = x * Zern[jndx11] + y * qval - Zern[kndx]
                dUdx[jndx] = Zern[jndx11] * nn + dUdx[kndx]
                dUdy[jndx] = qval * nn + dUdy[kndx]
                kndx += 1                        # increment kndx

            elif ((even < 0) and (mm == nn1 + 1)):            # logical “AND”
                pval = Zern[jndx1] + Zern[jndx11]
                Zern[jndx] = x * pval + y * Zern[jndx2] - Zern[kndx]
                dUdx[jndx] = pval * nn + dUdx[kndx]
                dUdy[jndx] = Zern[jndx2] * nn + dUdy[kndx]
                kndx += 1                        # increment kndx

            else:
                pval = Zern[jndx1] + Zern[jndx11]
                qval = Zern[jndx2] - Zern[jndx21]
                Zern[jndx] = x * pval + y * qval - Zern[kndx]
                dUdx[jndx] = pval * nn + dUdx[kndx]
                dUdy[jndx] = qval * nn + dUdy[kndx]
                kndx += 1                        # increment kndx

            jndx11 = jndx1                   # update indices
            jndx1 += 1
            jndx21 = jndx2
            jndx2 -= 1
            # End of inner azimuthal loop

        jend = jndx
        # print("jend", jend)
        # End of outer radial order loop

    # deal with indexing, throw out first and cut at last
    # computed term

    # Zern = Zern[1:jend + 1]
    # dUdx = dUdx[1:jend + 1]
    # dUdy = dUdy[1:jend + 1]

    # throw out first term (zero indexing didn't populate it)
    Zern = Zern[1:]
    dUdx = dUdx[1:]
    dUdy = dUdy[1:]

    return Zern, dUdx, dUdy


def fit(img):
    # https://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html

    filename = img+".assoc" #"SK_mediandots_0.005.fits.txt.npy"
    data = numpy.loadtxt(filename)
    xyMeas = data[:,:2]
    xyExpect = data[:,2:]

    # meanXY = numpy.mean(xyMeas, axis=0)
    # # meanXY = numpy.array([65, -35])  # roughly center of divergence?
    # xyMeas = xyMeas - meanXY
    # xyExpect = xyExpect - meanXY

    # # use the plot to find the center of divergence...
    err = xyExpect - xyMeas
    # magErr = numpy.linalg.norm(err, axis=1)

    # # throw out outliers
    # keep = magErr*1000 < 190
    # xyMeas = xyMeas[keep]
    # print("threw out", len(xyExpect) - len(xyMeas))
    # xyExpect = xyExpect[keep]
    # err = err[keep]

    # make it circular
    # _r = numpy.linalg.norm(xyMeas, axis=1)
    # maxRad = numpy.max(xyMeas[:,1])*0.8  # max y value
    # keep = _r <= maxRad
    # xyMeas = xyMeas[keep]
    # xyExpect = xyExpect[keep]
    # err = err[keep]

    plt.figure(figsize=(9,9))
    plt.quiver(xyMeas[:,0], xyMeas[:,1], err[:,0], err[:,1], angles="xy", scale=2)
    plt.axis("equal")
    plt.title("raw errors")
    plt.show()

    # xMid = 81.5
    # yMid = -72.5

    # xyMeas = xyMeas - numpy.array([xMid, yMid])
    # xyExpect = xyExpect - numpy.array([xMid, yMid])

    print("RMS err prefit (micron)", numpy.sqrt(numpy.mean(err**2))*1000)


    if True:
        model = AffineTransform()
        # model = EuclideanTransform()
        # model = SimilarityTransform()
        t1 = time.time()
        isOK = model.estimate(xyMeas, xyExpect)
        print("affine fit took", time.time()-t1)
        if not isOK:
            raise RuntimeError("affine fit failed")
        xyFitAff = model(xyMeas)

        print("model", model.translation, numpy.degrees(model.rotation), model.scale)
    else:
        xyFitAff = xyMeas


    err = xyExpect - xyFitAff
    print("RMS err postfit (micron)", numpy.sqrt(numpy.mean(err**2))*1000)
    # import pdb; pdb.set_trace()

    vecScale = 1
    plt.figure(figsize=(9,9))
    plt.quiver(xyFitAff[:,0], xyFitAff[:,1], err[:,0], err[:,1], angles="xy", scale=vecScale)
    plt.axis("equal")
    plt.title("affine fit")
    plt.show()


    # normalize everything to unit circle for zerns
    # figure out radial scale
    rFitAff = numpy.linalg.norm(xyFitAff, axis=1)
    tFitAff = numpy.arctan2(xyFitAff[:,1], xyFitAff[:,0])

    rExpect = numpy.linalg.norm(xyExpect, axis=1)
    tExpect = numpy.arctan2(xyExpect[:,1], xyExpect[:,0])

    rScale = 1/numpy.max(rFitAff)
    rFitAff *= rScale
    rExpect *= rScale

    xyFitUnit = numpy.array([
        rFitAff*numpy.cos(tFitAff),
        rFitAff*numpy.sin(tFitAff)
    ]).T

    xyExpectUnit = numpy.array([
        rExpect*numpy.cos(tExpect),
        rExpect*numpy.sin(tExpect)
    ]).T

    err = xyExpectUnit - xyFitUnit

    print("RMS err postfit scaled (micron)", numpy.sqrt(numpy.mean(err**2))*1000/rScale)
    # import pdb; pdb.set_trace()

    plt.figure(figsize=(9, 9))
    plt.quiver(xyFitUnit[:, 0], xyFitUnit[:, 1], err[:, 0], err[:, 1], angles="xy", scale=vecScale*rScale)
    plt.axis("equal")
    plt.title("Unit-ified")

    xErr = err[:,0]
    yErr = err[:,1]
    xyErr = numpy.hstack((xErr,yErr))

    print("mean/std xyErr", numpy.mean(xyErr), numpy.std(xyErr))

    xUnit = xyFitUnit[:,0]
    yUnit = xyFitUnit[:,1]

    if False:
        nRadTerms = 4
        t1 = time.time()
        zern, dx, dy = zerns2(xUnit, yUnit, nRadTerms)
        dx = dx[3:]
        dy = dy[3:]
        zern = zern[3:]

        mdx = numpy.mean(dx, axis=0)
        stdx = numpy.std(dx, axis=0)
        dx = (dx - mdx)/stdx

        mdy = numpy.mean(dy, axis=0)
        stdy = numpy.std(dy, axis=0)
        dy = (dy - mdy)/stdy

        mxyErr = numpy.mean(xyErr)
        stdErr = numpy.std(xyErr)
        xyErr = (xyErr - mxyErr)/stdErr

        print("zerns took", time.time()-t1)

        dxy = numpy.hstack((dx,dy)).T


        t1 = time.time()

        coef, resid, rank, s = numpy.linalg.lstsq(dxy, xyErr)
        print("zern fit took", time.time()-t1)

        plt.figure()
        plt.plot(numpy.arange(len(coef)), coef)
        plt.title("coeffs")

        dxFit = dx.T @ coef * stdErr + mxyErr
        dyFit = dy.T @ coef * stdErr + mxyErr
        zernFit = zern.T @ coef

        # plt.figure()
        # plt.hexbin(xUnit, yUnit, zernFit, gridsize=90)
        # plt.title("fit zern expansion (20 terms)")
        # sns.scatterplot(x=xUnit, y=yUnit, hue=zernFit)

        plt.figure(figsize=(9,9))
        plt.quiver(xUnit, yUnit, dxFit, dyFit, angles="xy", scale=vecScale*rScale*.8)
        plt.title("zernike fit directions terms=%i"%len(coef))

        xZernFit = xUnit + dxFit
        yZernFit = yUnit + dyFit

        xyZernFit = numpy.array([xZernFit, yZernFit]).T

        err = xyExpectUnit - xyZernFit

        print("RMS err post zern fit (micron)", numpy.sqrt(numpy.mean(err**2))*1000/rScale)

        plt.figure(figsize=(9,9))
        plt.quiver(xUnit, yUnit, err[:,0], err[:,0], angles="xy", scale=0.5*vecScale*rScale)
        plt.title("zernike fit residuals")
        plt.show()

        # plt.figure(figsize=(9,9))
        # plt.quiver(xUnit, yUnit, dxFit, dyFit, angles="xy", scale=vecScale*rScale)
        # plt.title("zernike fit residuals")
        # plt.show()

    # try the other method...
    else:
        # other zern type
        nPts = len(xUnit)
        dx = numpy.zeros((nPts, 20))
        dy = numpy.zeros((nPts, 20))
        t1 = time.time()
        for ii, (x, y) in enumerate(zip(xUnit, yUnit)):
            dxy = zerns(x,y)
            dx[ii, :] = dxy[:,0]
            dy[ii, :] = dxy[:,1]

        #standardize features
        # mdx = numpy.mean(dx,axis=0)
        # stdx = numpy.std(dx,axis=0)
        # dx = (dx - mdx)/stdx

        # mdy = numpy.mean(dy,axis=0)
        # stdy = numpy.std(dy,axis=0)
        # dy = (dy - mdy)/stdy

        dxy = numpy.vstack((dx,dy))

        # mdxy = numpy.mean(dxy, axis=0)
        # stdxy = numpy.std(dxy, axis=0)

        # dxy = (dxy - mdxy)/stdxy

        #standardize response
        # mxyErr = numpy.mean(xyErr)
        # stdErr = numpy.std(xyErr)
        # xyErr = (xyErr - mxyErr)/stdErr

        t1 = time.time()
        coef, resid, rank, s = numpy.linalg.lstsq(dxy, xyErr)
        print("zern fit took", time.time()-t1)

        plt.figure()
        plt.plot(numpy.arange(len(coef)), coef)
        plt.title("coeffs")
        plt.show()

        dxFit = dx @ coef
        dyFit = dy @ coef

        plt.figure(figsize=(9,9))
        plt.quiver(xUnit, yUnit, dxFit, dyFit, angles="xy", scale=vecScale*rScale)
        plt.title("zernike fit directions")

        xZernFit = xUnit + dxFit
        yZernFit = yUnit + dyFit

        xyZernFit = numpy.array([xZernFit, yZernFit]).T

        err = xyExpectUnit - xyZernFit

        # plt.figure()
        # errAng = numpy.degrees(numpy.arctan2(err[:,1], err[:,0]))
        # errMag = numpy.sqrt(err[:,1]**2+err[:,0]**2)
        # keep = errMag > numpy.mean(errMag)
        # errAng = errAng[keep]
        # plt.hist(errAng)
        # plt.title("error angle")
        # plt.show()

        print("RMS err post zern fit (micron)", numpy.sqrt(numpy.mean(err**2))*1000/rScale)


        plt.figure(figsize=(9,9))
        plt.quiver(xUnit, yUnit, err[:,0], err[:,1], angles="xy", scale=0.5*vecScale*rScale)
        plt.title("zernike fit residuals")
        plt.show()
        # import pdb; pdb.set_trace()


    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    # bestfile = "cam2_meandots_0.010.fits"
    # for file in glob.glob("*.fits"):
    for file in ["cam0_meandots_0.005.fits"]:
        print('on file', file)
        centroidSkimage(file, plot=False)
        associate(file)
        fit(file)
        print("\n\n\n")

    # xs = numpy.linspace(-0.5,0.5,10)
    # ys = xs[:]
    # dxy = zerns(xs[0],ys[0])
    # import pdb; pdb.set_trace()

    # fit()

    # nPts = 10000
    # maxOrder = 5
    # r = numpy.random.uniform(0,1,size=nPts)
    # t = numpy.random.uniform(0, numpy.pi*2, size=nPts)
    # x = numpy.sqrt(r)*numpy.cos(t)
    # y = numpy.sqrt(r)*numpy.sin(t)

    # z, dx, dy = zerns2(x,y,maxOrder)

    # for ii, (_dx, _dy) in enumerate(zip(dx, dy)):
    #     plt.figure(figsize=(9,9))
    #     plt.quiver(x, y, _dx, _dy, angles="xy", scale=100)
    #     plt.title("%i"%ii)
    #     plt.axis("equal")

    # plt.show()




    # maxOrd = 20 # orders start at 0
    # xs = numpy.zeros(50000) + 0.5
    # ys = numpy.zeros(50000) + 0.1
    # t1 = time.time()
    # out = zerns2(xs[0], ys[0], maxOrd)
    # print("took ", time.time()-t1)
    # print(out[1].shape, maxOrd)
    # # print(out[0])
    # print("\n\n")

    # out = zerns(xs[0], ys[0])

    # import pdb; pdb.set_trace()





"""
Pyguide summary:
found centroids 17470 in 1115.1218650341034
mediandots_0.001.fits mean error px 0.5733844822716175
found centroids 12457 in 842.449245929718
mediandots_0.006.fits mean error px 1.042507885494916
found centroids 17238 in 1210.1206440925598
mediandots_0.010.fits mean error px 0.8101053860166332
found centroids 16301 in 1116.7551679611206
mediandots_0.008.fits mean error px 0.8783760734513005
found centroids 17428 in 1225.8801662921906
mediandots_0.005.fits mean error px 0.5534137333519673
found centroids 17467 in 1193.7263021469116
mediandots_0.003.fits mean error px 0.5645482298225228
found centroids 12389 in 864.9349727630615
meandots_0.006.fits mean error px 1.0139918157579073
found centroids 17472 in 1181.9330270290375
meandots_0.001.fits mean error px 0.5752984206437881
found centroids 17471 in 1217.650855064392
meandots_0.003.fits mean error px 0.5643244498392236
found centroids 16101 in 1088.2285590171814
meandots_0.008.fits mean error px 1.0251576179161526
found centroids 17125 in 1199.1868779659271
meandots_0.005.fits mean error px 0.599100617192079
"""




# data = fitsio.read("bcam1-0001.fits")
# data = skimage.util.invert(data)
# data = data - numpy.min(data)
# data[data < 2500] = 0
# print(numpy.min(data), numpy.max(data))
# # data = (data-MaxCounts)*-1
# plt.figure(figsize=(8,8))
# plt.imshow(data, origin="lower")
# plt.show()

