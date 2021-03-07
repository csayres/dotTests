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

invmask[rowCen-1000:rowCen+1001,colCen-1000:colCen+1001] = 1
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
    data = data / numpy.max(data)
    return data


def centroidSkimage(img, plot=True):
    # unlike pyguide, now 0,0 is corner of LL pixel

    data = getImgData(img)

    if plot:
        plt.figure()
        imshow(data)
        plt.show()

    # data = data * invmask

    # data = (data - numpy.mean(data))/numpy.std(data)

    # data = data/numpy.max(data)


    # data = data[rowCen-150:rowCen+150, colCen-150:colCen+150] # center
    # data = data[2775:2775+150, 700:700+150] # top left corner
    # thresh = (data > numpy.percentile(data, 95))

    if plot:
        plt.figure()
        print("data stats", numpy.median(data), numpy.mean(data), numpy.std(data))
        plt.hist(data.flatten(), bins=100)

        plt.show()

    thresh = (data > 0.93)

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
        # _yCent, _xCent = region.weighted_centroid
        _yCent, _xCent = region.centroid
        _ecen = region.eccentricity
        _rad = region.equivalent_diameter/2
        xCent.append(_xCent)
        yCent.append(_yCent)
        ecen.append(_ecen)
        rad.append(_rad)

    centroidData = numpy.array([xCent, yCent, ecen, rad]).T


    ecenCut = centroidData[:,2] < 0.4
    centroidData = centroidData[ecenCut]


    if plot:
        plt.figure()
        plt.hist(centroidData[:, 2], bins=100)
        plt.title("ecen")

        plt.figure()
        plt.hist(centroidData[:, 3], bins=100)
        plt.title("rad")

        plt.show()

        t1 = time.time()
        plt.figure(figsize=(8,8))
        plt.title(img)
        imshow(sobel(data))  # extent is set correctly so centers line up
        for _x, _y, _e, _r in centroidData:
            plotCircle(_x, _y, _r)
        print("plotting took", time.time()-t1)
        plt.show()


    # save data
    numpy.savetxt(img + ".centroids", centroidData)




def associate(img):
    centFile = img+".centroids"
    if not os.path.exists(centFile):
        raise RuntimeError("must centroid data first")

    centroidData = numpy.loadtxt(centFile)

    # find the central dot
    x = centroidData[:, 0]
    y = centroidData[:, 1]

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
    # roughAngle = numpy.arctan2(dy,dx)

    print("rough scale", roughScale)

    # rotMat = numpy.array([
    #     [numpy.cos(roughAngle), -numpy.sin(roughAngle)],
    #     [numpy.sin(roughAngle), numpy.cos(roughAngle)]
    # ])

    # rotate and scale centroids, about central point
    # to put units into roughly mm and aligned with xy
    # calculate xy offset from xyMid to central pixel
    # even number of pixels, center of chip is at a pixel
    # corner
    # chipCenX = colCen - 0.5
    # chipCenY = rowCen - 0.5

    # offX = xMid - chipCenX
    # offY = yMid - chipCenY

    x = x - xMid
    y = y - yMid

    x *= roughScale
    y *= roughScale

    xy = numpy.array([x, y]).T

    # _xy = rotMat @ xy

    # fig, (ax1,ax2) = plt.subplots(2,1)
    # ax1.hist(xy[1], bins=1000)
    # ax1.set_xlim([-3.1, 3.1])
    # ax2.hist(_xy[1], bins=1000)
    # ax2.set_xlim([-3.1, 3.1])
    # plt.show()

    # xy = _xy.T

    xyAssoc = []
    centroidsSkipped = 0
    for x, y in xy:
        _rx = numpy.round(x)
        _ry = numpy.round(y)
        if numpy.abs(x-_rx) > 0.2:
            centroidsSkipped += 1
            continue
        if numpy.abs(y-_ry) > 0.2:
            centroidsSkipped += 1
            continue
        xyAssoc.append([x,y,_rx,_ry])

    print("skipped", centroidsSkipped, "centroids")
    xyAssoc = numpy.array(xyAssoc)

    ## shift origin to center of CCD
    xPixCCDCen = colCen - 0.5
    yPixCCDCen = rowCen - 0.5


    xCen = xPixCCDCen - xMid
    yCen = yPixCCDCen - yMid

    xCen *= roughScale
    yCen *= roughScale

    xyAssoc[:, 0] -= xCen
    xyAssoc[:, 2] -= xCen
    xyAssoc[:, 1] -= yCen
    xyAssoc[:, 3] -= yCen

    # put things back in pixels
    # we'll do a full fit against pixels

    xyAssoc[:,0] /= roughScale
    xyAssoc[:,1] /= roughScale

    plt.figure()
    plt.plot(xyAssoc[:,0], xyAssoc[:,1], 'or')
    plt.plot(xyAssoc[:,2], xyAssoc[:,3], 'ob')
    plt.show()

    # err = xyAssoc[:,2:] - xyAssoc[:,:2]
    # plt.figure(figsize=(9,9))
    # plt.quiver(xyAssoc[:,0], xyAssoc[:,1], err[:,0], err[:,1], scale=1.5/roughScale)
    # plt.show()

    # sort xy by
    numpy.savetxt(img+".assoc", xyAssoc)


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
    xyMeas = data[:,:2] # pixels
    xyExpect = data[:,2:]   # mm

    # meanXY = numpy.mean(xyMeas, axis=0)
    # # meanXY = numpy.array([65, -35])  # roughly center of divergence?
    # xyMeas = xyMeas - meanXY
    # xyExpect = xyExpect - meanXY

    # # use the plot to find the center of divergence...
    # err = xyExpect - xyMeas
    # magErr = numpy.linalg.norm(err, axis=1)

    # # throw out outliers
    # keep = magErr*1000 < 190
    # xyMeas = xyMeas[keep]
    # print("threw out", len(xyExpect) - len(xyMeas))
    # xyExpect = xyExpect[keep]
    # err = err[keep]

    # make it circular
    _r = numpy.linalg.norm(xyMeas, axis=1)
    maxRad = numpy.max(xyMeas[:,1])*0.8  # max y value
    keep = _r <= maxRad
    xyMeas = xyMeas[keep]
    xyExpect = xyExpect[keep]

    # plt.figure(figsize=(9,9))
    # plt.quiver(xyMeas[:,0], xyMeas[:,1], err[:,0], err[:,1], angles="xy", scale=2/0.02)
    # plt.axis("equal")
    # plt.title("raw errors")
    # plt.show()

    # xMid = 81.5
    # yMid = -72.5

    # xyMeas = xyMeas - numpy.array([xMid, yMid])
    # xyExpect = xyExpect - numpy.array([xMid, yMid])

    # print("RMS err prefit (micron)", numpy.sqrt(numpy.mean(err**2))*1000)


    if True:
        # model = AffineTransform()
        # model = EuclideanTransform()
        model = SimilarityTransform()
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
        nRadTerms = 6
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
        plt.quiver(xUnit, yUnit, err[:,0], err[:,1], angles="xy", scale=0.5*vecScale*rScale)
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
    # for file in glob.glob("cam0*.fits"):
    for file in ["cam0_meandots_0.010.fits"]:
        print('on file', file)
        centroidSkimage(file, plot=False)
        associate(file)
        fit(file)
        print("\n\n\n")

